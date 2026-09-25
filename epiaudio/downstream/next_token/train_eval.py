import os
import warnings
from itertools import chain
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
import tqdm
from epiaudio.downstream.next_token.audio_token_dataset import AudioTokenDataset
from epiaudio.downstream.next_token.audio_token_predictor import AudioTokenPredictor
from epiaudio.train_torch import _find_free_port
from epiaudio.dataset.audio_tokenizers import Tokenizer
from epiaudio.dataset.prepare_audio import TOKENIZERS

DEFAULT_SEED = 42

def flatten(x):
    for item in x:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def init_process(rank: int, world_size: int) -> tuple[bool, str]:
    dist.init_process_group(rank=rank, world_size=world_size)
    cuda = torch.cuda.is_available()
    device = str(torch.device(f"cuda:{rank}")) if cuda else "cpu"

    return cuda, device

def train(
    rank: int,
    world_size: int,
    model: AudioTokenPredictor,
    n_epochs: int,
    batch_size: int,
    data: AudioTokenDataset,
    pretrained_checkpoint_path: str,
    ddp_checkpoint_path: str,
    finetune: bool=False,
    zero_shot: bool=False
) -> None:
    cuda, device = init_process(rank, world_size)

    sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, sampler=sampler)

    model = model.to(device)
    model.train()
    wrapped_model = DDP(model, device_ids=[rank]) if cuda else model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(wrapped_model.parameters())

    disable = rank != 0

    n_epochs = 0 if zero_shot else n_epochs
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(loader, desc=f"epoch {epoch + 1}/{n_epochs}", disable=disable)
        for tokens in pbar:
            tokens = tokens.to(device)
            optimizer.zero_grad()
            _, logits = wrapped_model(tokens)
            loss = criterion(logits[:, :-1, :].transpose(1, 2), tokens[:, 1:])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    if rank == 0:
        if cuda:
            assert isinstance(wrapped_model, DDP)
            model_to_save = wrapped_model.module
        else:
            model_to_save = wrapped_model
        assert isinstance(model_to_save, AudioTokenPredictor)

        if cuda:
            torch.save(model_to_save.state_dict(), ddp_checkpoint_path)
        if not finetune and not zero_shot:
            torch.save(model_to_save.state_dict(), pretrained_checkpoint_path) # save model to this path only if we are pretraining

    dist.destroy_process_group()

def evaluate(
    rank: int,
    world_size: int,
    model: AudioTokenPredictor,
    batch_size: int,
    data: AudioTokenDataset,
    num_prefix_tokens: int,
    num_pred_tokens: int,
    tokenizer_name: str,
    queue: Queue
) -> None:
    _, device = init_process(rank, world_size)

    sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, sampler=sampler)

    model = model.to(device)
    model.eval()

    try:
        tokenizer = TOKENIZERS[tokenizer_name]["cls"](device=device)
    except KeyError:
        tokenizer = Tokenizer()
        warnings.warn(f"Tokenizer \"{tokenizer_name}\" could not be found.")

    criterion = nn.CrossEntropyLoss()
    losses, mean_accuracies, mean_encoder_dists, hidden_state_cos_sims = [], [], [], []

    end_idx = num_prefix_tokens + num_pred_tokens
    pbar = tqdm.tqdm(loader, desc="Generating predicted tokens")

    for tokens in pbar:
        tokens = tokens.to(device)
        tokens_actual = tokens[:, num_prefix_tokens:num_prefix_tokens + num_pred_tokens]

        eval_output = model.generate(
            tokens[:, :end_idx],
            num_prefix_tokens,
            num_pred_tokens,
            device=device
        )

        with torch.no_grad():
            hidden_state_pred, logits, tokens_pred = (tensor.to(device) for tensor in eval_output)
            hidden_state_pred = hidden_state_pred[:, end_idx - 1, :]
            hidden_state_actual, _ = model.forward(tokens)
            hidden_state_actual = hidden_state_actual[:, end_idx - 1, :]

            mean_acc = torch.mean((tokens_actual == tokens_pred).float(), dim=1).tolist()
            mean_accuracies.extend(mean_acc)

            quantized_vectors_actual = tokenizer.decode(tokens_actual)
            quantized_vectors_pred = tokenizer.decode(tokens_pred)

            loss = criterion(logits, tokens_actual).item()
            losses.append(loss)

            if quantized_vectors_actual.numel() > 0:
                diff = quantized_vectors_actual - quantized_vectors_pred
                dists = torch.linalg.vector_norm(diff, dim=1)
                mean_dists = torch.mean(dists, dim=1).tolist()
                mean_encoder_dists.extend(mean_dists)

            cos_sims = cosine_similarity(hidden_state_actual, hidden_state_pred, dim=1).tolist()
            hidden_state_cos_sims.extend(cos_sims)

    value_lists = [
        losses,
        mean_accuracies,
        mean_encoder_dists,
        hidden_state_cos_sims
    ]
    queue.put((rank, value_lists))

    dist.destroy_process_group()

def calculate_metrics(value_lists: list[list[float]]) -> dict[str, float | None]:
    ce_losses, mean_accuracies, mean_encoder_dists, hidden_state_cos_sims = value_lists
    metrics = dict()

    metrics["mean_ce_loss"] = np.mean(ce_losses)
    metrics["mean_accuracy"] = np.mean(mean_accuracies)
    mean_encoder_dist = np.mean(mean_encoder_dists) if mean_encoder_dists else None
    metrics["mean_avg_encoder_embedding_dist"] = mean_encoder_dist
    metrics["mean_hidden_state_cos_sim"] = np.mean(hidden_state_cos_sims)

    return metrics

def train_and_evaluate(
    cfg: DictConfig,
    num_prefix_tokens: int,
    num_pred_tokens: int,
    seed: int=DEFAULT_SEED,
    finetune: bool=False
) -> dict[str, float | None]:
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    ds_path = cfg.ds_path
    train_path = os.path.join(ds_path, "train.bin")
    test_path = os.path.join(ds_path, "test.bin")
    metadata_path = os.path.join(ds_path, "metadata.json")
    pretrained_checkpoint_path = cfg.pretrained_checkpoint_path
    ddp_checkpoint_path = cfg.ddp_checkpoint_path

    n_epochs = cfg.n_finetune_epochs if finetune else cfg.n_pretrain_epochs
    batch_size = cfg.batch_size
    num_layers = cfg.num_layers
    embed_dim = cfg.embed_dim
    per_head_dim = cfg.per_head_dim
    vocab_size = cfg.vocab_size
    seq_length = cfg.seq_length
    num_prefix_tokens = cfg.num_prefix_tokens
    num_pred_tokens = cfg.num_pred_tokens
    tokenizer_name = cfg.tokenizer_name
    zero_shot = cfg.zero_shot

    ds_train = AudioTokenDataset("train", train_path, metadata_path)
    ds_test = AudioTokenDataset("test", test_path, metadata_path)

    model = AudioTokenPredictor(
        num_layers=num_layers,
        embed_dim=embed_dim,
        per_head_dim=per_head_dim,
        vocab_size=vocab_size,
        seq_length=seq_length
    )

    if finetune or zero_shot:
        model.load_state_dict(torch.load(pretrained_checkpoint_path))

    port = _find_free_port()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    world_size = torch.cuda.device_count() if cuda else 1
    args = (
        world_size,
        model,
        n_epochs,
        batch_size,
        ds_train,
        pretrained_checkpoint_path,
        ddp_checkpoint_path,
        finetune,
        zero_shot
    )
    spawn(train, args=args, nprocs=world_size)
    if cuda:
        model.load_state_dict(torch.load(ddp_checkpoint_path))

    manager = mp.Manager()
    queue = manager.Queue()
    args = (
        world_size,
        model,
        batch_size,
        ds_test,
        num_prefix_tokens,
        num_pred_tokens,
        tokenizer_name,
        queue
    )
    spawn(evaluate, args=args, nprocs=world_size)
    results = [queue.get()[1] for _ in range(world_size)]
    value_lists = [list(chain.from_iterable(vals)) for vals in zip(*results)]
    metrics = calculate_metrics(value_lists)

    return metrics

import os
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, precision_score, f1_score
import tqdm
from epiaudio.downstream.audio_classification_dataset import AudioClassificationDataset
from epiaudio.downstream.audio_classifier import AudioClassifier
from epiaudio.train_torch import _find_free_port

DEFAULT_SEED = 42

def train(
    rank: int,
    world_size: int,
    model: AudioClassifier,
    n_epochs: int,
    batch_size: int,
    data: AudioClassificationDataset,
    checkpoint_path: str
) -> None:
    dist.init_process_group(rank=rank, world_size=world_size)
    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{rank}") if cuda else "cpu"

    sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, sampler=sampler)

    model = model.to(device)
    model.train()
    wrapped_model = DDP(model, device_ids=[rank]) if cuda else model

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(data.class_weights).to(device))
    optimizer = optim.Adam(wrapped_model.parameters())

    disable = rank != 0

    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)
        pbar = tqdm.tqdm(loader, desc=f"epoch {epoch + 1}/{n_epochs}", disable=disable)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = wrapped_model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    if rank == 0 and cuda:
        assert isinstance(wrapped_model, DDP)

        torch.save(wrapped_model.module.state_dict(), checkpoint_path)
    dist.destroy_process_group()

def evaluate(model: AudioClassifier, data: DataLoader) -> dict[str, float]:
    model.eval()
    y_true, y_pred = [], []

    for x, y in data:
        y_true.append(y)
        y_pred.append(model(x).argmax(dim=1))

    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_avg_precision": precision_score(y_true, y_pred, average="weighted"),
        "weighted_avg_f1_score": f1_score(y_true, y_pred, average="weighted")
    }

    return metrics

def train_and_evaluate(cfg: DictConfig, seed: int=DEFAULT_SEED) -> dict[str, float]:
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    ds_path = cfg.ds_path
    train_path = os.path.join(ds_path, "train.bin")
    test_path = os.path.join(ds_path, "test.bin")
    metadata_path = os.path.join(ds_path, "metadata.json")
    checkpoint_path = os.path.join(ds_path, "model_checkpoint.pt")

    n_epochs = cfg.n_epochs
    batch_size = cfg.batch_size
    num_layers = cfg.num_layers
    embed_dim = cfg.embed_dim
    per_head_dim = cfg.per_head_dim
    vocab_size = cfg.vocab_size
    seq_length = cfg.seq_length

    ds_train = AudioClassificationDataset("train", train_path, metadata_path)
    ds_test = AudioClassificationDataset("test", test_path, metadata_path)
    test_dataloader = DataLoader(ds_test, batch_size=batch_size)

    model = AudioClassifier(
        num_layers=num_layers,
        embed_dim=embed_dim,
        per_head_dim=per_head_dim,
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_classes=ds_train.num_classes
    )

    port = _find_free_port()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    world_size = torch.cuda.device_count() if cuda else 1
    spawn(train, args=(world_size, model, n_epochs, batch_size, ds_train, checkpoint_path), nprocs=world_size)

    if cuda:
        model.load_state_dict(torch.load(checkpoint_path))
    metrics = evaluate(model=model, data=test_dataloader)

    return metrics

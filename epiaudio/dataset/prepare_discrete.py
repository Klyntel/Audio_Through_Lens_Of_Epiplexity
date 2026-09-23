"""Prepare discrete non-audio corpora for EpiAudio PyTorch sweeps.

The token contracts intentionally mirror the reference scripts in
``epiplexity/picodo/dataset`` while emitting EpiAudio's metadata.yaml and
int16 memmaps.  Keeping int16 avoids the dtype ambiguity in the PyTorch data
loader for vocabularies that fit within signed 16-bit integers.
"""

from __future__ import annotations

import argparse
import pickle
import random
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

from experiments.bootstrap_modalities.prepare import TokenDatasetContract, TokenDatasetWriter


TOKEN_DTYPE = np.dtype(np.int16)
OPENWEBTEXT_ALPHABET = "\n" + "".join(chr(code) for code in range(32, 127))
LICHESS_TOKENS = [
    "<BOS>", "<EOS>", ",", ";", "+", "#", "=", "?", "!", "\n", " ", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "x", "a", "b",
    "c", "d", "e", "f", "g", "h", "K", "Q", "R", "B", "N", "O", "P", "(",
    ")", "l", "k", "v", ":", "*", "|", "r", "n", "q", "p", "w", "m",
]
LICHESS_CONTEXT_LENGTH = 512
OPENWEBTEXT_STOI = {character: index for index, character in enumerate(OPENWEBTEXT_ALPHABET)}
LICHESS_STOI = {token: index for index, token in enumerate(LICHESS_TOKENS)}
LICHESS_ITOS = {index: token for token, index in LICHESS_STOI.items()}


def _rounded_vocab_size(size: int) -> int:
    """Match picodo's multiple-of-32 embedding vocabulary rounding."""
    return ((size + 31) // 32) * 32


@dataclass(frozen=True)
class OpenWebTextTokenizer:
    """Reference printable-ASCII character tokenizer for OpenWebText."""

    name: str = "ascii96"
    alphabet: str = OPENWEBTEXT_ALPHABET

    @property
    def vocab_size(self) -> int:
        return _rounded_vocab_size(len(self.alphabet))

    @property
    def sequence_length(self) -> int:
        return 512

    def supports(self, text: str) -> bool:
        return all(character in OPENWEBTEXT_STOI for character in text)

    def encode_document(self, text: str) -> np.ndarray:
        if not self.supports(text):
            raise ValueError("OpenWebText document contains a character outside printable ASCII.")
        return np.asarray(
            [*(OPENWEBTEXT_STOI[character] for character in text), OPENWEBTEXT_STOI["\n"]],
            dtype=TOKEN_DTYPE,
        )

    def decode(self, token_ids: Sequence[int] | np.ndarray[Any, Any]) -> str:
        return "".join(OPENWEBTEXT_ALPHABET[int(token)] for token in token_ids)


@dataclass(frozen=True)
class LichessPuzzleTokenizer:
    """Reference fixed-context Lichess puzzle character tokenizer."""

    name: str = "puzzles"
    sequence_length: int = LICHESS_CONTEXT_LENGTH

    @property
    def vocab_size(self) -> int:
        return _rounded_vocab_size(len(LICHESS_TOKENS))

    @property
    def bos_id(self) -> int:
        return LICHESS_STOI["<BOS>"]

    @property
    def eos_id(self) -> int:
        return LICHESS_STOI["<EOS>"]

    def format_moves(self, context: str, target: str) -> str:
        text = context + target
        text = re.sub(r"\{[^}]*\}", "", re.sub(r"\d+\.\.?\.?\s*", "", text))
        text = re.sub(r"\s+", " ", text).strip()
        moves = text.split()
        formatted = "".join(
            f"{move}{',' if index % 2 == 0 else ';'}"
            for index, move in enumerate(moves)
        )
        return formatted[:-1] if formatted else ""

    def fits_context(self, context: str, target: str) -> bool:
        """Match the reference preparer's filtering of overlength puzzles."""
        return len(self.format_moves(context, target)) + 2 <= self.sequence_length

    def encode_example(self, context: str, target: str) -> tuple[np.ndarray, np.ndarray]:
        formatted = self.format_moves(context, target)
        unknown = sorted(set(formatted) - set(LICHESS_STOI))
        if unknown:
            raise ValueError(f"Lichess puzzle contains unsupported characters: {unknown!r}")
        content = [LICHESS_STOI[character] for character in formatted]
        if len(content) + 2 > self.sequence_length:
            raise ValueError(
                f"Lichess puzzle has {len(content) + 2} tokens, exceeding "
                f"the {self.sequence_length}-token reference context."
            )
        token_ids = np.full(self.sequence_length, self.eos_id, dtype=TOKEN_DTYPE)
        token_ids[0] = self.bos_id
        token_ids[1 : 1 + len(content)] = content
        eos_position = 1 + len(content)
        token_ids[eos_position] = self.eos_id

        target_start = max(formatted.rfind(";"), formatted.rfind(",")) + 2
        target_mask = np.zeros(self.sequence_length, dtype=np.bool_)
        target_mask[target_start:eos_position] = True
        return token_ids, target_mask

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(LICHESS_ITOS[int(token)] for token in token_ids)


@dataclass(frozen=True)
class Cifar5MGrayscaleTokenizer:
    """Reference CIFAR-5M RGB-to-grayscale 32x32 tokenizer."""

    name: str = "grayscale"
    vocab_size: int = 256
    sequence_length: int = 32 * 32

    def encode_batch(self, rgb_batch: np.ndarray) -> np.ndarray:
        if rgb_batch.ndim != 4 or rgb_batch.shape[1:] != (32, 32, 3):
            raise ValueError(
                "CIFAR-5M images must have shape [batch, 32, 32, 3], got "
                f"{tuple(rgb_batch.shape)}."
            )
        grayscale = rgb_batch.mean(axis=-1).astype(TOKEN_DTYPE)
        return grayscale.reshape(rgb_batch.shape[0], self.sequence_length)

    def decode(self, token_ids: np.ndarray) -> np.ndarray:
        tokens = np.asarray(token_ids, dtype=TOKEN_DTYPE)
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        if tokens.ndim != 2 or tokens.shape[1] != self.sequence_length:
            raise ValueError(
                f"CIFAR-5M decode expects [batch, {self.sequence_length}] tokens, "
                f"got {tuple(tokens.shape)}."
            )
        return tokens.reshape(-1, 32, 32)


def _dataset_token_chunks(dataset: Dataset, *, batch_size: int = 8192) -> Iterator[np.ndarray]:
    for batch in dataset.iter(batch_size=batch_size):
        rows = cast(Mapping[str, Sequence[Sequence[int]]], batch)
        yield np.concatenate([np.asarray(ids, dtype=TOKEN_DTYPE) for ids in rows["ids"]])


def prepare_openwebtext(
    output_dir: Path,
    *,
    dataset_id: str = "dylanebert/openwebtext",
    revision: str | None = None,
    test_fraction: float = 0.0005,
    seed: int = 2357,
    num_proc: int | None = None,
) -> None:
    """Prepare the reference printable-ASCII OpenWebText token stream."""
    tokenizer = OpenWebTextTokenizer()
    writer = TokenDatasetWriter(
        output_dir,
        TokenDatasetContract(
            dataset="openwebtext",
            tokenizer=tokenizer.name,
            vocab_size=tokenizer.vocab_size,
            sequence_length=tokenizer.sequence_length,
            source={"dataset_id": dataset_id, "revision": revision, "seed": seed, "test_fraction": test_fraction},
        ),
    )
    dataset = load_dataset(dataset_id, revision=revision)["train"]
    split = dataset.train_test_split(test_size=test_fraction, seed=seed, shuffle=True)

    def supported(example: dict[str, str]) -> bool:
        return tokenizer.supports(example["text"])

    def encode(example: dict[str, str]) -> dict[str, Any]:
        ids = tokenizer.encode_document(example["text"])
        return {"ids": ids.tolist(), "length": len(ids)}

    prepared: dict[str, Dataset] = {}
    for split_name, rows in split.items():
        split_name = str(split_name)
        filtered = rows.filter(supported, num_proc=num_proc, desc=f"filtering {split_name}")
        prepared[split_name] = filtered.map(
            encode,
            remove_columns=filtered.column_names,
            num_proc=num_proc,
            desc=f"tokenizing {split_name}",
        )

    train_tokens = int(np.sum(prepared["train"]["length"], dtype=np.int64))
    test_tokens = int(np.sum(prepared["test"]["length"], dtype=np.int64))
    for split_name, num_tokens in (("train", train_tokens), ("test", test_tokens)):
        writer.write_split(
            split_name,
            _dataset_token_chunks(prepared[split_name]),
            num_tokens=num_tokens,
            num_samples=len(prepared[split_name]),
        )
    writer.finalize()


def prepare_lichess_puzzles(
    output_dir: Path,
    *,
    dataset_id: str = "EleutherAI/lichess-puzzles",
    revision: str | None = None,
    test_size: int = 16_384,
    seed: int = 42,
    num_proc: int | None = None,
) -> None:
    """Prepare fixed 512-token Lichess puzzle rows and target masks."""
    tokenizer = LichessPuzzleTokenizer()
    writer = TokenDatasetWriter(
        output_dir,
        TokenDatasetContract(
            dataset="lichess_puzzles",
            tokenizer=tokenizer.name,
            vocab_size=tokenizer.vocab_size,
            sequence_length=tokenizer.sequence_length,
            source={"dataset_id": dataset_id, "revision": revision, "seed": seed, "test_size": test_size},
        ),
    )
    rows = load_dataset(dataset_id, revision=revision, split="train", num_proc=num_proc)
    rows = rows.filter(
        lambda row: tokenizer.fits_context(row["ctx"], row["target"]),
        num_proc=num_proc,
        desc="filtering overlength puzzles",
    )
    if len(rows) <= test_size:
        raise ValueError(f"Lichess dataset has {len(rows)} rows, not enough for test_size={test_size}.")
    rows = rows.shuffle(seed=seed)
    split_rows = {"train": rows.select(range(len(rows) - test_size)), "test": rows.select(range(len(rows) - test_size, len(rows)))}

    for split_name, split in split_rows.items():
        def encoded_rows() -> Iterator[tuple[np.ndarray, dict[str, np.ndarray]]]:
            for row in tqdm(split, desc=f"tokenizing {split_name}"):
                puzzle = cast(Mapping[str, str], row)
                token_ids, target_mask = tokenizer.encode_example(puzzle["ctx"], puzzle["target"])
                yield token_ids, {"mask": target_mask}

        writer.write_split(
            split_name,
            encoded_rows(),
            num_tokens=len(split) * tokenizer.sequence_length,
            num_samples=len(split),
            sidecars={"mask": np.bool_},
        )
    writer.finalize()
    with (output_dir / "meta.pkl").open("wb") as file:
        pickle.dump(
            {"vocab_size": len(LICHESS_TOKENS), "seed": seed, "ctx_len": tokenizer.sequence_length}, file
        )


def prepare_cifar5m(
    output_dir: Path,
    shard_paths: Sequence[Path],
    *,
    test_images: int = 2_000,
    seed: int = 2357,
    batch_size: int = 10_000,
) -> None:
    """Prepare reference CIFAR-5M grayscale image token streams from NPZ shards."""
    if not shard_paths:
        raise ValueError("At least one CIFAR-5M --shard path is required.")
    missing = [path for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CIFAR-5M shard(s) not found: {missing}")
    tokenizer = Cifar5MGrayscaleTokenizer()
    writer = TokenDatasetWriter(
        output_dir,
        TokenDatasetContract(
            dataset="cifar5m",
            tokenizer=tokenizer.name,
            vocab_size=tokenizer.vocab_size,
            sequence_length=tokenizer.sequence_length,
            source={"shards": [str(path) for path in shard_paths], "seed": seed, "test_images": test_images},
        ),
    )
    image_counts: list[int] = []
    for shard_path in shard_paths:
        with np.load(shard_path, mmap_mode="r") as shard:
            if "X" not in shard:
                raise ValueError(f"CIFAR-5M shard {shard_path} does not contain an 'X' image array.")
            image_counts.append(int(shard["X"].shape[0]))
    total_images = sum(image_counts)
    if total_images <= test_images:
        raise ValueError(f"CIFAR-5M has {total_images} images, not enough for test_images={test_images}.")
    test_indices = set(random.Random(seed).sample(range(total_images), k=test_images))

    split_images = {"train": total_images - test_images, "test": test_images}
    def tokenized_split_batches() -> Iterator[dict[str, np.ndarray]]:
        global_index = 0
        for shard_path in shard_paths:
            with np.load(shard_path, mmap_mode="r") as shard:
                images = shard["X"]
                for start in tqdm(range(0, len(images), batch_size), desc=f"tokenizing {shard_path.name}"):
                    tokens = tokenizer.encode_batch(images[start : start + batch_size])
                    global_indices = np.arange(global_index, global_index + len(tokens))
                    test_selection = np.fromiter(
                        (index in test_indices for index in global_indices), dtype=bool, count=len(tokens)
                    )
                    yield {"train": tokens[~test_selection], "test": tokens[test_selection]}
                    global_index += len(tokens)

    writer.write_splits(
        tokenized_split_batches(),
        splits={
            split_name: (image_count * tokenizer.sequence_length, image_count)
            for split_name, image_count in split_images.items()
        },
    )
    writer.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    openwebtext = subparsers.add_parser("openwebtext", help="Prepare printable-ASCII OpenWebText.")
    openwebtext.add_argument("--output", type=Path, default=Path("data/openwebtext_ascii96"))
    openwebtext.add_argument("--dataset-id", default="dylanebert/openwebtext")
    openwebtext.add_argument("--revision", default=None)
    openwebtext.add_argument("--test-fraction", type=float, default=0.0005)
    openwebtext.add_argument("--seed", type=int, default=2357)
    openwebtext.add_argument("--num-proc", type=int, default=None)

    lichess = subparsers.add_parser("lichess", help="Prepare fixed-context Lichess puzzles.")
    lichess.add_argument("--output", type=Path, default=Path("data/lichess_puzzles"))
    lichess.add_argument("--dataset-id", default="EleutherAI/lichess-puzzles")
    lichess.add_argument("--revision", default=None)
    lichess.add_argument("--test-size", type=int, default=16_384)
    lichess.add_argument("--seed", type=int, default=42)
    lichess.add_argument("--num-proc", type=int, default=None)

    cifar5m = subparsers.add_parser("cifar5m", help="Prepare grayscale CIFAR-5M shards.")
    cifar5m.add_argument("--output", type=Path, default=Path("data/cifar5m_grayscale"))
    cifar5m.add_argument("--shard", type=Path, action="append", required=True)
    cifar5m.add_argument("--test-images", type=int, default=2_000)
    cifar5m.add_argument("--seed", type=int, default=2357)
    cifar5m.add_argument("--batch-size", type=int, default=10_000)

    args = parser.parse_args()
    if args.dataset == "openwebtext":
        prepare_openwebtext(
            args.output,
            dataset_id=args.dataset_id,
            revision=args.revision,
            test_fraction=args.test_fraction,
            seed=args.seed,
            num_proc=args.num_proc,
        )
    elif args.dataset == "lichess":
        prepare_lichess_puzzles(
            args.output,
            dataset_id=args.dataset_id,
            revision=args.revision,
            test_size=args.test_size,
            seed=args.seed,
            num_proc=args.num_proc,
        )
    else:
        prepare_cifar5m(
            args.output,
            args.shard,
            test_images=args.test_images,
            seed=args.seed,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()

import random
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio
from epiaudio.dataset.zenodo_downloader import download_zenodo
from audio_preprocessing.datasets import AudioDataset

def create_dataset(input_path: Path, split_ratios: list[float], seed: int=42) -> AudioDataset:
    """
        Turns the audio files in the downloaded DEMAND dataset into an AudioDataset.
        input_path is the path to the downloaded DEMAND files.
        split_ratios is list of 2 floats: [training set ratio, validation set ratio]
        Splits each folder into the train/validation/evaluation sets according to the above ratios.
    """
    category_labels = {
        "D": "Domestic",
        "N": "Nature",
        "O": "Office",
        "P": "Public",
        "S": "Street",
        "T": "Transportation"
    }
    random.seed(seed)

    train_records, val_records, eval_records = [], [], []

    for folder in input_path.iterdir():
        if not folder.is_dir():
            continue
        for subfolder in folder.iterdir():
            if not subfolder.is_dir():
                continue
            audio_files = [f for f in subfolder.iterdir() if f.is_file()]
            random.shuffle(audio_files)

            train_ratio, val_ratio = split_ratios
            train_cutoff = int(train_ratio*len(audio_files))
            val_cutoff = train_cutoff + int(val_ratio*len(audio_files))
            label = category_labels[subfolder.name[0]]

            for i, file in enumerate(audio_files):
                if i < train_cutoff:
                    records = train_records
                elif i < val_cutoff:
                    records = val_records
                else:
                    records = eval_records
                records.append({"audio": str(file), "label": label})

    train_ds = Dataset.from_list(train_records).cast_column("audio", Audio())
    val_ds = Dataset.from_list(val_records).cast_column("audio", Audio())
    eval_ds = Dataset.from_list(eval_records).cast_column("audio", Audio())

    ds = AudioDataset(data=DatasetDict({
        "train": train_ds,
        "valid": val_ds,
        "eval": eval_ds
    }))

    return ds

record_id = "1227121"
download_zenodo(record_id, do_not_download=["scripts"])

input_path = Path.cwd() / "data" / f"zenodo_{record_id}"
demand = create_dataset(input_path, [.8, .1])
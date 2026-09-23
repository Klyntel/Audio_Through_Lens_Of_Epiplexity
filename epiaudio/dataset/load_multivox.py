import random
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio, Value
from epiaudio.dataset.zenodo_downloader import download_zenodo
import pandas as pd
from audio_preprocessing.datasets import AudioDataset

PART1_RECORD_ID = "17058101"
PART2_RECORD_ID = "17065497"

def load_multivox(
    split_ratios: list[float] | None=None,
    dataset_path: str="",
    download: bool=True,
    seed: int=42
) -> AudioDataset:
    """
        Turns the audio files in the downloaded MultiVox dataset into an AudioDataset.
        Will download the dataset if download is True.
        dataset_path is the path to the MultiVox dataset.
        split_ratios is list of 2 floats: [training set ratio, validation set ratio]
        Splits each folder into the train/validation/evaluation sets according to the above ratios.
    """
    dataset_path = dataset_path if dataset_path else f"data/zenodo_{PART1_RECORD_ID}+{PART2_RECORD_ID}"
    if download:
        download_zenodo(
            PART1_RECORD_ID,
            output_dir=dataset_path,
            do_not_download=["MULTIVOX_extended_dataset_description_and_supplement", "README"]
        )
        download_zenodo(PART2_RECORD_ID, output_dir=dataset_path)

    random.seed(seed)
    split_ratios = split_ratios if split_ratios else [.8, .1]
    records = {"audio": [], "Path": [], "audio_capture_type": []}
    dataset_dir = Path(dataset_path)

    for folder in dataset_dir.iterdir():
        if not folder.is_dir():
            continue
        for subfolder in folder.iterdir():
            if not subfolder.is_dir():
                continue
            for subsubfolder in subfolder.iterdir():
                if not subsubfolder.is_dir():
                    continue
                path = subsubfolder.name
                for file in subsubfolder.iterdir():
                    if not (file.is_file() and file.suffix == ".wav"):
                        continue
                    audio_capture_type = file.name.split("_Song")[0]
                    records["audio"].append(str(file))
                    records["Path"].append(path)
                    records["audio_capture_type"].append(audio_capture_type)

    df = pd.DataFrame.from_dict(records)
    metadata = pd.read_csv(str(dataset_dir / "metadata.csv"))
    df = pd.merge(df, metadata, how="left", on="Path")

    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("audio", Value("string"))
    ds = ds.cast_column("audio", Audio())

    val_eval_size = 1 - split_ratios[0]
    split1 = ds.train_test_split(test_size=val_eval_size)
    relative_test_size = (1 - sum(split_ratios)) / val_eval_size
    split2 = split1["test"].train_test_split(test_size=relative_test_size)

    ds = AudioDataset(data=DatasetDict({
        "train": split1["train"],
        "val": split2["train"],
        "eval": split2["test"],
    }))

    return ds
load_multivox(download=False)
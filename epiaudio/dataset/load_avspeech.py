import requests
from datasets import Dataset, DatasetDict, Audio
from audio_preprocessing.datasets import AudioDataset
from datasets import load_dataset

DATASET_NAME = "ProgramComputer/avspeech-visual-audio"
DEFAULT_COLUMNS = ["clip_id", "avspeech_metadata", "audio"]
DEFAULT_SPLITS = ["train", "test"]
DEFAULT_NUM_ROWS = {split: -1 for split in DEFAULT_SPLITS}

def stream_avspeech(split: str, columns: list[str] | None=None, num_rows: int=-1) -> Dataset:
    """
        Streams the AVSpeech dataset and returns the resulting partial dataset.
        Can specify the splits and columns you want (to avoid the video data), and the number of rows.
        If num_rows < 0 then the entire dataset will be streamed.
    """
    resp = requests.get(
        "https://datasets-server.huggingface.co/size",
        params={"dataset": DATASET_NAME},
    ).json()
    total_rows = 0
    for metadata in resp["size"]["splits"]:
        if metadata["split"] == split:
            total_rows = metadata["num_rows"]
            break
    num_rows = min(num_rows, total_rows) if num_rows >= 0 else total_rows

    columns = columns if columns else DEFAULT_COLUMNS
    dataset = load_dataset(DATASET_NAME, split=split, columns=columns, streaming=True)
    records = list(dataset.take(num_rows))
    ds = Dataset.from_list(records).cast_column("audio", Audio())

    return ds

def load_avspeech(
    splits: list[str] | None=None,
    columns: list[str] | None=None,
    num_rows: dict[str, int] | None=None
) -> AudioDataset:
    """
        Returns an AudioDataset wrapping the AVSpeech dataset.
        Can specify splits and columns and number of rows in each split.
    """
    splits = splits if splits else DEFAULT_SPLITS
    columns = columns if columns else DEFAULT_COLUMNS
    num_rows = num_rows if num_rows else DEFAULT_NUM_ROWS

    data = DatasetDict({split: stream_avspeech(split, columns=columns, num_rows=num_rows[split]) for split in splits})
    ds = AudioDataset(data=data)

    return ds
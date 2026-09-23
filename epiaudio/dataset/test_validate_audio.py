from datasets import Dataset, DatasetDict

from audio_preprocessing.datasets import AudioDataset
from epiaudio.dataset.validate_audio import validate_audio


def test_embedded_audio_dataset_skips_path_validation() -> None:
    dataset = AudioDataset(
        data=DatasetDict(
            {
                "train": Dataset.from_dict({"audio": [None], "id": [1]}),
                "test": Dataset.from_dict({"audio": [None], "id": [2]}),
            }
        )
    )

    cleaned = validate_audio(dataset)

    assert len(cleaned.data["train"]) == 1
    assert len(cleaned.data["test"]) == 1

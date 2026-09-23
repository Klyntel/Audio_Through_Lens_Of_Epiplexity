from audio_preprocessing.datasets import AudioDataset
from datasets import load_dataset

DATASET_NAME = "meganwei/syntheory"
CONFIGS = [
    "chords",
    "intervals",
    "notes",
    "scales",
    "simple_progressions",
    "tempos",
    "time_signatures"
]
DEFAULT_TEST_RATIO = 0.2
DEFAULT_SEED = 42

def load_syntheory(config: str, test_ratio: float=DEFAULT_TEST_RATIO, seed: int=DEFAULT_SEED) -> AudioDataset:
    """
        Loads a particular config of the syntheory dataset.
        Creates a train test split because this dataset has only a training set.
    """
    assert config in CONFIGS, f"Config must be one of {CONFIGS}, got {config}."

    ds = load_dataset(DATASET_NAME, config)["train"]
    data = ds.train_test_split(test_size=DEFAULT_TEST_RATIO, seed=DEFAULT_SEED)

    return AudioDataset(data=data)
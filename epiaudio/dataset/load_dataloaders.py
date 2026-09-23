from collections.abc import Callable

import torch
from datasets import DatasetDict

from audio_preprocessing.dataset import (
    load_aishell1,
    load_aishell3,
    load_avspeech,
    load_bat,
    load_clothoaqa,
    load_cochlscene,
    load_common_voice,
    load_datased,
    load_demand,
    load_eigenscape,
    load_esd,
    load_fake_or_real,
    load_fleurs,
    load_fma,
    load_libritts,
    load_macs,
    load_meld,
    load_mls,
    load_multivox,
    load_nonspeech7k,
    load_ravdess,
    load_sonyc_ust,
    load_spatial_librispeech,
    load_tau_nigens21,
    load_tau_urban_2022,
    load_toyadmos,
    load_tut2016,
    load_tut2017,
    load_urbansound,
    load_vggsound,
    load_vocal_sound,
    load_voxpopuli,
)
from audio_preprocessing.dataset.base_loader import AudioDataset
from audio_preprocessing.realize import read_window


def load_generic_decoder(dataset_row):
    # read_window returns a numpy array; the tokenizers expect a torch.Tensor,
    # matching what torchcodec's samples.data hands the other decoders directly.
    audio, sample_rate = read_window(
        dataset_row["audio_path"],
        dataset_row["clip_offset"],
        dataset_row["clip_duration"],
    )
    if sample_rate is None:
        raise ValueError(f"Could not determine the sample rate for {dataset_row['audio_path']}.")
    return torch.from_numpy(audio), sample_rate


def load_generic_audio_dataset(data_loader: Callable[[], AudioDataset]) -> AudioDataset:
    """Normalize split names and create one deterministic holdout when needed."""
    ds = data_loader()
    data = {split: rows.shuffle(seed=0) for split, rows in ds.data.items()}
    if "test" not in data and "eval" in data:
        data["test"] = data.pop("eval")
    if "val" not in data and "valid" in data:
        data["val"] = data.pop("valid")
    if "test" not in data:
        train = data.get("train")
        if train is None or len(train) < 2:
            raise ValueError("Dataset needs a train split with at least two rows to create a test split.")
        split = train.train_test_split(test_size=0.1, seed=0)
        data["train"] = split["train"]
        data["test"] = split["test"]
    ds.data = DatasetDict(data)
    return ds


def _generic_dataset_entry(
    data_loader_factory: Callable[[], Callable[[], AudioDataset]],
) -> dict[str, object]:
    return {
        "loader": lambda: load_generic_audio_dataset(data_loader_factory()),
        "decode": load_generic_decoder,
    }


AUDIO_PREPROCESSING_DATASETS = {
    "aishell1": _generic_dataset_entry(
        lambda: load_aishell1.AISHELL1Loader(prepare=True)
    ),
    "aishell3": _generic_dataset_entry(
        lambda: load_aishell3.AISHELL3Loader(prepare=True)
    ),
    "avspeech_smoke": {
        "loader": lambda: load_generic_audio_dataset(
            lambda: load_avspeech.load_avspeech(num_rows=100)
        ),
        "decode": None,
    },
    "bat": _generic_dataset_entry(lambda: load_bat.BATLoader()),
    "clothoaqa": _generic_dataset_entry(lambda: load_clothoaqa.ClothoAQALoader()),
    "cochlscene": _generic_dataset_entry(lambda: load_cochlscene.CochlSceneLoader()),
    "common_voice": _generic_dataset_entry(
        lambda: load_common_voice.CommonVoiceLoader(locales=("en",))
    ),
    "datased": _generic_dataset_entry(lambda: load_datased.DataSEDLoader()),
    "demand": {
        # All 17 official 16 kHz environments (SCAFE has no 16 kHz release on Zenodo,
        # only 48 kHz -- confirmed against record 1227121's file listing). Previously
        # restricted to just DKITCHEN_16k/NPARK_16k as a smoke-test-sized placeholder;
        # widened here to the full corpus for an actual sweep.
        "loader": lambda root=None: load_generic_audio_dataset(
            load_demand.DEMANDLoader(root=root, only=[
                "DKITCHEN_16k", "DLIVING_16k", "DWASHING_16k",
                "NFIELD_16k", "NPARK_16k", "NRIVER_16k",
                "OHALLWAY_16k", "OMEETING_16k", "OOFFICE_16k",
                "PCAFETER_16k", "PRESTO_16k", "PSTATION_16k",
                "SPSQUARE_16k", "STRAFFIC_16k",
                "TBUS_16k", "TCAR_16k", "TMETRO_16k",
            ])
        ),
        "decode": load_generic_decoder,
        "supports_root": True,
    },
    "eigenscape": {
        "loader": lambda root=None: load_generic_audio_dataset(
            load_eigenscape.EigenScapeLoader(
                root=root,
                only=["Lite-EigenScape", "Metadata-EigenScape"],
                scenes=["Beach", "Woodland"],
            )
        ),
        "decode": load_generic_decoder,
        "supports_root": True,
    },
    "esd": _generic_dataset_entry(lambda: load_esd.ESDLoader()),
    "fake_or_real_original": _generic_dataset_entry(
        lambda: load_fake_or_real.FakeOrRealLoader(version="for-original")
    ),
    "fleurs": _generic_dataset_entry(
        lambda: load_fleurs.FLEURSLoader(languages=("en_us",))
    ),
    "fma_small": _generic_dataset_entry(lambda: load_fma.FMALoader(subset="small")),
    "libritts": _generic_dataset_entry(lambda: load_libritts.LibriTTSLoader()),
    "macs": _generic_dataset_entry(lambda: load_macs.MACSLoader()),
    "meld": _generic_dataset_entry(lambda: load_meld.MELDLoader()),
    "mls": _generic_dataset_entry(lambda: load_mls.MLSLoader(languages=("german",))),
    "multivox": _generic_dataset_entry(lambda: load_multivox.MultiVoxLoader()),
    "nonspeech7k": _generic_dataset_entry(lambda: load_nonspeech7k.NonSpeech7kLoader()),
    "ravdess": _generic_dataset_entry(lambda: load_ravdess.RAVDESSLoader()),
    "ravdess_speech": _generic_dataset_entry(
        lambda: load_ravdess.RAVDESSLoader(vocal_channels=("speech",))
    ),
    "sonyc_ust": _generic_dataset_entry(lambda: load_sonyc_ust.SONYCUSTLoader()),
    "spatial_librispeech": _generic_dataset_entry(
        lambda: load_spatial_librispeech.SpatialLibriSpeechLoader()
    ),
    "tau_nigens21": _generic_dataset_entry(
        lambda: load_tau_nigens21.TAUNIGENS21Loader()
    ),
    "tau_urban_2022": _generic_dataset_entry(
        lambda: load_tau_urban_2022.TAUUrban2022Loader()
    ),
    "toyadmos_toycar": _generic_dataset_entry(
        lambda: load_toyadmos.ToyADMOSLoader(subsets=("ToyCar",))
    ),
    "toyadmos_toyconveyor": _generic_dataset_entry(
        lambda: load_toyadmos.ToyADMOSLoader(subsets=("ToyConveyor",))
    ),
    "toyadmos_toytrain": _generic_dataset_entry(
        lambda: load_toyadmos.ToyADMOSLoader(subsets=("ToyTrain",))
    ),
    "tut2016_acoustic_scenes": _generic_dataset_entry(
        lambda: load_tut2016.TUT2016Loader()
    ),
    "tut2017_acoustic_scenes": _generic_dataset_entry(
        lambda: load_tut2017.TUT2017Loader()
    ),
    "urbansound": {
        "loader": lambda root=None: load_generic_audio_dataset(
            load_urbansound.UrbanSoundLoader(root=root)
        ),
        "decode": load_generic_decoder,
        "supports_root": True,
    },
    "vggsound": _generic_dataset_entry(
        lambda: load_vggsound.VGGSoundLoader(prepare=True)
    ),
    "vocal_sound_16k": _generic_dataset_entry(
        lambda: load_vocal_sound.VocalSoundLoader(sample_rate="16k")
    ),
    "vocal_sound_44k": _generic_dataset_entry(
        lambda: load_vocal_sound.VocalSoundLoader(sample_rate="44k")
    ),
    # Comet contains both historical spellings; retain aliases for replayability.
    "vocalsound_16k": _generic_dataset_entry(
        lambda: load_vocal_sound.VocalSoundLoader(sample_rate="16k")
    ),
    "vocalsound_44k": _generic_dataset_entry(
        lambda: load_vocal_sound.VocalSoundLoader(sample_rate="44k")
    ),
    "voxpopuli_en": _generic_dataset_entry(
        lambda: load_voxpopuli.VoxPopuliLoader(languages=("en",))
    ),
}

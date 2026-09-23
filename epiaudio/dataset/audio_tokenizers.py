from pathlib import Path

from transformers import (
    WhisperProcessor,
    AutoProcessor,
    AutoFeatureExtractor,
    DacModel,
    Wav2Vec2FeatureExtractor,
    HubertModel,
    WavLMModel,
    XcodecModel
)
from typing import Any, cast
from abc import abstractmethod
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
import torchaudio
import torch
import numpy as np
import joblib
import os

class Tokenizer():
    # HuggingFace repo ids this tokenizer needs — used to warm the local cache
    # once from the main process before worker processes load the model.
    MODEL_IDS: list[str] = []

    @classmethod
    def get_model_ids(cls, **kwargs) -> list[str]:
        return cls.MODEL_IDS

    @classmethod
    def download_models_local(cls):
        """ Downloads a hugging face
            Model

            overwrite if your tokenizer
            is not a hugging face model
        """
        for repo_id in cls.get_model_ids():
            snapshot_download(repo_id=repo_id)

    @abstractmethod
    def tokenize(self, y: torch.Tensor, sr: int, shape: tuple) -> torch.Tensor:
        pass

    def tokenize_batch(self, ys: list, srs: list, n_tokens: int, dtype: Any = np.int16) -> np.ndarray:
        tokens = [self.tokenize(y, sr, (n_tokens,)) for y, sr in zip(ys, srs)]
        return np.stack([t.detach().cpu().numpy() for t in tokens]).astype(dtype)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes an input sequence of tokens back to continuous embeddings.
        Override when subclassing.
        """
        return torch.empty(0)

    @classmethod
    def create_cfg(cls) -> dict:
        """Overwrites config for spefific tokenizers depending
        on their sequence length and vocab size
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement create_cfg(): it likely produces "
            "continuous features rather than discrete tokens, so model.V/model.L "
            "don't apply the same way as for a discrete-vocab tokenizer."
        )

    @classmethod
    def write_metadata(cls, ds_path: str) -> None:
        """Write create_cfg()'s overrides to <ds_path>/metadata.yaml."""
        os.makedirs(ds_path, exist_ok=True)
        OmegaConf.save(config=OmegaConf.create(cls.create_cfg()), f=os.path.join(ds_path, "metadata.yaml"))

class WhisperTokenizer(Tokenizer):
    TARGET_SR = 16_000
    MODEL_IDS = ["openai/whisper-base"]

    def __init__(self, device: Any = None, local_files_only: bool = False):
        # no GPU model to place — the processor only does CPU-side mel-spectrogram extraction.
        # `device` is accepted so this tokenizer is constructible interchangeably with the others.
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base", local_files_only=local_files_only)

    def tokenize(self, y: torch.Tensor, sr: int, shape: tuple) -> torch.Tensor:
        if sr != self.TARGET_SR:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR)

        tokens = self.processor(
            y[0].numpy(),       # channel 0; shape is (channels, samples) from torchcodec
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
            truncation=True,
            max_length=self.TARGET_SR * 5,
        ).input_features[0]    # (80, 500) mel spectrogram

        assert tokens.shape == torch.Size(shape), (tokens.shape, shape, "not same")
        return tokens

    def tokenize_batch(self, ys: list, srs: list, n_tokens: int, dtype: Any = np.float32) -> np.ndarray:
        resampled = [
            torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR) if sr != self.TARGET_SR else y
            for y, sr in zip(ys, srs)
        ]
        features = self.processor(
            [y[0].numpy() for y in resampled],  # channel 0 per clip
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
            truncation=True,
            max_length=self.TARGET_SR * 5,
        ).input_features  # (B, 80, 500)
        return features.numpy().astype(dtype)


class DACTokenizer(Tokenizer):
    MODEL_IDS = ["descript/dac_16khz"]
    TARGET_SR = 16_000
    N_CODEBOOKS = 12
    FRAMES_PER_CLIP = 250  # 50 frames/sec * 5s
    SEQ_LEN = N_CODEBOOKS * FRAMES_PER_CLIP  # flattened per-clip length; matches prepare_audio.py's shape=(250, 12)

    @classmethod
    def create_cfg(cls) -> dict:
        from transformers import DacConfig
        config = DacConfig.from_pretrained(cls.MODEL_IDS[0])
        return {"model": {"V": config.codebook_size, "L": cls.SEQ_LEN}}

    def __init__(self, device: Any = None, local_files_only: bool = False):
        super().__init__()

        device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = cast(DacModel, DacModel.from_pretrained("descript/dac_16khz", local_files_only=local_files_only))
        torch.nn.Module.to(self.model, device)
        self.processor = AutoProcessor.from_pretrained("descript/dac_16khz", local_files_only=local_files_only)

    def tokenize(self, y: torch.Tensor, sr: int, shape: tuple) -> torch.Tensor:
        if sr != self.TARGET_SR:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR)
        
        max_samples = 5*self.TARGET_SR
        # DAC doesn't allow you to truncate and pad simultaneously, so we do so manually.
        audio = y[0][:max_samples]
        if audio.shape[0] < max_samples:
            audio = torch.nn.functional.pad(audio, (0, max_samples - audio.shape[0]))

        inputs = self.processor(
            audio,
            sampling_rate=self.TARGET_SR,
            return_tensors="pt"
        ).to(self.model.device)
        input_values = inputs["input_values"]
        assert input_values is not None
        with torch.no_grad():
            encode_output = cast(Any, self.model.encode(input_values))
        # (B, n_codebooks=12, n_frames=250) -> (B, frames, codebooks)
        tokens: torch.Tensor = encode_output.audio_codes[0].transpose(0, 1).contiguous()

        assert tokens.shape == torch.Size(shape), (tokens.shape, shape, "not same")
        return tokens

    def tokenize_batch(self, ys: list, srs: list, n_tokens: int, dtype: Any = np.int16) -> np.ndarray:
        max_samples = 16000 * 5
        audios = []
        for y, sr in zip(ys, srs):
            if sr != self.TARGET_SR:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR)
            audio = y[0][:max_samples]
            if audio.shape[0] < max_samples:
                audio = torch.nn.functional.pad(audio, (0, max_samples - audio.shape[0]))
            audios.append(audio.numpy())

        inputs = self.processor(
            audios,  # list of mono arrays — batches to (B, 1, max_samples)
            sampling_rate=16000,
            return_tensors="pt",
        ).to(self.model.device)
        input_values = inputs["input_values"]
        assert input_values is not None
        with torch.no_grad():
            encode_output = cast(Any, self.model.encode(input_values))
        # (B, n_codebooks=12, n_frames=250) -> (B, frames, codebooks)
        tokens: torch.Tensor = encode_output.audio_codes.transpose(1, 2).contiguous()

        return tokens.detach().cpu().numpy().astype(dtype)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens = x.shape
        num_frames = num_tokens // self.N_CODEBOOKS
        x = x.reshape(x.shape[0], num_frames, self.N_CODEBOOKS).transpose(1, 2)
        y = self.model.quantizer.from_codes(x)[0]
        assert isinstance(y, torch.Tensor)

        return y


class SQCodecTokenizer(Tokenizer):
    TARGET_SR = 16_000
    MODEL_NAME = "6kbps"  # 355.56 tokens/s → 1800 tokens per 5s clip; codebook size 117,649
    MODEL_IDS = ["6kbps"]
    VOCAB_SIZE = 117_649  # FSQ, levels=[7,7,7,7,7,7] -> 7**6; confirmed via 6kbps.toml's vq_config
    SEQ_LEN = 1778  # true content length: 355.56 frames/sec * 5s. EnCodec.preprocess() right-pads
    # the input up to a multiple of fill_length before encoding (see sq_codec/en_codec.py), so
    # encode_audio() actually returns 1800 frames for a 5s clip — the trailing ~22 are encoded
    # padding, not real content. tokenize() truncates to SEQ_LEN to drop them.

    @classmethod
    def create_cfg(cls) -> dict:
        return {"model": {"V": cls.VOCAB_SIZE, "L": cls.SEQ_LEN}}

    def __init__(self, device: Any = None, local_files_only: bool = False):
        super().__init__()
        import sq_codec

        # If weights are local to disk, nothing
        # gets downloaded, local_files_only exists only
        # to maintain class structure
        self.codec = sq_codec.get_model(self.MODEL_NAME)
        self.codec.network.eval()
        self.codec.network.to(device=device)

    @classmethod
    def download_models_local(cls):
        import sq_codec
        # If you trace the sq_codec repo
        # the weights are downloaded locally in 
        # https://github.com/zhai-lw/SQCodec/blob/2de0af32d26b2aefde5c6fe421f8a5dc735aac83/sq_codec/__init__.py#L44

        # This includes loading the weights from disk if they are found
        # at the right path!
        sq_codec.get_model(cls.MODEL_NAME)

    def tokenize(self, y: torch.Tensor, sr: int, shape: tuple) -> torch.Tensor:
        if sr != self.TARGET_SR:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR)

        # Codec expects (B, T) with no channel dimension
        audio = y[0].unsqueeze(0)  # (channels, T) → (1, T)
        device = next(self.codec.network.parameters()).device
        audio = audio.to(device)

        with torch.no_grad():
            _, indices = self.codec.encode_audio(audio)

        # encode_audio() includes trailing encoded-padding frames beyond the true
        # content length (see SEQ_LEN's comment) — drop them to match `shape`.
        tokens: torch.Tensor = indices[0, :shape[0]].cpu()  # (T_frames,)
        assert tokens.shape == torch.Size(shape), (tokens.shape, shape, "not same")
        return tokens

    # TODO add batch_tokenize function for SQCodecToeknizer
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        from sq_codec.quantize import VQEmbed # anti-lint shenanigans
        quantizer = cast(VQEmbed, self.codec.network.quantizer)
        y = quantizer.to_features(x)
        assert isinstance(y, torch.Tensor)
        y = y.transpose(1, 2)

        return y


class EnCodecTokenizer(Tokenizer):
    TARGET_SR = 24_000
    BANDWIDTH = 6.0  # kbps → 8 codebooks, 75 frames/sec → 375 frames per 5s clip
    MODEL_IDS = ["facebook/encodec_24khz"]
    SEQ_LEN = 375  # 75 frames/sec * 5s; only codebook 0 is kept (see tokenize())

    @classmethod
    def create_cfg(cls) -> dict:
        from transformers import EncodecConfig
        config = EncodecConfig.from_pretrained(cls.MODEL_IDS[0])
        return {"model": {"V": config.codebook_size, "L": cls.SEQ_LEN}}

    def __init__(self, device: Any = None, local_files_only: bool = False):
        super().__init__()
        from transformers import EncodecModel
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # yeah pytorch werid sometimes so we ignore device type for pyright
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=local_files_only).to(self.device)  # pyright: ignore[reportArgumentType]
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz", local_files_only=local_files_only)
        self.model.eval()

    def tokenize(self, y: torch.Tensor, sr: int, shape: tuple) -> torch.Tensor:
        if sr != self.TARGET_SR:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR)

        inputs = self.processor(
            raw_audio=y[0].numpy(),  # channel 0, numpy array
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            encode_result = cast(Any, self.model.encode(
                inputs["input_values"],
                inputs["padding_mask"],
                bandwidth=self.BANDWIDTH,
            ))

        # audio_codes: (1, n_examples=1, n_codebooks=8, n_frames=375) — dim 0 is a fixed
        # HF-internal axis, not our batch; dim 1 is the actual per-example index.
        # take first codebook → integer tokens in [0, 1023], compatible with model.V=1024
        tokens: torch.Tensor = encode_result.audio_codes[0, 0, 0]  # shape: (n_frames,)

        assert tokens.shape == torch.Size(shape), (tokens.shape, shape, "not same")
        return tokens

    def tokenize_batch(self, ys: list, srs: list, n_tokens: int, dtype: Any = np.int16) -> np.ndarray:
        resampled = [
            torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR) if sr != self.TARGET_SR else y
            for y, sr in zip(ys, srs)
        ]
        inputs = self.processor(
            raw_audio=[y[0].numpy() for y in resampled],  # channel 0 per clip
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            encode_result = cast(Any, self.model.encode(
                inputs["input_values"],
                inputs["padding_mask"],
                bandwidth=self.BANDWIDTH,
            ))

        # dim 0 is the fixed HF-internal axis (see tokenize()); dim 1 is the per-example index.
        tokens: torch.Tensor = encode_result.audio_codes[0, :, 0]  # (B, n_frames)
        return tokens.detach().cpu().numpy().astype(dtype)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.quantizer.decode(x.unsqueeze(0))


# Derived from MIT licensed textlesslib (https://github.com/facebookresearch/textlesslib)

class HuBERTTokenizer(Tokenizer):
    """
    Discrete speech tokenizer following TWIST (Hassid et al., 2023)
    Encodes audio as a sequence of k-means unit IDs via mhubert-base-25hz

    quantizer_path must be the joblib KMeans checkpoint from textlesslib
    fitted on mhubert-base-25hz layer-11 hidden states, vocab_size=500

    tokenize() returns (1, time_steps) for one clip — e.g. (1, 124) for a
    5-second clip, confirmed against the real model rather than assumed from
    "25Hz": the conv feature encoder's kernel/stride arithmetic doesn't divide
    a 5s clip into a round 125 frames. tokenize_batch() (used by prepare_ds)
    returns the (B, time_steps) stack directly, matching the other tokenizers
    in this file.

    deduplicate=True collapses repeated consecutive unit ids, so different
    clips produce different numbers of tokens. That's fine for tokenize() on
    a single clip, but tokenize_batch() needs one fixed shape per clip to
    stack into an array (and prepare_ds needs one fixed shape to lay clips
    out in its memmap), so tokenize_batch() rejects deduplicate=True rather
    than crash confusingly inside np.stack. Use deduplicate=False for the
    prepare_ds/tokenize_batch pipeline, or call tokenize() per clip directly.
    """

    MODEL_ID    = "slprl/mhubert-base-25hz"
    MODEL_IDS   = [MODEL_ID]  # so prepare_ds's download_models_local() pre-fetches this
    TARGET_SR   = 16_000
    KM_LAYER    = 11    # textlesslib __init__.py: mhubert-base-25hz uses layer 11
    VOCAB_SIZE  = 500
    DEFAULT_DURATION_S = 5

    @classmethod
    def create_cfg(cls) -> dict:
        # Assumes the default max_duration_s and deduplicate=False; an instance
        # configured differently won't actually produce this many tokens.
        # L is derived from the model's own conv encoder arithmetic rather than
        # a flat frames-per-second guess — "25Hz" undercounts the frames a real
        # 5s clip produces (124, not 125), since the conv kernels overhang the
        # edges of the input instead of dividing it evenly.
        from transformers import HubertConfig
        config = HubertConfig.from_pretrained(cls.MODEL_ID)
        seq_len = cls.TARGET_SR * cls.DEFAULT_DURATION_S
        for kernel, stride in zip(config.conv_kernel, config.conv_stride):
            seq_len = (seq_len - kernel) // stride + 1
        return {"model": {"V": cls.VOCAB_SIZE, "L": seq_len}}

    def __init__(
        self,
        quantizer_path: str | Path,
        max_duration_s: int = 5,
        deduplicate: bool = False,
        device: torch.device | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()

        if isinstance(max_duration_s, bool) or \
                not isinstance(max_duration_s, int) or \
                max_duration_s <= 0:
            raise ValueError(
                f"max_duration_s must be a positive int, got {max_duration_s!r}"
            )

        self.device         = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_duration_s = max_duration_s
        self.deduplicate    = deduplicate

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.MODEL_ID, local_files_only=local_files_only
        )

        self.model = cast(HubertModel, HubertModel.from_pretrained(
            self.MODEL_ID, local_files_only=local_files_only
        ))
        self.model.eval()
        torch.nn.Module.to(self.model, self.device)

        self.kmeans = joblib.load(quantizer_path)
        if self.kmeans.n_clusters != self.VOCAB_SIZE:
            raise ValueError(
                f"Loaded KMeans has {self.kmeans.n_clusters} clusters, "
                f"expected {self.VOCAB_SIZE}"
            )

        self.centroids = torch.from_numpy(self.kmeans.cluster_centers_).to(
            device=self.device, dtype=torch.float32
        )

        del self.kmeans # has served its purpose; free memory

    @property
    def max_samples(self) -> int:
        return self.TARGET_SR * self.max_duration_s

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def _prepare_waveform(self, y: torch.Tensor) -> torch.Tensor:
        """Mix to mono and truncate/pad to self.max_samples — same
        truncate-or-pad contract DACTokenizer/WhisperTokenizer apply in this
        file, so HuBERTTokenizer behaves the same way when called outside the
        prepare_audio.py pipeline (which already fixes clip length upstream)."""
        # textlesslib uses waveform.mean(0) — averages channels rather than
        # selecting one
        waveform = y.mean(0)[: self.max_samples]
        if waveform.shape[0] < self.max_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_samples - waveform.shape[0]))
        return waveform

    def _hidden_states(self, input_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(
                input_values.to(self.device),
                output_hidden_states=True,
            )
        # hidden_states[0] = CNN encoder, [1–12] = Transformer layers
        return outputs.hidden_states[self.KM_LAYER]

    def tokenize(self, y: torch.Tensor, sr: int, shape: tuple) -> torch.Tensor:
        if sr != self.TARGET_SR:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR)

        waveform = self._prepare_waveform(y)

        inputs = self.feature_extractor(
            waveform.detach().cpu().numpy(),
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
        )

        input_values = inputs["input_values"]
        assert input_values is not None
        features = self._hidden_states(input_values)[0]  # drop the batch-of-1 dim
        unit_ids = torch.cdist(features, self.centroids).argmin(dim=-1)

        if self.deduplicate:
            unit_ids = torch.unique_consecutive(unit_ids)

        tokens = unit_ids.unsqueeze(0)

        if not self.deduplicate:
            assert tokens.shape == torch.Size(shape), (
                f"Shape mismatch: got {tokens.shape}, expected {torch.Size(shape)} (sr={sr})"
            )

        return tokens

    def tokenize_batch(self, ys: list, srs: list, n_tokens: int, dtype: Any = np.int16) -> np.ndarray:
        if self.deduplicate:
            raise NotImplementedError(
                "HuBERTTokenizer(deduplicate=True) produces a different number of "
                "unit ids per clip and can't be stacked into tokenize_batch()'s "
                "fixed-shape array. Call tokenize() on each clip individually instead."
            )

        waveforms = []
        for y, sr in zip(ys, srs):
            if sr != self.TARGET_SR:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR)
            waveforms.append(self._prepare_waveform(y))

        inputs = self.feature_extractor(
            [w.detach().cpu().numpy() for w in waveforms],
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
        )
        input_values = inputs["input_values"]
        assert input_values is not None
        features = self._hidden_states(input_values)  # (B, T, H)

        centroids = self.centroids.unsqueeze(0).expand(features.shape[0], -1, -1)
        unit_ids = torch.cdist(features, centroids).argmin(dim=-1)  # (B, T)

        assert unit_ids.shape[1:] == (n_tokens,), (
            f"Shape mismatch: got {tuple(unit_ids.shape)}, expected (B, {n_tokens})"
        )

        return unit_ids.detach().cpu().numpy().astype(dtype)


class WavLMTokenizer(Tokenizer):
    """Extracts WavLM hidden-state features from raw audio.

    Args:
        model_id: HuggingFace model ID, e.g. "microsoft/wavlm-base" or "microsoft/wavlm-large".
        layer: Transformer layer index to use as the output (0-indexed). -1 uses the final
            hidden state directly, without storing intermediate activations.
    """

    TARGET_SR = 16_000

    def __init__(self, model_id: str = "microsoft/wavlm-base", layer: int = -1, device: Any = None, local_files_only: bool = False):
        super().__init__()

        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = layer
        self.max_samples = self.TARGET_SR * 5

        self.processor = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = cast(WavLMModel, WavLMModel.from_pretrained(model_id, local_files_only=local_files_only))
        torch.nn.Module.to(self.model, self.device)
        self.model.eval()

    @classmethod
    def get_model_ids(cls, model_id: str = "microsoft/wavlm-base", **kwargs) -> list[str]:
        return [model_id]

    def tokenize(self, y: torch.Tensor, sr: int, shape: tuple) -> torch.Tensor:
        if sr != self.TARGET_SR:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR)

        # Truncate to 5 seconds, then zero-pad if shorter — same contract as DACTokenizer.
        audio = y[0][: self.max_samples]
        if audio.shape[0] < self.max_samples:
            audio = torch.nn.functional.pad(audio, (0, self.max_samples - audio.shape[0]))

        inputs = self.processor(
            audio.cpu().numpy(),
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
        ).to(self.device)

        need_all_layers = self.layer != -1
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=need_all_layers)

        if need_all_layers:
            features = outputs.hidden_states[self.layer][0]  # [time, hidden_size]
        else:
            features = outputs.last_hidden_state[0]  # [time, hidden_size]

        assert features.shape == torch.Size(shape), (features.shape, shape, "not same")
        return features


class XCodecTokenizer(Tokenizer):
    MODEL_IDS = ["hf-audio/xcodec-hubert-general"]
    TARGET_SR = 16_000
    VOCAB_SIZE = 1024
    N_CODEBOOKS = 8
    FRAMES_PER_CLIP = 250
    SEQ_LEN = N_CODEBOOKS * FRAMES_PER_CLIP

    @classmethod
    def create_cfg(cls) -> dict:
        return {"model": {"V": cls.VOCAB_SIZE, "L": cls.SEQ_LEN}}

    def __init__(self, device: Any = None, local_files_only: bool=False):
        super().__init__()

        device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = self.MODEL_IDS[0]
        self.model = XcodecModel.from_pretrained(
            model_id,
            local_files_only=local_files_only
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id,
            local_files_only=local_files_only
        )
        torch.nn.Module.to(self.model, device)

    def tokenize(self, y: torch.Tensor, sr: int, shape: tuple) -> torch.Tensor:
        if sr != self.TARGET_SR:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR)

        max_samples = 5*self.TARGET_SR
        audio = y[0][:max_samples]
        if audio.shape[0] < max_samples:
            audio = torch.nn.functional.pad(audio, (0, max_samples - audio.shape[0]))

        inputs = self.feature_extractor(
            raw_audio=audio,
            sampling_rate=self.TARGET_SR,
            return_tensors="pt"
        ).to(self.model.device)
        input_values = inputs["input_values"]
        assert input_values is not None

        with torch.no_grad():
            encode_output = cast(Any, self.model.encode(input_values))
        # encode_output.audio_codes[0] has shape (n_codebooks, n_frames)
        tokens: torch.Tensor = encode_output.audio_codes[0].transpose(0, 1).contiguous()
        assert tokens.shape == torch.Size(shape), (tokens.shape, shape, "not same")

        return tokens

    def tokenize_batch(
        self,
        ys: list,
        srs: list,
        n_tokens: int,
        dtype: Any = np.int16
    ) -> np.ndarray:
        max_samples = 5*self.TARGET_SR
        audios = []
        for y, sr in zip(ys, srs):
            if sr != self.TARGET_SR:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.TARGET_SR)
            audio = y[0][:max_samples]
            if audio.shape[0] < max_samples:
                audio = torch.nn.functional.pad(audio, (0, max_samples - audio.shape[0]))
            audios.append(audio.numpy())

        inputs = self.feature_extractor(
            audios,  # list of mono arrays — batches to (B, 1, max_samples)
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
        ).to(self.model.device)
        input_values = inputs["input_values"]
        assert input_values is not None

        with torch.no_grad():
            encode_output = cast(Any, self.model.encode(input_values))
        # encode_output.audio_codes has shape (batch_size, n_codebooks, n_frames)
        tokens: torch.Tensor = encode_output.audio_codes.transpose(1, 2).contiguous()

        return tokens.detach().cpu().numpy().astype(dtype)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens = x.shape
        num_frames = num_tokens // self.N_CODEBOOKS
        # input shape must be (n_codebooks, batch_size, n_frames)
        x = x.reshape(x.shape[0], num_frames, self.N_CODEBOOKS).transpose(1, 2).transpose(0, 1)
        y = self.model.quantizer.decode(x).to(self.model.device)
        assert isinstance(y, torch.Tensor)

        return y
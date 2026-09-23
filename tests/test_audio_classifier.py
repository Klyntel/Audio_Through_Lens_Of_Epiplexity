import unittest
from typing import cast

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from epiaudio.downstream.audio_classifier import (
    AudioClassifier,
    CausalAudioEncoder,
)
from epiaudio.model_torch import (
    TransformerBackbone,
    TransformerBlock,
    TransformerDecoder,
)


MODEL_ARGS = {
    "num_layers": 2,
    "embed_dim": 12,
    "per_head_dim": 4,
    "vocab_size": 32,
    "seq_length": 6,
}


class CausalAudioEncoderTest(unittest.TestCase):
    def test_matches_epiplexity_backbone_initialization(self) -> None:
        decoder_config = OmegaConf.create(
            {
                "N": MODEL_ARGS["num_layers"],
                "D": MODEL_ARGS["embed_dim"],
                "dh": MODEL_ARGS["per_head_dim"],
                "V": MODEL_ARGS["vocab_size"],
                "L": MODEL_ARGS["seq_length"],
                "embed_init_std": 0.1,
            }
        )
        torch.manual_seed(7)
        decoder = TransformerDecoder(decoder_config)
        torch.manual_seed(7)
        encoder = CausalAudioEncoder(**MODEL_ARGS)

        self.assertIsInstance(decoder, TransformerBackbone)
        self.assertIsInstance(encoder, TransformerBackbone)
        decoder_backbone = {
            name: value
            for name, value in decoder.state_dict().items()
            if not name.startswith("readout.")
        }
        encoder_state = encoder.state_dict()
        self.assertEqual(decoder_backbone.keys(), encoder_state.keys())
        for name, expected in decoder_backbone.items():
            torch.testing.assert_close(encoder_state[name], expected)

        self.assertTrue(
            all(isinstance(block, TransformerBlock) for block in encoder.blocks)
        )
        self.assertEqual(
            [block.branch_multiplier for block in encoder.blocks],
            [0.5, 0.5],
        )

    def test_decoder_keeps_legacy_checkpoint_keys(self) -> None:
        decoder_config = OmegaConf.create(
            {
                "N": MODEL_ARGS["num_layers"],
                "D": MODEL_ARGS["embed_dim"],
                "dh": MODEL_ARGS["per_head_dim"],
                "V": MODEL_ARGS["vocab_size"],
                "L": MODEL_ARGS["seq_length"],
                "embed_init_std": 0.1,
            }
        )
        decoder = TransformerDecoder(decoder_config)
        expected_keys = {"embed.weight", "pos_embed.weight", "readout.weight"}
        for index in range(MODEL_ARGS["num_layers"]):
            prefix = f"blocks.{index}"
            expected_keys.update(
                {
                    f"{prefix}.attn.query_proj.weight",
                    f"{prefix}.attn.key_proj.weight",
                    f"{prefix}.attn.value_proj.weight",
                    f"{prefix}.attn.output_proj.weight",
                    f"{prefix}.mlp.fc1.weight",
                    f"{prefix}.mlp.fc2.weight",
                }
            )

        self.assertEqual(set(decoder.state_dict()), expected_keys)
        checkpoint = {
            name: torch.randn_like(parameter)
            for name, parameter in decoder.state_dict().items()
        }
        decoder.load_state_dict(checkpoint, strict=True)
        for name, expected in checkpoint.items():
            torch.testing.assert_close(decoder.state_dict()[name], expected)

    def test_causal_last_state_can_condition_on_the_full_prefix(self) -> None:
        torch.manual_seed(11)
        encoder = CausalAudioEncoder(**MODEL_ARGS)
        for module in encoder.blocks:
            block = cast(TransformerBlock, module)
            nn.init.normal_(block.attn.output_proj.weight, std=0.1)
            nn.init.normal_(block.mlp.fc2.weight, std=0.1)

        original = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        changed_prefix = torch.tensor([[9, 2, 3, 4]], dtype=torch.long)
        changed_future = torch.tensor([[1, 2, 3, 9]], dtype=torch.long)

        original_features = encoder(original)
        prefix_features = encoder(changed_prefix)
        future_features = encoder(changed_future)

        self.assertFalse(
            torch.allclose(original_features[:, -1], prefix_features[:, -1])
        )
        torch.testing.assert_close(
            original_features[:, :-1],
            future_features[:, :-1],
        )


class AudioClassifierTest(unittest.TestCase):
    def test_outputs_class_logits_and_probabilities(self) -> None:
        model = AudioClassifier(**MODEL_ARGS, num_classes=5)
        audio_tokens = torch.randint(0, 32, (3, 4), dtype=torch.int32)

        logits = model(audio_tokens)
        probabilities = model.predict_proba(audio_tokens)

        self.assertEqual(logits.shape, (3, 5))
        self.assertEqual(probabilities.shape, (3, 5))
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3))
        torch.testing.assert_close(probabilities, torch.full((3, 5), 0.2))
        self.assertIsNone(model.readout.bias)
        torch.testing.assert_close(
            model.readout.weight,
            torch.zeros_like(model.readout.weight),
        )

    def test_cross_entropy_trains_readout_and_backbone(self) -> None:
        torch.manual_seed(13)
        model = AudioClassifier(**MODEL_ARGS, num_classes=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        audio_tokens = torch.randint(0, 32, (4, 6))
        labels = torch.tensor([0, 1, 2, 1])

        loss = nn.functional.cross_entropy(model(audio_tokens), labels)
        loss.backward()
        readout_gradient = model.readout.weight.grad
        self.assertIsNotNone(readout_gradient)
        assert readout_gradient is not None
        self.assertGreater(readout_gradient.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()

        second_loss = nn.functional.cross_entropy(model(audio_tokens), labels)
        second_loss.backward()
        embed_gradient = model.encoder.embed.weight.grad
        self.assertIsNotNone(embed_gradient)
        assert embed_gradient is not None
        self.assertGreater(embed_gradient.abs().sum().item(), 0)

    def test_rejects_appended_label_and_invalid_token_tensors(self) -> None:
        model = AudioClassifier(**MODEL_ARGS, num_classes=3)
        audio_tokens = torch.randint(0, 32, (2, 6))
        labels = torch.tensor([[0], [1]])

        with self.assertRaisesRegex(ValueError, "exceeds configured maximum"):
            model(torch.cat((audio_tokens, labels), dim=1))
        with self.assertRaisesRegex(ValueError, "shape"):
            model(torch.ones(6, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "at least one token"):
            model(torch.empty((2, 0), dtype=torch.long))
        with self.assertRaisesRegex(TypeError, "torch.int32 or torch.int64"):
            model(torch.ones((2, 6), dtype=torch.float32))

    def test_rejects_invalid_model_configuration(self) -> None:
        cases = {
            "layers": ({"num_layers": 0}, "num_layers"),
            "heads": ({"per_head_dim": 5}, "divisible"),
            "vocabulary": ({"vocab_size": 0}, "vocab_size"),
            "sequence": ({"seq_length": 0}, "seq_length"),
            "classes": ({"num_classes": 0}, "num_classes"),
            "initialization": ({"embed_init_std": 0.0}, "embed_init_std"),
        }
        for name, (updates, message) in cases.items():
            with self.subTest(name=name):
                arguments = {**MODEL_ARGS, "num_classes": 3, **updates}
                with self.assertRaisesRegex(ValueError, message):
                    AudioClassifier(**arguments)


if __name__ == "__main__":
    unittest.main()

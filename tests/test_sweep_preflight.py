from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from epiaudio.sweep_preflight import validate_sweep


def _config(**parameters):
    defaults = {
        "T": {"value": 31_250_000},
        "T_eval": {"value": 131072},
        "train_student": {"value": False},
        "ds_path": {"values": ["data/example_encodec"]},
        "B": {"value": 16},
        "model.V": {"value": 1024},
        "model.L": {"value": 375},
    }
    defaults.update(parameters)
    return OmegaConf.create(
        {"method": "pf_grid", "metric": {"name": "K(M)"}, "parameters": defaults}
    )


class SweepPreflightTests(unittest.TestCase):
    def _write_dataset(
        self,
        root: Path,
        *,
        tokenizer: str = "encodec",
        v: int = 1024,
        length: int = 375,
    ) -> None:
        ds_path = root / "data" / f"example_{tokenizer}"
        ds_path.mkdir(parents=True)
        (ds_path / "train.bin").touch()
        (ds_path / "test.bin").touch()
        OmegaConf.save(
            OmegaConf.create({"model": {"V": v, "L": length}}),
            ds_path / "metadata.yaml",
        )
        (root / ".env").write_text("COMET_ML_API=test-only\n")

    def test_valid_sweep_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root)

            result = validate_sweep(
                _config(),
                "sweep.yaml",
                method="pf_grid",
                repo_root=root,
            )

            self.assertTrue(result.ready)
            self.assertFalse([issue for issue in result.issues if issue.level == "ERROR"])

    def test_no_eval_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root)
            cfg = _config(**{"T_eval": {"value": 1000}, "B": {"value": 512}})

            result = validate_sweep(
                cfg,
                "sweep.yaml",
                method="grid",
                repo_root=root,
            )

            messages = "\n".join(issue.message for issue in result.issues)
            self.assertFalse(result.ready)
            self.assertIn(
                "T_eval must be at least B*L or no eval steps will occur",
                messages
            )

    def test_wrong_non_metadata_run_invariants_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root)
            cfg = _config(
                train_student={"value": True},
                **{"model.V": {"value": 256}, "model.L": {"value": 512}},
            )

            result = validate_sweep(
                cfg,
                "sweep.yaml",
                method="grid",
                repo_root=root,
            )

            messages = "\n".join(issue.message for issue in result.issues)
            self.assertFalse(result.ready)
            self.assertIn("expected 'pf_grid'", messages)
            self.assertIn("expected False", messages)

    def test_metadata_values_override_yaml_model_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root)
            cfg = _config(
                **{"model.V": {"value": 256}, "model.L": {"value": 512}},
            )

            result = validate_sweep(
                cfg,
                "sweep.yaml",
                method="pf_grid",
                repo_root=root,
            )

            self.assertTrue(result.ready)
            labels = [label for label, _, _ in result.checks]
            self.assertIn("metadata model.V", labels)
            self.assertIn("metadata model.L", labels)
            self.assertNotIn("YAML model.V", labels)
            self.assertNotIn("YAML model.L", labels)

    def test_wrong_metadata_model_values_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root, v=256, length=512)

            result = validate_sweep(
                _config(),
                "sweep.yaml",
                method="pf_grid",
                repo_root=root,
            )

            messages = "\n".join(issue.message for issue in result.issues)
            self.assertFalse(result.ready)
            self.assertIn("metadata model.V=256; expected 1024", messages)
            self.assertIn("metadata model.L=512; expected 375", messages)

    def test_missing_metadata_is_blocked_even_when_yaml_has_model_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds_path = root / "data" / "example_encodec"
            ds_path.mkdir(parents=True)
            (ds_path / "train.bin").touch()
            (ds_path / "test.bin").touch()
            (root / ".env").write_text("COMET_ML_API=test-only\n")
            cfg = _config(
                **{"model.V": {"value": 256}, "model.L": {"value": 512}},
            )

            result = validate_sweep(
                cfg,
                "sweep.yaml",
                method="pf_grid",
                repo_root=root,
            )

            messages = "\n".join(issue.message for issue in result.issues)
            self.assertFalse(result.ready)
            self.assertIn("model.L/model.V must come from dataset metadata", messages)

    def test_multiple_tokenizers_use_their_own_metadata_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root)
            self._write_dataset(root, tokenizer="dac", length=3000)
            cfg = _config(
                ds_path={
                    "values": [
                        "data/example_encodec",
                        "data/example_dac",
                    ]
                },
            )
            del cfg.parameters["model.V"]
            del cfg.parameters["model.L"]

            result = validate_sweep(
                cfg,
                "sweep.yaml",
                method="pf_grid",
                repo_root=root,
            )

            self.assertTrue(result.ready)
            self.assertEqual(
                sum(label == "metadata model.L" for label, _, _ in result.checks),
                2,
            )

    def test_missing_data_and_metadata_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_sweep(
                _config(),
                "sweep.yaml",
                method="pf_grid",
                repo_root=tmp,
            )

            messages = "\n".join(issue.message for issue in result.issues)
            self.assertFalse(result.ready)
            self.assertIn("metadata.yaml", messages)
            self.assertIn("train.bin", messages)
            self.assertIn("test.bin", messages)

    def test_cli_overrides_are_validated_as_effective_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root)

            result = validate_sweep(
                _config(train_student={"value": True}),
                "sweep.yaml",
                method="pf_grid",
                override_args=["train_student=false"],
                repo_root=root,
            )

            self.assertTrue(result.ready)


if __name__ == "__main__":
    unittest.main()

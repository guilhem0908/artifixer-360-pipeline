import hashlib
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from model_training.data.utils import load_encoded_prompt
from scripts import generate_fixed_caption_embedding as fixed_caption


CAPTION = "A stable modern interior with white walls, pale floors, and natural daylight."


def _local_model(tmp_path: Path) -> Path:
    root = tmp_path / "wan-local"
    tokenizer = root / "tokenizer"
    text_encoder = root / "text_encoder"
    tokenizer.mkdir(parents=True)
    text_encoder.mkdir()
    (tokenizer / "tokenizer.json").write_bytes(b"fixed-tokenizer")
    (text_encoder / "config.json").write_bytes(b"fixed-config")
    (text_encoder / "weights.safetensors").write_bytes(b"fixed-weights")
    return root


class _FakeTokenizer:
    def __call__(self, caption: str, **kwargs):
        assert caption == CAPTION
        assert kwargs == {
            "padding": "max_length",
            "max_length": 512,
            "truncation": True,
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }
        input_ids = torch.zeros((1, 512), dtype=torch.int64)
        attention_mask = torch.zeros((1, 512), dtype=torch.int64)
        input_ids[0, :3] = torch.tensor([101, 202, 1])
        attention_mask[0, :3] = 1
        return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


class _FakeTextEncoder:
    def __init__(self, *, change_second_pass: bool = False):
        self.calls = 0
        self.change_second_pass = change_second_pass

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        assert input_ids.device.type == "cpu"
        assert attention_mask.device.type == "cpu"
        self.calls += 1
        embeddings = torch.zeros((1, 512, 4096), dtype=torch.bfloat16)
        embeddings[0, 0].fill_(1.25)
        embeddings[0, 1].fill_(-2.5)
        embeddings[0, 2] = torch.arange(4096, dtype=torch.float32).to(torch.bfloat16)
        if self.change_second_pass and self.calls == 2:
            embeddings[0, 0, 0] = 9.0
        return SimpleNamespace(last_hidden_state=embeddings)


def _backend_factory(*, change_second_pass: bool = False):
    def factory(model_root: Path, device: str) -> fixed_caption.EncodingBackend:
        assert model_root.is_absolute()
        assert device == "cpu"
        return fixed_caption.EncodingBackend(
            torch=torch,
            tokenizer=_FakeTokenizer(),
            text_encoder=_FakeTextEncoder(change_second_pass=change_second_pass),
            prompt_clean=lambda value: value,
            device=torch.device("cpu"),
        )

    return factory


def test_generated_hdf5_round_trips_through_artifixer_loader(tmp_path: Path):
    model = _local_model(tmp_path)
    output = tmp_path / "caption.h5"

    result = fixed_caption.generate_fixed_caption_embedding(
        caption=CAPTION,
        output_path=output,
        model_path=model,
        backend_factory=_backend_factory(),
    )

    assert result["status"] == "PASS_FIXED_CAPTION_EMBEDDING"
    assert result["sequence_length"] == 3
    assert result["hidden_size"] == 4096
    assert result["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with h5py.File(output, "r") as hdf5:
        assert list(hdf5.keys()) == ["fixed_caption"]
        dataset = hdf5["fixed_caption"]
        assert dataset.dtype == np.dtype("uint16")
        assert dataset.shape == (3, 4096)
        assert dataset.attrs["caption"] == CAPTION
        assert np.asarray(dataset.attrs["image_indices"]).dtype == np.dtype("int64")
        stored_bits = dataset[:]
        assert hdf5.attrs["embedding_sha256"] == hashlib.sha256(stored_bits.tobytes()).hexdigest()

    loaded, loaded_caption = load_encoded_prompt([output], torch.Generator().manual_seed(0))
    assert loaded_caption == CAPTION
    assert loaded.dtype == torch.bfloat16
    assert loaded.shape == (512, 4096)
    assert torch.equal(loaded[:3].view(torch.uint16), torch.from_numpy(stored_bits))
    assert torch.count_nonzero(loaded[3:]).item() == 0


def test_same_inputs_produce_byte_identical_hdf5(tmp_path: Path):
    model = _local_model(tmp_path)
    model_copy = tmp_path / "same-model-at-another-path"
    shutil.copytree(model, model_copy)
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"

    first_result = fixed_caption.generate_fixed_caption_embedding(
        caption=CAPTION,
        output_path=first,
        model_path=model,
        backend_factory=_backend_factory(),
    )
    second_result = fixed_caption.generate_fixed_caption_embedding(
        caption=CAPTION,
        output_path=second,
        model_path=model_copy,
        backend_factory=_backend_factory(),
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_result["output_sha256"] == second_result["output_sha256"]
    assert first_result["model_tree_sha256"] == second_result["model_tree_sha256"]


def test_refuses_to_overwrite_existing_output_before_loading_model(tmp_path: Path):
    model = _local_model(tmp_path)
    output = tmp_path / "caption.h5"
    output.write_bytes(b"do-not-touch")
    factory_called = False

    def forbidden_factory(model_root: Path, device: str):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("backend must not load")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fixed_caption.generate_fixed_caption_embedding(
            caption=CAPTION,
            output_path=output,
            model_path=model,
            backend_factory=forbidden_factory,
        )

    assert not factory_called
    assert output.read_bytes() == b"do-not-touch"


def test_non_reproducible_encoder_fails_without_publishing(tmp_path: Path):
    model = _local_model(tmp_path)
    output = tmp_path / "caption.h5"

    with pytest.raises(RuntimeError, match="not bitwise reproducible"):
        fixed_caption.generate_fixed_caption_embedding(
            caption=CAPTION,
            output_path=output,
            model_path=model,
            backend_factory=_backend_factory(change_second_pass=True),
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".caption.h5.*.tmp"))


def test_model_tree_hash_gate_fails_before_backend_load(tmp_path: Path):
    model = _local_model(tmp_path)
    output = tmp_path / "caption.h5"

    with pytest.raises(RuntimeError, match="model tree SHA-256 mismatch"):
        fixed_caption.generate_fixed_caption_embedding(
            caption=CAPTION,
            output_path=output,
            model_path=model,
            expected_model_tree_sha256="0" * 64,
            backend_factory=lambda *_: pytest.fail("backend must not load after a hash mismatch"),
        )

    assert not output.exists()


@pytest.mark.parametrize("missing_subfolder", ["tokenizer", "text_encoder"])
def test_requires_complete_local_model_layout(tmp_path: Path, missing_subfolder: str):
    model = _local_model(tmp_path)
    target = model / missing_subfolder
    for child in target.iterdir():
        child.unlink()
    target.rmdir()

    with pytest.raises(FileNotFoundError, match="missing required subfolder"):
        fixed_caption.generate_fixed_caption_embedding(
            caption=CAPTION,
            output_path=tmp_path / "caption.h5",
            model_path=model,
            backend_factory=_backend_factory(),
        )


def test_local_backend_forces_offline_from_pretrained_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = _local_model(tmp_path)
    calls = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("tokenizer", path, kwargs, os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]))
            return object()

    class LoadedEncoder:
        config = SimpleNamespace(d_model=4096)

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True

    class FakeUMT5Encoder:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("encoder", path, kwargs, os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]))
            return LoadedEncoder()

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    backend = fixed_caption.load_local_backend(
        model,
        "cpu",
        runtime_importer=lambda: (torch, FakeAutoTokenizer, FakeUMT5Encoder, lambda value: value),
    )

    assert backend.device == torch.device("cpu")
    assert [call[0] for call in calls] == ["tokenizer", "encoder"]
    assert all(call[1] == str(model) for call in calls)
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert all(call[2]["trust_remote_code"] is False for call in calls)
    assert all(call[3:] == ("1", "1") for call in calls)
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


@pytest.mark.parametrize("caption", ["", "   ", "bad\x00caption"])
def test_rejects_invalid_explicit_captions(caption: str):
    with pytest.raises(ValueError):
        fixed_caption.canonicalize_caption(caption)

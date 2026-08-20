#!/usr/bin/env python3
"""Generate a deterministic ArtiFixer prompt HDF5 from an explicit caption.

The produced file intentionally contains exactly one top-level dataset.  Its
values are the raw uint16 representation of a bfloat16 Wan UMT5 embedding,
which is the format consumed by ``model_training.data.utils.load_encoded_prompt``.

Model loading is local-only.  This script never falls back to the Hugging Face
Hub, and it verifies two independent encoder passes before publishing a file.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


FORMAT_VERSION = "artifixer-fixed-caption-v1"
DEFAULT_DATASET_NAME = "fixed_caption"
MAX_SEQUENCE_LENGTH = 512
EXPECTED_HIDDEN_SIZE = 4096
MAX_CAPTION_UTF8_BYTES = 32 * 1024
HASH_CHUNK_SIZE = 8 * 1024 * 1024
REQUIRED_MODEL_SUBFOLDERS = ("tokenizer", "text_encoder")


@dataclass(frozen=True)
class EncodingBackend:
    """Runtime objects needed for fixed-caption encoding."""

    torch: Any
    tokenizer: Any
    text_encoder: Any
    prompt_clean: Callable[[str], str]
    device: Any


@dataclass(frozen=True)
class EncodedCaption:
    """A prompt encoded in the on-disk representation expected by ArtiFixer."""

    uint16_embedding: Any
    cleaned_caption: str
    tokenization_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_caption(caption: str) -> str:
    """Return the single canonical caption representation accepted by the CLI."""

    if not isinstance(caption, str):
        raise TypeError("caption must be a string")
    canonical = unicodedata.normalize("NFC", caption.replace("\r\n", "\n").replace("\r", "\n")).strip()
    if not canonical:
        raise ValueError("caption must not be empty or whitespace-only")
    if "\x00" in canonical:
        raise ValueError("caption must not contain NUL characters")
    if len(canonical.encode("utf-8")) > MAX_CAPTION_UTF8_BYTES:
        raise ValueError(f"caption exceeds the {MAX_CAPTION_UTF8_BYTES}-byte UTF-8 limit")
    return canonical


def validate_dataset_name(dataset_name: str) -> str:
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError("dataset name must be a non-empty string")
    if dataset_name != dataset_name.strip():
        raise ValueError("dataset name must not have surrounding whitespace")
    if dataset_name in {".", ".."} or "/" in dataset_name or "\x00" in dataset_name:
        raise ValueError("dataset name must be one top-level HDF5 component")
    return dataset_name


def _validate_expected_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError("expected model-tree SHA-256 must be exactly 64 hexadecimal characters")
    return normalized


def resolve_local_model_root(model_path: Path) -> Path:
    try:
        root = Path(model_path).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"local model path does not exist: {model_path}") from error
    if not root.is_dir():
        raise NotADirectoryError(f"local model path is not a directory: {root}")
    for subfolder in REQUIRED_MODEL_SUBFOLDERS:
        candidate = root / subfolder
        if not candidate.is_dir():
            raise FileNotFoundError(f"local model is missing required subfolder: {candidate}")
    return root


def _model_files(model_root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for subfolder in REQUIRED_MODEL_SUBFOLDERS:
        subtree = model_root / subfolder
        for candidate in subtree.rglob("*"):
            if candidate.is_file():
                files.append(candidate)
            elif candidate.is_symlink():
                raise RuntimeError(f"model tree contains a broken or unsupported symlink: {candidate}")
            elif not candidate.is_dir():
                raise RuntimeError(f"model tree contains a non-file entry: {candidate}")
    ordered = tuple(sorted(files, key=lambda path: path.relative_to(model_root).as_posix()))
    if not ordered:
        raise RuntimeError(f"local model contains no files under {REQUIRED_MODEL_SUBFOLDERS}: {model_root}")
    return ordered


def model_tree_sha256(model_root: Path) -> str:
    """Hash model filenames, sizes, and contents in a stable order."""

    paths = _model_files(model_root)
    tree_digest = hashlib.sha256()
    for path in paths:
        before = path.stat()
        content_sha256 = sha256_file(path)
        after = path.stat()
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity:
            raise RuntimeError(f"model file changed while it was being hashed: {path}")
        relative = path.relative_to(model_root).as_posix()
        record = f"{relative}\0{after.st_size}\0{content_sha256}\n".encode("utf-8")
        tree_digest.update(record)
    if _model_files(model_root) != paths:
        raise RuntimeError("model tree changed while it was being hashed")
    return tree_digest.hexdigest()


def model_tree_identity(model_root: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    """Capture a cheap immutable-tree identity for checks around model loading."""

    identity = []
    for path in _model_files(model_root):
        stat = path.stat()
        identity.append(
            (
                path.relative_to(model_root).as_posix(),
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    return tuple(identity)


@contextlib.contextmanager
def _offline_environment() -> Iterator[None]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _import_runtime_dependencies() -> tuple[Any, Any, Any, Callable[[str], str]]:
    import torch
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, UMT5EncoderModel

    return torch, AutoTokenizer, UMT5EncoderModel, prompt_clean


def _configure_determinism(torch: Any, device_name: str) -> Any:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        device = torch.device(device_name)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid torch device: {device_name}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu', 'cuda', or an explicit CUDA device such as 'cuda:0'")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; detected {torch.cuda.device_count()} device(s)"
            )
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    return device


def load_local_backend(
    model_root: Path,
    device_name: str,
    *,
    runtime_importer: Callable[[], tuple[Any, Any, Any, Callable[[str], str]]] | None = None,
) -> EncodingBackend:
    """Load Wan's tokenizer and UMT5 encoder without permitting network access."""

    importer = runtime_importer or _import_runtime_dependencies
    with _offline_environment():
        torch, auto_tokenizer, umt5_encoder, prompt_clean = importer()
        device = _configure_determinism(torch, device_name)
        tokenizer = auto_tokenizer.from_pretrained(
            str(model_root),
            subfolder="tokenizer",
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        text_encoder = umt5_encoder.from_pretrained(
            str(model_root),
            subfolder="text_encoder",
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
        )
        moved_encoder = text_encoder.to(device)
        if moved_encoder is not None:
            text_encoder = moved_encoder
        text_encoder.eval()

    configured_hidden_size = getattr(getattr(text_encoder, "config", None), "d_model", None)
    if configured_hidden_size is not None and int(configured_hidden_size) != EXPECTED_HIDDEN_SIZE:
        raise RuntimeError(
            f"text encoder hidden size is {configured_hidden_size}, expected {EXPECTED_HIDDEN_SIZE} for ArtiFixer"
        )
    return EncodingBackend(
        torch=torch,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt_clean=prompt_clean,
        device=device,
    )


def _tensor_sha256(torch: Any, *tensors: Any) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        cpu_tensor = tensor.detach().to("cpu").contiguous()
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(cpu_tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(cpu_tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _encode_once(caption: str, backend: EncodingBackend) -> EncodedCaption:
    torch = backend.torch
    cleaned_caption = backend.prompt_clean(caption)
    if not isinstance(cleaned_caption, str) or not cleaned_caption:
        raise RuntimeError("Wan prompt_clean returned an empty or non-string caption")
    if "\x00" in cleaned_caption:
        raise RuntimeError("Wan prompt_clean returned a caption containing a NUL character")
    if len(cleaned_caption.encode("utf-8")) > MAX_CAPTION_UTF8_BYTES:
        raise RuntimeError("Wan prompt_clean returned an unexpectedly large caption")

    tokenized = backend.tokenizer(
        cleaned_caption,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = getattr(tokenized, "input_ids", None)
    attention_mask = getattr(tokenized, "attention_mask", None)
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise RuntimeError("tokenizer did not return torch input_ids and attention_mask tensors")
    expected_shape = (1, MAX_SEQUENCE_LENGTH)
    if tuple(input_ids.shape) != expected_shape or tuple(attention_mask.shape) != expected_shape:
        raise RuntimeError(
            f"tokenizer returned shapes {tuple(input_ids.shape)} and {tuple(attention_mask.shape)}, "
            f"expected {expected_shape}"
        )

    mask_cpu = attention_mask.detach().to("cpu")
    if not torch.all((mask_cpu == 0) | (mask_cpu == 1)).item():
        raise RuntimeError("tokenizer attention mask is not binary")
    sequence_length = int(mask_cpu.sum().item())
    if not 1 <= sequence_length <= MAX_SEQUENCE_LENGTH:
        raise RuntimeError(f"invalid encoded prompt sequence length: {sequence_length}")
    expected_mask = torch.arange(MAX_SEQUENCE_LENGTH, device="cpu") < sequence_length
    if not torch.equal(mask_cpu[0].bool(), expected_mask):
        raise RuntimeError("tokenizer attention mask is not a contiguous prefix")

    tokenization_sha256 = _tensor_sha256(torch, input_ids, attention_mask)
    with torch.inference_mode():
        output = backend.text_encoder(
            input_ids.to(backend.device),
            attention_mask.to(backend.device),
        )
    embeddings = getattr(output, "last_hidden_state", None)
    expected_embedding_shape = (1, MAX_SEQUENCE_LENGTH, EXPECTED_HIDDEN_SIZE)
    if not isinstance(embeddings, torch.Tensor) or tuple(embeddings.shape) != expected_embedding_shape:
        actual_shape = None if embeddings is None else tuple(embeddings.shape)
        raise RuntimeError(
            f"text encoder returned shape {actual_shape}, expected {expected_embedding_shape}"
        )
    if embeddings.dtype != torch.bfloat16:
        raise RuntimeError(f"text encoder returned dtype {embeddings.dtype}, expected torch.bfloat16")
    active_embeddings = embeddings[0, :sequence_length].detach().to("cpu").contiguous()
    if not torch.isfinite(active_embeddings.float()).all().item():
        raise RuntimeError("text encoder returned non-finite prompt embeddings")
    uint16_embedding = active_embeddings.view(torch.uint16).numpy().copy()

    return EncodedCaption(
        uint16_embedding=uint16_embedding,
        cleaned_caption=cleaned_caption,
        tokenization_sha256=tokenization_sha256,
    )


def encode_caption_reproducibly(caption: str, backend: EncodingBackend) -> EncodedCaption:
    first = _encode_once(caption, backend)
    second = _encode_once(caption, backend)
    if first.cleaned_caption != second.cleaned_caption:
        raise RuntimeError("prompt cleaning was not reproducible across two passes")
    if first.tokenization_sha256 != second.tokenization_sha256:
        raise RuntimeError("tokenization was not reproducible across two passes")

    import numpy as np

    if not np.array_equal(first.uint16_embedding, second.uint16_embedding):
        raise RuntimeError("text-encoder output was not bitwise reproducible across two passes")
    return first


def _write_hdf5(
    path: Path,
    *,
    dataset_name: str,
    caption: str,
    encoded: EncodedCaption,
    model_sha256: str,
) -> None:
    import h5py
    import numpy as np

    with h5py.File(path, "w", libver="earliest") as hdf5:
        hdf5.attrs["format_version"] = FORMAT_VERSION
        hdf5.attrs["storage_dtype"] = "bfloat16_as_uint16"
        hdf5.attrs["max_sequence_length"] = MAX_SEQUENCE_LENGTH
        hdf5.attrs["hidden_size"] = EXPECTED_HIDDEN_SIZE
        hdf5.attrs["caption_sha256"] = hashlib.sha256(caption.encode("utf-8")).hexdigest()
        hdf5.attrs["cleaned_caption"] = encoded.cleaned_caption
        hdf5.attrs["cleaned_caption_sha256"] = hashlib.sha256(
            encoded.cleaned_caption.encode("utf-8")
        ).hexdigest()
        hdf5.attrs["tokenization_sha256"] = encoded.tokenization_sha256
        hdf5.attrs["embedding_sha256"] = hashlib.sha256(
            encoded.uint16_embedding.tobytes(order="C")
        ).hexdigest()
        hdf5.attrs["model_tree_sha256"] = model_sha256
        dataset = hdf5.create_dataset(
            dataset_name,
            data=encoded.uint16_embedding,
            dtype=np.uint16,
            track_times=False,
        )
        dataset.attrs["caption"] = caption
        dataset.attrs["image_indices"] = np.empty((0,), dtype=np.int64)
        hdf5.flush()


def validate_prompt_hdf5(
    path: Path,
    *,
    dataset_name: str,
    caption: str,
    expected_embedding: Any,
) -> None:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as hdf5:
        keys = list(hdf5.keys())
        if keys != [dataset_name]:
            raise RuntimeError(f"prompt HDF5 must contain exactly [{dataset_name!r}], found {keys}")
        dataset = hdf5[dataset_name]
        if dataset.dtype != np.dtype("uint16"):
            raise RuntimeError(f"prompt dataset dtype is {dataset.dtype}, expected uint16")
        if dataset.ndim != 2 or dataset.shape[1] != EXPECTED_HIDDEN_SIZE:
            raise RuntimeError(
                f"prompt dataset shape is {dataset.shape}, expected N x {EXPECTED_HIDDEN_SIZE}"
            )
        if not 1 <= dataset.shape[0] <= MAX_SEQUENCE_LENGTH:
            raise RuntimeError(f"prompt dataset sequence length is invalid: {dataset.shape[0]}")
        if dataset.attrs.get("caption") != caption:
            raise RuntimeError("prompt dataset caption attribute does not match the explicit caption")
        image_indices = dataset.attrs.get("image_indices")
        if image_indices is None or np.asarray(image_indices).dtype != np.dtype("int64"):
            raise RuntimeError("prompt dataset image_indices attribute is missing or not int64")
        if np.asarray(image_indices).shape != (0,):
            raise RuntimeError("fixed-caption image_indices must be empty")
        if not np.array_equal(dataset[:], expected_embedding):
            raise RuntimeError("prompt HDF5 payload changed during serialization")
        if hdf5.attrs.get("format_version") != FORMAT_VERSION:
            raise RuntimeError("prompt HDF5 format version is missing or incorrect")


def _publish_temp_file(temp_path: Path, output_path: Path, *, replace: bool) -> None:
    if replace:
        os.replace(temp_path, output_path)
        return
    try:
        os.link(temp_path, output_path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}") from error
    temp_path.unlink()


def generate_fixed_caption_embedding(
    *,
    caption: str,
    output_path: Path,
    model_path: Path,
    dataset_name: str = DEFAULT_DATASET_NAME,
    device: str = "cpu",
    expected_model_tree_sha256: str | None = None,
    replace: bool = False,
    backend_factory: Callable[[Path, str], EncodingBackend] | None = None,
) -> dict[str, Any]:
    """Encode and atomically publish one fixed-caption prompt file."""

    caption = canonicalize_caption(caption)
    dataset_name = validate_dataset_name(dataset_name)
    expected_model_sha256 = _validate_expected_sha256(expected_model_tree_sha256)
    model_root = resolve_local_model_root(model_path)
    output_path = Path(output_path).expanduser().absolute()
    resolved_output = output_path.resolve(strict=False)
    if resolved_output.is_relative_to(model_root):
        raise ValueError("output path must be outside the immutable local model tree")
    if output_path.is_symlink():
        raise RuntimeError(f"refusing to publish through a symlink: {output_path}")
    if output_path.exists():
        if output_path.is_dir():
            raise IsADirectoryError(f"output path is a directory: {output_path}")
        if not replace:
            raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    model_sha256 = model_tree_sha256(model_root)
    if expected_model_sha256 is not None and model_sha256 != expected_model_sha256:
        raise RuntimeError(
            f"local model tree SHA-256 mismatch: expected {expected_model_sha256}, got {model_sha256}"
        )

    model_identity = model_tree_identity(model_root)
    factory = backend_factory or load_local_backend
    backend = factory(model_root, device)
    encoded = encode_caption_reproducibly(caption, backend)
    if model_tree_identity(model_root) != model_identity:
        raise RuntimeError("local model tree changed while the caption was being encoded")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        _write_hdf5(
            temp_path,
            dataset_name=dataset_name,
            caption=caption,
            encoded=encoded,
            model_sha256=model_sha256,
        )
        validate_prompt_hdf5(
            temp_path,
            dataset_name=dataset_name,
            caption=caption,
            expected_embedding=encoded.uint16_embedding,
        )
        with temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        _publish_temp_file(temp_path, output_path, replace=replace)
        validate_prompt_hdf5(
            output_path,
            dataset_name=dataset_name,
            caption=caption,
            expected_embedding=encoded.uint16_embedding,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "status": "PASS_FIXED_CAPTION_EMBEDDING",
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "dataset_name": dataset_name,
        "caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
        "cleaned_caption_sha256": hashlib.sha256(encoded.cleaned_caption.encode("utf-8")).hexdigest(),
        "tokenization_sha256": encoded.tokenization_sha256,
        "model_tree_sha256": model_sha256,
        "sequence_length": int(encoded.uint16_embedding.shape[0]),
        "hidden_size": int(encoded.uint16_embedding.shape[1]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic, local-only ArtiFixer fixed-caption embedding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--caption", required=True, help="Explicit caption text to encode")
    parser.add_argument("--output", required=True, type=Path, help="Destination caption.h5 path")
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Local Wan Diffusers model root containing tokenizer/ and text_encoder/",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--device", default="cpu", help="Explicit torch device, for example cpu or cuda:0")
    parser.add_argument(
        "--expected-model-tree-sha256",
        default=None,
        help="Optional immutable-model gate; generation fails if the local tree differs",
    )
    parser.add_argument("--replace", action="store_true", help="Atomically replace an existing regular output file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_fixed_caption_embedding(
        caption=args.caption,
        output_path=args.output,
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        device=args.device,
        expected_model_tree_sha256=args.expected_model_tree_sha256,
        replace=args.replace,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

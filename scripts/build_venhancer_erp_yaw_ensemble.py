#!/usr/bin/env python3
"""Prepare and fuse a seam-safe two-yaw VEnhancer ERP run.

VEnhancer is not longitude-circular: encoding an ERP at its native yaw exposes
the panorama seam to zero padding.  This tool prepares a second input rolled by
180 degrees *before* VAE encoding, then combines both enhanced outputs so each
yaw owns the longitude farthest from its artificial image boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is None:
            raise ValueError(f"decoder returned an empty frame: {path}")
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"video contains no frames: {path}")
    return frames, fps


def validate_frames(
    frames: Sequence[np.ndarray],
    *,
    expected_count: int,
    expected_width: int,
    expected_height: int,
    label: str,
) -> None:
    if len(frames) != expected_count:
        raise ValueError(f"{label} has {len(frames)} frames, expected {expected_count}")
    expected_shape = (expected_height, expected_width, 3)
    for index, frame in enumerate(frames):
        if frame.shape != expected_shape or frame.dtype != np.uint8:
            raise ValueError(
                f"{label} frame {index} is {frame.shape}/{frame.dtype}, "
                f"expected {expected_shape}/uint8"
            )


def write_png_sequence(directory: Path, frames: Sequence[np.ndarray]) -> list[dict[str, object]]:
    directory.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        path = directory / f"{index:05d}.png"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"failed to write {path}")
        records.append(
            {
                "index": index,
                "name": path.name,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def encode_frames(
    frames_dir: Path,
    output: Path,
    *,
    fps: float,
    frame_count: int,
    lossless: bool,
) -> None:
    if lossless:
        codec = ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0"]
    else:
        codec = [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            format(fps, ".12g"),
            "-start_number",
            "0",
            "-i",
            str(frames_dir / "%05d.png"),
            "-frames:v",
            str(frame_count),
            *codec,
            str(output),
        ],
        check=True,
    )


def direct_weight(width: int) -> np.ndarray:
    """Weight native-yaw output centrally and 180-yaw output at the ERP seam."""

    if width < 16 or width % 2:
        raise ValueError("ERP width must be an even integer of at least 16 pixels")
    x = np.arange(width, dtype=np.float32)
    distance = np.abs(x - width / 2.0) / width
    inner = 0.25
    outer = 0.375
    weight = np.ones(width, dtype=np.float32)
    weight[distance >= outer] = 0.0
    transition = (distance > inner) & (distance < outer)
    phase = (distance[transition] - inner) / (outer - inner)
    weight[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return weight.reshape(1, width, 1)


def fuse_frame(direct: np.ndarray, yaw_unrolled: np.ndarray) -> np.ndarray:
    if direct.shape != yaw_unrolled.shape or direct.ndim != 3:
        raise ValueError("ensemble inputs must be equal-shape HWC images")
    weight = direct_weight(direct.shape[1])
    fused = direct.astype(np.float32) * weight + yaw_unrolled.astype(np.float32) * (1.0 - weight)
    return np.clip(np.rint(fused), 0, 255).astype(np.uint8)


def apply_seam_guard(enhanced: np.ndarray, source: np.ndarray, pixels: int) -> np.ndarray:
    """Soft-lock the source only near the longitude wrap boundary."""

    if enhanced.shape != source.shape or enhanced.ndim != 3:
        raise ValueError("seam guard inputs must be equal-shape HWC images")
    width = enhanced.shape[1]
    if pixels < 1 or pixels >= width // 4:
        raise ValueError("seam guard must be between 1 pixel and one quarter width")
    x = np.arange(width, dtype=np.float32)
    distance = np.minimum(x, width - 1 - x)
    source_weight = np.zeros(width, dtype=np.float32)
    active = distance < pixels
    source_weight[active] = 0.5 * (1.0 + np.cos(np.pi * distance[active] / pixels))
    source_weight = source_weight.reshape(1, width, 1)
    guarded = enhanced.astype(np.float32) * (1.0 - source_weight) + source.astype(
        np.float32
    ) * source_weight
    return np.clip(np.rint(guarded), 0, 255).astype(np.uint8)


def frame_seam_metrics(frame: np.ndarray) -> dict[str, float]:
    image = frame.astype(np.float32) / 255.0
    seam_values = np.abs(image[:, 0] - image[:, -1]).mean(axis=1)
    left_adjacent = np.abs(image[:, 0] - image[:, 1]).mean(axis=1)
    right_adjacent = np.abs(image[:, -1] - image[:, -2]).mean(axis=1)
    adjacent = 0.5 * (left_adjacent + right_adjacent)
    seam_mae = float(seam_values.mean())
    adjacent_mae = float(adjacent.mean())
    return {
        "mae": seam_mae,
        "p95": float(np.percentile(seam_values, 95.0)),
        "adjacent_column_mae": adjacent_mae,
        "normalized_ratio": seam_mae / max(adjacent_mae, 1.0e-8),
    }


def aggregate_seams(frames: Sequence[np.ndarray]) -> dict[str, object]:
    records = [frame_seam_metrics(frame) for frame in frames]

    def summary(key: str) -> dict[str, float]:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95.0)),
            "max": float(values.max()),
        }

    return {
        "frame_count": len(records),
        "mae": summary("mae"),
        "p95": summary("p95"),
        "adjacent_column_mae": summary("adjacent_column_mae"),
        "normalized_ratio": summary("normalized_ratio"),
        "frames": [{"index": index, **record} for index, record in enumerate(records)],
    }


def evaluate_seams(
    inputs: Sequence[np.ndarray], predictions: Sequence[np.ndarray]
) -> dict[str, object]:
    baseline = aggregate_seams(inputs)
    prediction = aggregate_seams(predictions)
    thresholds = {
        "maximum_erp_seam_mae": 0.06,
        "maximum_erp_seam_p95": 0.18,
        "absolute_baseline_slack": 0.002,
        "maximum_regression_ratio": 1.25,
        "mae_regression_slack": 0.005,
        "p95_regression_slack": 0.01,
        "normalized_ratio_slack": 0.5,
    }
    effective_mae_cap = max(
        thresholds["maximum_erp_seam_mae"],
        baseline["mae"]["max"] + thresholds["absolute_baseline_slack"],
    )
    effective_p95_cap = max(
        thresholds["maximum_erp_seam_p95"],
        baseline["p95"]["max"] + thresholds["absolute_baseline_slack"],
    )
    thresholds["effective_maximum_erp_seam_mae"] = effective_mae_cap
    thresholds["effective_maximum_erp_seam_p95"] = effective_p95_cap
    checks = {
        "absolute_mae_all_frames": prediction["mae"]["max"]
        <= effective_mae_cap,
        "absolute_p95_all_frames": prediction["p95"]["max"]
        <= effective_p95_cap,
        "mae_median_non_regression": prediction["mae"]["median"]
        <= baseline["mae"]["median"] * thresholds["maximum_regression_ratio"]
        + thresholds["mae_regression_slack"],
        "mae_p95_non_regression": prediction["mae"]["p95"]
        <= baseline["mae"]["p95"] * thresholds["maximum_regression_ratio"]
        + thresholds["mae_regression_slack"],
        "p95_median_non_regression": prediction["p95"]["median"]
        <= baseline["p95"]["median"] * thresholds["maximum_regression_ratio"]
        + thresholds["p95_regression_slack"],
        "normalized_ratio_median_non_regression": prediction["normalized_ratio"]["median"]
        <= baseline["normalized_ratio"]["median"] * thresholds["maximum_regression_ratio"]
        + thresholds["normalized_ratio_slack"],
    }
    return {
        "schema_version": 1,
        "method": "erp_first_last_column_with_input_referenced_non_regression",
        "thresholds": thresholds,
        "input": baseline,
        "prediction": prediction,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def prepare(args: argparse.Namespace) -> dict[str, object]:
    source = args.input_video.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    all_frames, fps = read_video(source)
    if not math.isclose(fps, args.fps, rel_tol=0.0, abs_tol=0.02):
        raise ValueError(f"input fps is {fps}, expected {args.fps}")
    stop = args.start_index + args.frame_count
    if args.start_index < 0 or stop > len(all_frames):
        raise ValueError("requested frame interval is outside the source video")
    selected = all_frames[args.start_index:stop]
    validate_frames(
        selected,
        expected_count=args.frame_count,
        expected_width=args.width,
        expected_height=args.height,
        label="source interval",
    )
    if args.yaw_shift <= 0 or args.yaw_shift >= args.width:
        raise ValueError("yaw shift must be strictly inside the ERP width")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        direct_dir = staging / "direct_frames"
        yaw_dir = staging / "yaw180_frames"
        direct_records = write_png_sequence(direct_dir, selected)
        yaw_records = write_png_sequence(
            yaw_dir, [np.roll(frame, args.yaw_shift, axis=1) for frame in selected]
        )
        direct_video = staging / "direct_input.mkv"
        yaw_video = staging / "yaw180_input.mkv"
        encode_frames(direct_dir, direct_video, fps=args.fps, frame_count=args.frame_count, lossless=True)
        encode_frames(yaw_dir, yaw_video, fps=args.fps, frame_count=args.frame_count, lossless=True)
        manifest = {
            "schema": "artifixer.venhancer_erp_yaw_ensemble_input_v1",
            "source": {
                "path": str(source),
                "sha256": sha256(source),
                "decoded_frame_count": len(all_frames),
                "fps": fps,
            },
            "selection": {
                "start_index": args.start_index,
                "stop_index_exclusive": stop,
                "frame_count": args.frame_count,
                "width": args.width,
                "height": args.height,
                "yaw_shift_pixels": args.yaw_shift,
            },
            "direct_frames": direct_records,
            "yaw180_frames": yaw_records,
            "videos": {
                "direct": {"path": direct_video.name, "sha256": sha256(direct_video)},
                "yaw180": {"path": yaw_video.name, "sha256": sha256(yaw_video)},
            },
            "verdict": "PASS",
        }
        (staging / "input_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def finalize(args: argparse.Namespace) -> dict[str, object]:
    input_dir = args.input_frames.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    input_paths = sorted(input_dir.glob("*.png"))
    expected_names = [f"{index:05d}.png" for index in range(args.expected_count)]
    if [path.name for path in input_paths] != expected_names:
        raise ValueError("input frames are not a contiguous zero-based PNG sequence")
    inputs = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in input_paths]
    if any(frame is None for frame in inputs):
        raise ValueError("cannot decode one or more source PNG frames")
    direct, direct_fps = read_video(args.direct_video.expanduser().resolve())
    yaw, yaw_fps = read_video(args.yaw_video.expanduser().resolve())
    for actual, label in ((direct_fps, "direct"), (yaw_fps, "yaw180")):
        if not math.isclose(actual, args.fps, rel_tol=0.0, abs_tol=0.02):
            raise ValueError(f"{label} enhanced fps is {actual}, expected {args.fps}")
    for frames, label in ((inputs, "input"), (direct, "direct"), (yaw, "yaw180")):
        validate_frames(
            frames,
            expected_count=args.expected_count,
            expected_width=args.width,
            expected_height=args.height,
            label=label,
        )
    yaw_unrolled = [np.roll(frame, -args.yaw_shift, axis=1) for frame in yaw]
    fused = [fuse_frame(left, right) for left, right in zip(direct, yaw_unrolled)]
    ensemble = [
        apply_seam_guard(enhanced, source, args.seam_guard_pixels)
        for enhanced, source in zip(fused, inputs)
    ]
    comparisons = [np.concatenate([source, result], axis=1) for source, result in zip(inputs, ensemble)]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        ensemble_dir = staging / "ensemble_frames"
        write_png_sequence(ensemble_dir, ensemble)
        comparison_dir = staging / "comparison_frames"
        write_png_sequence(comparison_dir, comparisons)
        video = staging / args.output_name
        comparison = staging / args.comparison_name
        encode_frames(ensemble_dir, video, fps=args.fps, frame_count=args.expected_count, lossless=False)
        encode_frames(
            comparison_dir,
            comparison,
            fps=args.fps,
            frame_count=args.expected_count,
            lossless=False,
        )
        seam_report = evaluate_seams(inputs, ensemble)
        (staging / "seam_qc.json").write_text(
            json.dumps(seam_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            from scripts.evaluate_temporal_panorama import evaluate as evaluate_temporal
        except ModuleNotFoundError:
            # Keep the executable script usable as either ``python -m scripts``
            # or ``python scripts/...py`` from the repository root.
            from evaluate_temporal_panorama import evaluate as evaluate_temporal

        temporal_report = evaluate_temporal(
            input_dir=input_dir,
            prediction_dir=ensemble_dir,
            output=staging / "temporal_qc.json",
            required_boundaries=tuple(args.required_boundary),
        )
        checks = {
            "erp_seam": seam_report["verdict"] == "PASS",
            "temporal": temporal_report["verdict"] == "PASS",
        }
        manifest = {
            "schema": "artifixer.venhancer_erp_yaw_ensemble_result_v1",
            "method": {
                "model": "VEnhancer v2",
                "native_yaw_pass": True,
                "preencode_yaw_shift_pass_pixels": args.yaw_shift,
                "fusion": "two-plateau circular raised-cosine",
                "source_locked_seam_guard_pixels_per_side": args.seam_guard_pixels,
            },
            "frame_count": args.expected_count,
            "resolution": [args.width, args.height],
            "fps": args.fps,
            "outputs": {
                "video": {
                    "path": video.name,
                    "sha256": sha256(video),
                    "size_bytes": video.stat().st_size,
                },
                "comparison": {
                    "path": comparison.name,
                    "sha256": sha256(comparison),
                    "size_bytes": comparison.stat().st_size,
                },
                "ensemble_frames": {
                    "path": ensemble_dir.name,
                    "count": args.expected_count,
                },
            },
            "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }
        (staging / "result.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--input-video", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--start-index", type=int, default=0)
    prepare_parser.add_argument("--frame-count", type=int, required=True)
    prepare_parser.add_argument("--width", type=int, default=1024)
    prepare_parser.add_argument("--height", type=int, default=512)
    prepare_parser.add_argument("--fps", type=float, default=15.0)
    prepare_parser.add_argument("--yaw-shift", type=int, default=512)
    prepare_parser.set_defaults(handler=prepare)

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--input-frames", type=Path, required=True)
    finalize_parser.add_argument("--direct-video", type=Path, required=True)
    finalize_parser.add_argument("--yaw-video", type=Path, required=True)
    finalize_parser.add_argument("--output-dir", type=Path, required=True)
    finalize_parser.add_argument("--expected-count", type=int, required=True)
    finalize_parser.add_argument("--width", type=int, default=1024)
    finalize_parser.add_argument("--height", type=int, default=512)
    finalize_parser.add_argument("--fps", type=float, default=15.0)
    finalize_parser.add_argument("--yaw-shift", type=int, default=512)
    finalize_parser.add_argument("--seam-guard-pixels", type=int, default=48)
    finalize_parser.add_argument("--required-boundary", type=int, action="append", default=[])
    finalize_parser.add_argument(
        "--output-name", default="venhancer_v2_erp_yaw_ensemble_1024x512_15fps.mp4"
    )
    finalize_parser.add_argument(
        "--comparison-name", default="input_vs_venhancer_v2_erp_yaw_ensemble_2048x512_15fps.mp4"
    )
    finalize_parser.set_defaults(handler=finalize)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        report = args.handler(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"error: {error}") from error
    print(json.dumps({"verdict": report["verdict"]}, sort_keys=True))
    raise SystemExit(0 if report["verdict"] == "PASS" else 2)


if __name__ == "__main__":
    main()

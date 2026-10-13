"""GUI adapter for the single detector and rectification pipeline in id_card_crop."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import id_card_crop as card_algorithm

PRINT_SIZE = (957, 614)  # 8.1 x 5.2 cm at 300 DPI
RECOGNITION_DPI = 300
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class CropResult:
    status: str
    message: str
    output_path: Path | None = None
    diagnostic_path: Path | None = None
    confidence: float = 0.0
    rotation: int | None = None
    quality_warning: str | None = None
    preview_path: Path | None = None
    debug_dir: Path | None = None
    rectangle: tuple[float, float, float, float, float] | None = None
    recognition_enhanced_path: Path | None = None


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _save_image(path: Path, image: np.ndarray, *, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, format="JPEG", quality=quality,
                              dpi=(300, 300), subsampling=0)


def _diagnostic_overlay(image: np.ndarray, quad: np.ndarray,
                        crop_quad: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    scale = max(1.0, image.shape[1] / 1400.0)
    cv2.polylines(overlay, [quad.astype(np.int32).reshape(-1, 1, 2)], True,
                  (0, 190, 255), max(2, round(scale * 2)))
    cv2.polylines(overlay, [crop_quad.astype(np.int32).reshape(-1, 1, 2)], True,
                  (0, 220, 0), max(2, round(scale * 2)))
    for index, point in enumerate(crop_quad):
        center = tuple(np.rint(point).astype(int))
        cv2.circle(overlay, center, max(4, round(scale * 5)), (0, 0, 255), -1)
        cv2.putText(overlay, str(index + 1), center, cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.6, scale * 0.8), (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def process_image(input_path: str | Path, output_dir: str | Path) -> CropResult:
    """Run id_card_crop's full detector, edge refinement, flattening and crop."""
    source = Path(input_path)
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    debug_dir = destination / f"{source.stem}_debug"
    output_path = destination / f"{source.stem}_8.1x5.2cm.jpg"
    diagnostic_path = destination / f"{source.stem}_diagnostic.jpg"
    info = {
        "source": str(source),
        "algorithm": "id_card_crop.process_image_data",
        "target_cm": [8.1, 5.2],
        "dpi": 300,
        "shrink_fraction": 0.0,
    }

    try:
        image = _load_image(source)
        result = card_algorithm.process_image_data(
            image, source_path=source, dpi=300, shrink=0.0, verbose=False,
        )
        if result is None:
            info.update(status="manual_review", message="신분증 외곽을 찾지 못했습니다.")
            debug_dir.mkdir(parents=True, exist_ok=True)
            _save_image(debug_dir / "00_source.jpg", image)
            (debug_dir / "run.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            return CropResult("manual_review", info["message"],
                              diagnostic_path=diagnostic_path, debug_dir=debug_dir)

        quad = np.asarray(result["quad"], dtype=np.float32)
        crop_quad = np.asarray(result["crop_quad"], dtype=np.float32)
        corrected = result["image"]
        _save_image(output_path, corrected, quality=100)
        _save_image(diagnostic_path, _diagnostic_overlay(image, quad, crop_quad))
        debug_dir.mkdir(parents=True, exist_ok=True)
        _save_image(debug_dir / "00_source.jpg", image)
        _save_image(debug_dir / "01_card_boundary_and_crop.jpg",
                    _diagnostic_overlay(image, quad, crop_quad))
        _save_image(debug_dir / "02_final.jpg", corrected, quality=100)

        warning = None
        if min(image.shape[:2]) < min(PRINT_SIZE):
            warning = "원본 해상도가 낮아 확대 출력 시 선명도가 제한될 수 있습니다."
        info.update(
            status="complete",
            message="id_card_crop 알고리즘으로 보정했습니다.",
            confidence=float(result["confidence"]),
            rotation=int(result["rotation"]),
            source_size_px=[int(image.shape[1]), int(image.shape[0])],
            output_size_px=[int(corrected.shape[1]), int(corrected.shape[0])],
            output_path=str(output_path),
            outer_corners_xy=quad.tolist(),
            crop_corners_xy=crop_quad.tolist(),
            quality_warning=warning,
        )
        (debug_dir / "run.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return CropResult("complete", info["message"], output_path,
                          diagnostic_path, float(result["confidence"]),
                          int(result["rotation"]), warning, debug_dir=debug_dir)
    except (OSError, ValueError, cv2.error) as exc:
        info.update(status="failed", message=str(exc))
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "run.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return CropResult("failed", f"이미지 처리 실패: {exc}",
                          debug_dir=debug_dir)


def _image_metrics(image: np.ndarray) -> dict:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low, median, high = np.percentile(lab[:, :, 0], [1, 50, 99])
    lightness_p10, lightness_p90 = np.percentile(lab[:, :, 0], [10, 90])
    saturation_p50, saturation_p90, saturation_p99 = np.percentile(
        hsv[:, :, 1], [50, 90, 99],
    )
    return {
        "lightness_mean": round(float(lab[:, :, 0].mean()), 2),
        "lightness_p01_p50_p99": [round(float(v), 2) for v in (low, median, high)],
        "lightness_p10_p90": [round(float(lightness_p10), 2), round(float(lightness_p90), 2)],
        "saturation_mean": round(float(hsv[:, :, 1].mean()), 2),
        "saturation_p50_p90_p99": [
            round(float(v), 2) for v in (saturation_p50, saturation_p90, saturation_p99)
        ],
        "laplacian_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    }


# Numeric before/after profiles from examples that were meaningfully edited.
# The first pair was effectively unchanged and is used as the no-op case.
# Only measurements are retained; no source ID photographs are embedded.
_PAIR_PERCENTILES = np.array(
    [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.5],
    dtype=np.float32,
)
_PAIR_LIGHTNESS_TARGET = np.array(
    [0, 0, 0, 0, 23, 122, 162, 185, 202, 215, 227, 239, 250, 255, 255, 255],
    dtype=np.float32,
)
_PAIR3_LIGHTNESS_TARGET = np.array(
    [0, 0, 0, 1, 6, 93, 152, 178, 196, 209, 220, 230, 242, 249, 254, 255],
    dtype=np.float32,
)
_PHOTO_LIGHTNESS_TARGET = np.array(
    [1, 2, 9, 17, 33, 99, 160, 185, 204, 220, 234, 244, 251, 253, 255, 255],
    dtype=np.float32,
)
_CALIBRATION_INPUTS = np.array([
    # lightness mean, p10, p50, p90, saturation mean
    [158.5, 41, 176, 222, 69.4],  # 비포2
    [164.1, 42, 187, 225, 51.2],  # 비포3
    [130.34, 41, 145, 182, 51.5],  # photo source
], dtype=np.float32)
_CALIBRATION_OUTPUTS = np.stack([
    _PAIR_LIGHTNESS_TARGET,
    _PAIR3_LIGHTNESS_TARGET,
    _PHOTO_LIGHTNESS_TARGET,
])
_CALIBRATION_SATURATION_GAINS = np.array([1.3602, 1.5160, 1.3612], dtype=np.float32)
_CALIBRATION_FEATURE_SCALES = np.array([15, 15, 25, 30, 15], dtype=np.float32)


def _select_pair_profile(metrics: dict) -> tuple[np.ndarray, float, list[float]]:
    """Interpolate the closest human-edited pair profiles for this input."""
    p10, p50, p90 = metrics["lightness_p10_p90"][0], \
        metrics["lightness_p01_p50_p99"][1], metrics["lightness_p10_p90"][1]
    features = np.array([
        metrics["lightness_mean"], p10, p50, p90, metrics["saturation_mean"],
    ], dtype=np.float32)
    distances = np.sum(
        ((_CALIBRATION_INPUTS - features) / _CALIBRATION_FEATURE_SCALES) ** 2,
        axis=1,
    )
    weights = np.exp(-distances / (2.0 * 0.75 ** 2))
    weights /= max(float(weights.sum()), 1e-12)
    output_profile = weights @ _CALIBRATION_OUTPUTS
    saturation_gain = float(weights @ _CALIBRATION_SATURATION_GAINS)
    return output_profile.astype(np.float32), saturation_gain, [
        round(float(v), 5) for v in weights
    ]


def _paired_levels_lut(lightness: np.ndarray, target_profile: np.ndarray) -> tuple[np.ndarray, dict]:
    """Build a pointwise Levels curve from this image to its paired profile."""
    source_points = np.percentile(lightness, _PAIR_PERCENTILES).astype(np.float32)
    target_points = np.maximum.accumulate(target_profile)

    x_points = np.concatenate(([0.0], source_points, [255.0]))
    y_points = np.concatenate(([0.0], target_points, [255.0]))
    # Quantized/flat images can repeat percentile input values. Merge such
    # anchors without creating a backwards or discontinuous tone curve.
    unique_x: list[float] = []
    unique_y: list[float] = []
    for x_value, y_value in zip(x_points, y_points):
        if unique_x and x_value <= unique_x[-1]:
            unique_y[-1] = max(unique_y[-1], float(y_value))
        else:
            unique_x.append(float(x_value))
            unique_y.append(float(y_value))
    unique_y = np.maximum.accumulate(np.asarray(unique_y, dtype=np.float32)).tolist()
    lut = np.interp(np.arange(256), unique_x, unique_y)
    lut = np.clip(np.maximum.accumulate(lut), 0, 255).astype(np.uint8)
    details = {
        "input_percentiles": [float(v) for v in _PAIR_PERCENTILES],
        "input_levels": [round(float(v), 2) for v in source_points],
        "output_levels": [round(float(v), 2) for v in target_points],
        "black_input_level": round(float(source_points[4]), 2),
        "midtone_input_level": round(float(source_points[8]), 2),
        "white_input_level": round(float(source_points[14]), 2),
    }
    return lut[lightness], details


def _scale_lab_chroma_to_mean(image: np.ndarray, target_saturation: float) -> tuple[np.ndarray, float]:
    """Increase color while retaining the Levels-adjusted lightness channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    original_a, original_b = lab[:, :, 1].copy(), lab[:, :, 2].copy()

    def render(gain: float) -> np.ndarray:
        candidate = lab.copy()
        candidate[:, :, 1] = np.clip(128 + (original_a - 128) * gain, 0, 255)
        candidate[:, :, 2] = np.clip(128 + (original_b - 128) * gain, 0, 255)
        return cv2.cvtColor(candidate.astype(np.uint8), cv2.COLOR_LAB2BGR)

    low_gain, high_gain = 0.5, 3.0
    for _ in range(18):
        gain = (low_gain + high_gain) / 2.0
        candidate = render(gain)
        measured = float(cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
        if measured < target_saturation:
            low_gain = gain
        else:
            high_gain = gain
    gain = (low_gain + high_gain) / 2.0
    return render(gain), gain


def _auto_enhance_card(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply a learned Levels curve and restrained chroma from manual examples."""
    image = np.uint8(np.clip(image, 0, 255))
    before = _image_metrics(image)
    lightness_low, _, lightness_high = before["lightness_p01_p50_p99"]
    saturation_mean = before["saturation_mean"]
    contrast_span = lightness_high - lightness_low

    # The first before/after pair changed negligibly. Keep images with a healthy
    # tonal range untouched; use the other two pairs only for visibly weak input.
    needs_enhancement = (
        before["lightness_mean"] < 166.0
        or saturation_mean < 58.0
        or contrast_span < 230.0
    )
    log: list[str] = []
    lightness_curve = None
    chroma_gain = 1.0
    target_saturation_mean = saturation_mean

    if needs_enhancement:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        target_profile, saturation_gain, profile_weights = _select_pair_profile(before)
        corrected_l, lightness_curve = _paired_levels_lut(lightness, target_profile)
        leveled = cv2.cvtColor(cv2.merge([corrected_l, channel_a, channel_b]),
                               cv2.COLOR_LAB2BGR)
        log.append("paired_levels_curve")

        # The manual examples' color increase is best represented by their
        # median before/after saturation ratio. Apply it in LAB chroma so the
        # Levels-adjusted lightness does not collapse as it did with HSV scaling.
        target_saturation_mean = min(saturation_mean * saturation_gain, 110.0)
        enhanced, chroma_gain = _scale_lab_chroma_to_mean(
            leveled, target_saturation_mean,
        )
        log.append("paired_chroma_gain")
    else:
        # The lightly edited first pair is the reference behavior for good input.
        enhanced = image.copy()
        log.append("quality_already_adequate_noop")

    details = {
        "method": "paired_before_after_percentile_transfer",
        "input_scope": "selected_image",
        "metrics_before": before,
        "metrics_after": _image_metrics(enhanced),
        "quality_triggered": needs_enhancement,
        "actions": log,
        "calibration": {
            "meaningfully_edited_pairs": [
                "비포2→애프터2", "비포3→애프터3", "photo→애프타",
            ],
            "near_noop_pair": "비포1→애프터1",
            "percentiles": [float(v) for v in _PAIR_PERCENTILES],
            "profile_weights": profile_weights if needs_enhancement else [0, 0, 0],
            "selected_lightness_after_profile": [float(v) for v in target_profile]
            if needs_enhancement else None,
            "source_images_embedded": False,
        },
        "applied": {
            "correction_triggered": needs_enhancement,
            "saturation_mean_before": round(saturation_mean, 2),
            "saturation_mean_target": round(target_saturation_mean, 2),
            "lab_chroma_gain": round(float(chroma_gain), 4),
            "lightness_curve": lightness_curve,
        },
    }
    return enhanced, details


def enhance_for_recognition(input_path: str | Path,
                            output_path: str | Path,
                            original_path: str | Path | None = None) -> tuple[Path, dict]:
    """Apply recognition-oriented tonal corrections without detecting or cropping."""
    source = Path(original_path) if original_path else Path(input_path)
    destination = Path(output_path)
    image = _load_image(source)
    enhanced, details = _auto_enhance_card(image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(destination, format="JPEG", quality=98,
                              dpi=(RECOGNITION_DPI, RECOGNITION_DPI), subsampling=0)
    details.update({
        "input_scope": "selected_input_without_additional_crop",
        "source_image": str(source),
        "source_size_px": [int(image.shape[1]), int(image.shape[0])],
        "output_size_px": [int(enhanced.shape[1]), int(enhanced.shape[0])],
        "output_dpi": RECOGNITION_DPI,
        "dpi_is_metadata_only": True,
        "spatial_processing": "none: original dimensions retained, no crop, rotation, or rescale",
    })
    return destination, details

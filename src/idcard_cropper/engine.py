"""Conservative card detection and print-size correction using OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
from PIL import Image, ImageOps


PRINT_SIZE = (957, 602)  # 8.1 x 5.1 cm at 300 DPI
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


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    return np.array(
        [points[np.argmin(sums)], points[np.argmin(diffs)],
         points[np.argmax(sums)], points[np.argmax(diffs)]], dtype=np.float32
    )  # top-left, top-right, bottom-right, bottom-left


def _is_nearly_parallel_card(quad: np.ndarray) -> bool:
    """Return true when a scan is a rotated rectangle with negligible perspective."""
    tl, tr, br, bl = quad
    edges = (tr - tl, br - bl, bl - tl, br - tr)
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    if min(lengths) <= 0:
        return False

    def direction_delta(a: np.ndarray, b: np.ndarray) -> float:
        angle_a = float(np.degrees(np.arctan2(a[1], a[0])))
        angle_b = float(np.degrees(np.arctan2(b[1], b[0])))
        return abs((angle_a - angle_b + 90.0) % 180.0 - 90.0)

    width_delta = abs(lengths[0] - lengths[1]) / max(lengths[0], lengths[1])
    height_delta = abs(lengths[2] - lengths[3]) / max(lengths[2], lengths[3])
    return (max(width_delta, height_delta) <= 0.04
            and direction_delta(edges[0], edges[1]) <= 3.0
            and direction_delta(edges[2], edges[3]) <= 3.0)


def _move_corners_inside_component(
    mask: np.ndarray, quad: np.ndarray, details: dict | None = None,
) -> np.ndarray | None:
    """Inset all four corners equally until each lies inside the detected card."""
    tl, tr, br, bl = quad

    def unit(vector: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(vector))
        return vector / length if length > 0 else vector

    inward_axes = (
        (unit(tr - tl), unit(bl - tl)),
        (unit(tl - tr), unit(br - tr)),
        (unit(tr - br), unit(bl - br)),
        (unit(br - bl), unit(tl - bl)),
    )
    side_lengths = (np.linalg.norm(tr - tl), np.linalg.norm(br - bl),
                    np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    max_inset = max(1, int(round(min(side_lengths) * 0.05)))
    required: list[int] = []
    height, width = mask.shape[:2]

    for corner, (axis_a, axis_b) in zip(quad, inward_axes):
        entry = None
        for distance in range(max_inset + 1):
            point = np.rint(corner + distance * (axis_a + axis_b)).astype(int)
            x, y = int(point[0]), int(point[1])
            if not (0 <= x < width and 0 <= y < height):
                continue
            # Require a short continuous run so a stray background speck
            # cannot make a rounded corner look like part of the card.
            run = True
            for extra in (1, 2):
                sample = np.rint(corner + (distance + extra) * (axis_a + axis_b)).astype(int)
                sx, sy = int(sample[0]), int(sample[1])
                if not (0 <= sx < width and 0 <= sy < height and mask[sy, sx]):
                    run = False
                    break
            if mask[y, x] and run:
                entry = distance
                break
        if entry is None:
            return None
        required.append(entry)

    # One shared inset keeps all four crop edges parallel. Use the deepest
    # corner entry so every point is inside the card rather than on scan paper.
    inset = max(required)
    if details is not None:
        details.update(corner_entry_px=required, shared_inset_px=inset)
    adjusted = np.array([
        corner + inset * (axis_a + axis_b)
        for corner, (axis_a, axis_b) in zip(quad, inward_axes)
    ], dtype=np.float32)
    return adjusted


def _deskew_scan(image: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the scan so the detected card edge is horizontal before cropping."""
    edge = corners[1] - corners[0]
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
    if abs(angle) < 0.05:
        return image, corners

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(round(height * sine + width * cosine))
    new_height = int(round(height * cosine + width * sine))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]

    rotated = cv2.warpAffine(
        image, matrix, (new_width, new_height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    aligned_corners = cv2.transform(corners[None, :, :], matrix)[0]
    return rotated, aligned_corners


def _background_lab(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    band = max(2, int(min(h, w) * 0.025))
    border_pixels = np.concatenate((
        image[:band].reshape(-1, 3), image[-band:].reshape(-1, 3),
        image[:, :band].reshape(-1, 3), image[:, -band:].reshape(-1, 3),
    ))
    border = cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    return np.median(border.reshape(-1, 3), axis=0).astype(np.float32)


def _candidate_contour(
    image: np.ndarray, debug: dict | None = None,
) -> tuple[np.ndarray | None, float]:
    """Find the dominant non-background component and fit its outer quadrilateral."""
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = _background_lab(image)
    distance = np.linalg.norm(lab - bg[None, None, :], axis=2)
    k = max(3, int(round(min(h, w) * 0.004)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    best: tuple[np.ndarray, float] | None = None
    best_details: dict | None = None
    thresholds_tried: list[float] = []

    # Multiple thresholds handle both faint card edges and patterned cards.
    for threshold in (7.0, 10.0, 14.0, 19.0, 25.0, 32.0):
        thresholds_tried.append(threshold)
        mask = (distance >= threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < max(500, int(h * w * 0.00025)) or area > h * w * 0.8:
            continue
        component = np.where(labels == idx, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        quad = None
        fit_mode = "contour_quad"
        for epsilon in (0.012, 0.018, 0.024, 0.032, 0.042, 0.055):
            approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                points = approx.reshape(4, 2).astype(np.float32)
                ordered = _order_points(points)
                widths = (np.linalg.norm(ordered[1] - ordered[0]),
                          np.linalg.norm(ordered[2] - ordered[3]))
                heights = (np.linalg.norm(ordered[3] - ordered[0]),
                           np.linalg.norm(ordered[2] - ordered[1]))
                ratio = max(np.mean(widths), np.mean(heights)) / max(1.0, min(np.mean(widths), np.mean(heights)))
                if 1.35 <= ratio <= 2.05:
                    # Scans are usually a rotated rectangle, not a perspective
                    # view. A minimum-area rectangle keeps opposite crop edges
                    # parallel and avoids one corner being pulled inward by a
                    # rounded/low-contrast card edge. Keep the fitted quad for
                    # genuine perspective distortion.
                    if _is_nearly_parallel_card(ordered):
                        ordered = _order_points(cv2.boxPoints(cv2.minAreaRect(hull)))
                        fit_mode = "minimum_area_rectangle"
                    quad = ordered
                    break
        if quad is None:
            continue

        outer_quad = quad.copy()
        inset_details: dict = {}
        inner_quad = _move_corners_inside_component(component, quad, inset_details)
        if inner_quad is None:
            continue

        qarea = abs(cv2.contourArea(inner_quad.reshape(-1, 1, 2)))
        fill = min(1.0, cv2.contourArea(contour) / max(qarea, 1.0))
        # Prefer a substantial, compact component; weak fits are reviewed manually.
        score = min(1.0, area / max(h * w * 0.015, 1.0)) * (0.65 + 0.35 * fill)
        if best is None or score > best[1]:
            best = (inner_quad, float(score))
            best_details = {
                "outer_corners_xy": outer_quad.tolist(),
                "inner_corners_xy": inner_quad.tolist(),
                "selected_threshold_lab": threshold,
                "component_area_px": area,
                "component_fill_ratio": float(fill),
                "fit_mode": fit_mode,
                **inset_details,
                "component_mask": component,
            }
    if debug is not None:
        debug["thresholds_tried_lab"] = thresholds_tried
        if best_details:
            debug.update(best_details)
    return best if best else (None, 0.0)


def _save_debug_image(folder: Path, name: str, image: np.ndarray) -> Path:
    path = folder / name
    if image.ndim == 2:
        Image.fromarray(image.astype(np.uint8)).save(path)
    else:
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path, quality=94)
    return path


def _draw_debug_corners(image: np.ndarray, corners: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    marked = image.copy()
    cv2.polylines(marked, [corners.astype(np.int32)], True, color, 3)
    for index, point in enumerate(corners.astype(int)):
        cv2.circle(marked, tuple(point), 5, (0, 0, 255), -1)
        cv2.putText(marked, str(index + 1), tuple(point + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    return marked


def _write_debug_log(folder: Path, info: dict) -> None:
    path = folder / "run.json"

    def to_json(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        raise TypeError(f"Unsupported debug value: {type(value).__name__}")

    path.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=to_json), encoding="utf-8")


def _warp(image: np.ndarray, corners: np.ndarray, debug: dict | None = None) -> np.ndarray:
    tl, tr, br, bl = corners
    width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if width < 2 or height < 2:
        raise ValueError("검출된 카드 경계의 크기가 너무 작습니다.")
    destination = np.array([[0, 0], [width - 1, 0],
                            [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    if debug is not None:
        debug["perspective_transform_matrix"] = matrix
        debug["warp_destination_size_px"] = [width, height]
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def _read_rotation(
    card: np.ndarray, score_log: list[dict] | None = None,
) -> tuple[int | None, str | None]:
    """Choose orientation from local OCR confidence, without retaining recognized text."""
    try:
        import pytesseract
        from pytesseract import Output
        app_root = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
        if getattr(sys, "frozen", False):
            bundled_tesseract = app_root / "tesseract" / "tesseract.exe"
            if bundled_tesseract.is_file():
                pytesseract.pytesseract.tesseract_cmd = str(bundled_tesseract)
            else:
                return None, "패키지에 포함된 Tesseract 실행 파일을 찾지 못했습니다. 다시 빌드해야 합니다."

        tessdata_dir = app_root / "tessdata"
        korean_model = tessdata_dir / "kor.traineddata"
        if not korean_model.is_file():
            tessdata_dir = Path(__file__).parent / "tessdata"
            korean_model = tessdata_dir / "kor.traineddata"
        if korean_model.is_file():
            # Passing an absolute Windows path through pytesseract's config
            # parser strips backslashes or retains quotes, so use the
            # TESSDATA_PREFIX environment variable instead.
            import os
            os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
            try:
                pytesseract.get_tesseract_version()
            except Exception as exc:
                return None, f"Tesseract 실행 실패 ({type(exc).__name__}): {str(exc).splitlines()[0][:160]}"
            # OSD is unreliable on Korean identity cards. Compare only local
            # recognition confidence across rotations; never expose recognized text.
            h, w = card.shape[:2]
            # Do not enlarge OCR input: all tested scales chose the same
            # orientation for the sample, while upscaling took longer. Cap
            # only very large cards to keep OCR work bounded.
            scale = min(1.0, 1400.0 / max(h, w))
            enlarged = (cv2.resize(card, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
                        if scale < 1.0 else card)
            scores: list[tuple[int, float, int]] = []
            # The print format is landscape, so a portrait crop only needs the
            # two rotations that make it landscape. A landscape crop compares
            # its current orientation with the upside-down alternative.
            if w >= h:
                rotations = {
                    0: enlarged,
                    180: cv2.rotate(enlarged, cv2.ROTATE_180),
                }
            else:
                rotations = {
                    90: cv2.rotate(enlarged, cv2.ROTATE_90_CLOCKWISE),
                    270: cv2.rotate(enlarged, cv2.ROTATE_90_COUNTERCLOCKWISE),
                }
            for degrees, candidate in rotations.items():
                data = pytesseract.image_to_data(
                    candidate, lang="kor", output_type=Output.DICT,
                    config="--psm 6",
                )
                confidences = [float(conf) for conf, word in zip(data["conf"], data["text"])
                               if word.strip() and float(conf) >= 0]
                if confidences:
                    average = sum(confidences) / len(confidences)
                    scores.append((degrees, average, len(confidences)))
                    if score_log is not None:
                        score_log.append({"rotation_degrees": degrees,
                                          "mean_confidence": average,
                                          "recognized_token_count": len(confidences)})
            scores.sort(key=lambda item: item[1], reverse=True)
            if scores:
                best = scores[0]
                runner_up = scores[1][1] if len(scores) > 1 else 0.0
                if best[1] >= 35.0 and best[2] >= 4 and best[1] - runner_up >= 12.0:
                    return best[0], None
            return None, "OCR 방향 점수 차이가 작아 회전 확인이 필요합니다."

        if getattr(sys, "frozen", False):
            return None, "패키지에 포함된 한국어 OCR 모델을 찾지 못했습니다. 다시 빌드해야 합니다."

        data = pytesseract.image_to_osd(card, output_type=Output.DICT)
        confidence = float(data.get("orientation_conf", 0))
        rotation = int(data.get("rotate", 0)) % 360
        if score_log is not None:
            score_log.append({"osd_rotation_degrees": rotation,
                              "orientation_confidence": confidence})
        if rotation in (0, 90, 180, 270) and confidence >= 2.0:
            return rotation, None
        return None, "문자 방향 판별 신뢰도가 낮아 회전 확인이 필요합니다."
    except Exception as exc:
        detail = " ".join(str(exc).splitlines())[:160]
        return None, f"Tesseract OCR 오류 ({type(exc).__name__}): {detail}"


def _fit_print_size(card: np.ndarray) -> np.ndarray:
    target_w, target_h = PRINT_SIZE
    h, w = card.shape[:2]
    shrinking = target_w < w or target_h < h
    # Fill the print dimensions exactly; this avoids retaining scan background
    # as letterboxing. The small aspect-ratio adjustment is explicit and avoids cropping.
    return cv2.resize(card, (target_w, target_h),
                      interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)


def _save_crop_preview(card: np.ndarray, source: Path, destination: Path) -> Path:
    """Save a card-only preview at print dimensions, even if OCR needs review."""
    if card.shape[0] > card.shape[1]:
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    preview = _fit_print_size(card)
    path = destination / f"{source.stem}_crop_preview.jpg"
    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, format="JPEG", quality=95, dpi=(300, 300), subsampling=0)
    return path


def process_image(input_path: str | Path, output_dir: str | Path) -> CropResult:
    """Process one JPG/PNG. Uncertain detection or orientation never saves a final image."""
    source = Path(input_path)
    destination = Path(output_dir)
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        return CropResult("failed", "JPG, JPEG, PNG 파일만 지원합니다.")
    try:
        image = _load_image(source)
    except Exception:
        return CropResult("failed", "이미지를 열 수 없습니다.")

    destination.mkdir(parents=True, exist_ok=True)
    debug_dir = destination / f"{source.stem}_debug"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {
        "source_file": source.name,
        "source_size_px": [int(image.shape[1]), int(image.shape[0])],
        "status": "processing",
        "ocr_text_saved": False,
    }
    _save_debug_image(debug_dir, "00_source.jpg", image)

    detection: dict = {}
    corners, confidence = _candidate_contour(image, detection)
    component_mask = detection.pop("component_mask", None)
    info["detection"] = {key: value for key, value in detection.items()}
    info["detection_confidence"] = confidence
    if "outer_corners_xy" in detection:
        outer = np.asarray(detection["outer_corners_xy"], dtype=np.float32)
        _save_debug_image(debug_dir, "01_outer_candidate_points.jpg",
                          _draw_debug_corners(image, outer, (0, 180, 255)))
    if corners is not None:
        _save_debug_image(debug_dir, "02_inner_crop_points.jpg",
                          _draw_debug_corners(image, corners, (0, 200, 0)))
    if component_mask is not None:
        _save_debug_image(debug_dir, "03_selected_component_mask.png", component_mask)

    if corners is not None and confidence >= 0.42:
        edge = corners[1] - corners[0]
        info["deskew_angle_degrees"] = float(np.degrees(np.arctan2(edge[1], edge[0])))
        image, corners = _deskew_scan(image, corners)
        info["deskewed_size_px"] = [int(image.shape[1]), int(image.shape[0])]
        info["deskewed_corners_xy"] = corners
        _save_debug_image(debug_dir, "04_deskewed_points.jpg",
                          _draw_debug_corners(image, corners, (0, 200, 0)))

    diagnostic_path = destination / f"{source.stem}_diagnostic.jpg"
    diagnostic = image.copy()
    if corners is not None:
        diagnostic = _draw_debug_corners(diagnostic, corners, (0, 200, 0))
    Image.fromarray(cv2.cvtColor(diagnostic, cv2.COLOR_BGR2RGB)).save(diagnostic_path, quality=92)

    if corners is None or confidence < 0.42:
        info.update(status="manual_review", message="카드 외곽을 확실하게 검출하지 못했습니다.")
        _write_debug_log(debug_dir, info)
        return CropResult("manual_review", "카드 외곽을 확실하게 검출하지 못했습니다.",
                          diagnostic_path=diagnostic_path, confidence=confidence,
                          debug_dir=debug_dir)
    try:
        card = _warp(image, corners, info)
    except Exception:
        info.update(status="manual_review", message="원근 보정에 실패했습니다.")
        _write_debug_log(debug_dir, info)
        return CropResult("manual_review", "원근 보정에 실패했습니다.",
                          diagnostic_path=diagnostic_path, confidence=confidence,
                          debug_dir=debug_dir)

    info["warped_card_size_px"] = [int(card.shape[1]), int(card.shape[0])]
    _save_debug_image(debug_dir, "05_warped_card_before_orientation.jpg", card)

    rotation_scores: list[dict] = []
    rotation, orientation_warning = _read_rotation(card, rotation_scores)
    info["ocr_rotation_scores"] = rotation_scores
    info["ocr_selected_rotation_degrees"] = rotation
    if rotation is None:
        preview_path = _save_crop_preview(card, source, destination)
        preview = _load_image(preview_path)
        _save_debug_image(debug_dir, "06_orientation_review_preview.jpg", preview)
        info.update(status="manual_review", message=orientation_warning or "방향을 확인해야 합니다.",
                    preview_path=preview_path)
        _write_debug_log(debug_dir, info)
        return CropResult("manual_review", orientation_warning or "방향을 확인해야 합니다.",
                          diagnostic_path=diagnostic_path, confidence=confidence,
                          preview_path=preview_path, debug_dir=debug_dir)
    applied_rotation = rotation or 0
    if applied_rotation == 90:
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    elif applied_rotation == 180:
        card = cv2.rotate(card, cv2.ROTATE_180)
    elif applied_rotation == 270:
        card = cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Landscape print dimensions; rotate portrait cards to landscape without altering content.
    if card.shape[0] > card.shape[1]:
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    info["upright_card_size_px"] = [int(card.shape[1]), int(card.shape[0])]
    _save_debug_image(debug_dir, "06_upright_card_before_resize.jpg", card)
    warning = None
    if min(card.shape[:2]) < 602:
        warning = "원본 카드 해상도가 낮아 확대 출력 시 선명도가 제한됩니다."
    final = _fit_print_size(card)
    output_path = destination / f"{source.stem}_8.1x5.1cm.jpg"
    rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(output_path, format="JPEG", quality=95, dpi=(300, 300), subsampling=0)
    info.update(status="complete", message="보정 및 인화 규격 변환이 완료되었습니다.",
                quality_warning=warning, output_path=output_path,
                print_size_px=list(PRINT_SIZE), output_dpi=300)
    _save_debug_image(debug_dir, "07_final_8.1x5.1cm.jpg", final)
    _write_debug_log(debug_dir, info)
    return CropResult("complete", "보정 및 인화 규격 변환이 완료되었습니다.", output_path,
                      diagnostic_path, confidence, applied_rotation, warning,
                      debug_dir=debug_dir)

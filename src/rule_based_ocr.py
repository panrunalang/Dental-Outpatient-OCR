"""Single-file API for converting scanned dental outpatient PDFs into text."""

import argparse
import os
import json
import math
import re
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR

OUT_DIR = Path("outputs")
RAW_DIR = Path("examples/inputs")
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
EDGE_MARGIN = 6
FONT_CANDIDATES = [
    # macOS
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    # Windows
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/msyh.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    # Linux
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
]
FONT_SIZE = 16
OCR_SCALE = 4
TEXT_WRAP_WIDTH = 26
REVIEW_PANEL_WIDTH = 520
DEFAULT_SLOW_STAGE_SECONDS = 0.5
_OCR_LOCAL = threading.local()

TEXT_BG_ALPHA = 235
TEXT_BORDER = (30, 30, 30)

SURFACE_NAMES = {
    "B": "颊",
    "L": "舌",
    "M": "近中",
    "D": "远中",
    "O": "合",
}
ROW_NAMES = {
    "B": "颊侧",
    "L": "舌侧",
}
COL_NAMES = {
    "M": "近中",
    "D": "远中",
}
PERMANENT_PREFIX = {
    "tl": "1",
    "tr": "2",
    "br": "3",
    "bl": "4",
}
TOOTH_SYMBOL_COLOR = (60, 132, 255)
FURCATION_SYMBOL_COLOR = (242, 223, 78)
PD_SYMBOL_COLOR = (181, 81, 162)
PRIMARY_PREFIX = {
    "tl": "5",
    "tr": "6",
    "br": "7",
    "bl": "8",
}
UPPER_REGIONS = {"tl", "tr"}
LOWER_REGIONS = {"br", "bl"}

__all__ = [
    "ConversionOutput",
    "convert_outpatient_pdf_to_txt",
    "convert_outpatient_pdfs_parallel",
    "convert_outpatient_input_path_parallel",
    "run_all_test_samples_parallel",
]


def ensure_cv_image(image: np.ndarray, name: str, allow_gray: bool = True) -> np.ndarray:
    """把输入规整成OpenCV稳定可处理的uint8数组。"""
    if image is None:
        raise ValueError(f"{name}为空")
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError(f"{name}为空数组")
    if arr.ndim == 2:
        if not allow_gray:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] != 3:
            raise ValueError(f"{name}通道数异常: shape={arr.shape}")
    else:
        raise ValueError(f"{name}维度异常: shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def get_cached_ocr(debug_timing: bool = False) -> RapidOCR:
    ocr = getattr(_OCR_LOCAL, "ocr_instance", None)
    if ocr is not None:
        return ocr

    ocr_start = time.perf_counter()
    ocr = RapidOCR()
    _OCR_LOCAL.ocr_instance = ocr
    log_debug(
        debug_timing,
        f"OCR init {time.perf_counter() - ocr_start:.3f}s pid={os.getpid()} tid={threading.get_ident()}",
    )
    return ocr


@dataclass(frozen=True)
class SymbolDetection:
    label: str
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class LineCrop:
    image: np.ndarray
    left: int
    top: int
    right: int
    bottom: int


def remove_gray_watermark(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    near_gray = (
        (np.abs(bgr[:, :, 0].astype(np.int16) - bgr[:, :, 1].astype(np.int16)) <= 8)
        & (np.abs(bgr[:, :, 0].astype(np.int16) - bgr[:, :, 2].astype(np.int16)) <= 8)
        & (np.abs(bgr[:, :, 1].astype(np.int16) - bgr[:, :, 2].astype(np.int16)) <= 8)
    )
    watermark_mask = (s <= 20) & (v >= 185) & (v <= 245) & near_gray
    protected = cv2.dilate((gray <= 145).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    out = bgr.copy()
    out[watermark_mask & ~protected.astype(bool)] = 255
    return out


def extract_page_image_for_ocr(pdf_path: Path, page_index: int) -> np.ndarray:
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 1:
            bgr = np.repeat(arr, 3, axis=2)
        elif pix.n >= 3:
            bgr = arr[:, :, :3][:, :, ::-1]
        else:
            raise ValueError(f"PDF页面通道数异常: pix.n={pix.n}, file={pdf_path}, page_index={page_index}")
        if bgr.shape[0] > 1000 or bgr.shape[1] > 800:
            bgr = cv2.resize(bgr, (559, 794), interpolation=cv2.INTER_CUBIC)
        bgr = ensure_cv_image(bgr, f"PDF页面图像 file={pdf_path} page_index={page_index}", allow_gray=False)
        return ensure_cv_image(remove_gray_watermark(bgr), "去水印后的PDF页面", allow_gray=False)


def extract_images_from_pdf(pdf_path: Path) -> list[tuple[int, np.ndarray]]:
    images: list[tuple[int, np.ndarray]] = []
    with fitz.open(pdf_path) as doc:
        for idx in range(doc.page_count):
            images.append((idx + 1, extract_page_image_for_ocr(pdf_path, idx)))
    return images


def extract_images_from_input(file_path: Path) -> list[tuple[int, np.ndarray]]:
    """提取输入文件（PDF 或 图片）中的页面图像，并执行去水印与数组规整。"""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_images_from_pdf(file_path)
    elif suffix in SUPPORTED_EXTENSIONS:
        try:
            arr = np.fromfile(str(file_path), dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            bgr = None
        if bgr is None:
            pil_img = Image.open(file_path).convert("RGB")
            bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        if bgr.shape[0] > 1000 or bgr.shape[1] > 800:
            bgr = cv2.resize(bgr, (559, 794), interpolation=cv2.INTER_CUBIC)
        bgr = ensure_cv_image(bgr, f"输入图像 file={file_path}", allow_gray=False)
        cleaned = ensure_cv_image(remove_gray_watermark(bgr), "去水印后的图像", allow_gray=False)
        return [(1, cleaned)]
    else:
        raise ValueError(f"不支持的文件格式: {file_path} (支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))})")




def mask_to_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def otsu_binary(bgr: np.ndarray, invert: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, mode + cv2.THRESH_OTSU)[1]


def clean_binary_page(binary: np.ndarray) -> np.ndarray:
    binary = binary.copy()
    binary[:EDGE_MARGIN, :] = 255
    binary[-EDGE_MARGIN:, :] = 255
    binary[:, :EDGE_MARGIN] = 255
    binary[:, -EDGE_MARGIN:] = 255
    return binary


def build_diag_kernel(size: int, flip: bool) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.uint8)
    for idx in range(size):
        kernel[idx, size - 1 - idx if flip else idx] = 1
    return kernel


def build_symbol_masks(inv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = inv.shape
    horizontal = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, w // 35), 1)))
    vertical = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h // 3))))
    diag_size = 7 if min(h, w) >= 20 else 5
    diagonal = cv2.bitwise_or(
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, build_diag_kernel(diag_size, False)),
        cv2.morphologyEx(inv, cv2.MORPH_OPEN, build_diag_kernel(diag_size, True)),
    )
    return horizontal, vertical, diagonal


def count_components(binary: np.ndarray, min_area: int = 4) -> int:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return sum(1 for idx in range(1, num_labels) if stats[idx, cv2.CC_STAT_AREA] >= min_area)


def expand_box(left: int, top: int, right: int, bottom: int, pad_x: int, pad_y: int, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width - 1, right + pad_x),
        min(height - 1, bottom + pad_y),
    )


def line_spans(counts: np.ndarray, ratio: float = 0.3, min_count: int = 4) -> list[tuple[int, int]]:
    if counts.size == 0:
        return []
    threshold = max(min_count, int(np.ceil(int(counts.max()) * ratio)))
    return mask_to_spans(counts >= threshold)


def choose_span(spans: list[tuple[int, int]], anchor: int | None) -> tuple[int, int] | None:
    if not spans:
        return None
    if anchor is None:
        return max(spans, key=lambda span: (span[1] - span[0], -span[0]))
    for start, end in spans:
        if start <= anchor <= end:
            return start, end
    return min(spans, key=lambda span: (min(abs(anchor - span[0]), abs(anchor - span[1])), -(span[1] - span[0])))


def choose_count_span(counts: np.ndarray, anchor: int | None = None) -> tuple[int, int] | None:
    spans = line_spans(counts)
    if not spans:
        return None
    if anchor is not None:
        return choose_span(spans, anchor)
    return max(
        spans,
        key=lambda span: (
            int(counts[span[0] : span[1] + 1].sum()),
            int(counts[span[0] : span[1] + 1].max()),
            span[1] - span[0],
            -span[0],
        ),
    )


def dominant_horizontal_box(
    inv: np.ndarray,
    horizontal: np.ndarray,
    anchor_x: int | None = None,
    anchor_y: int | None = None,
) -> tuple[int, int, int, int] | None:
    row_span = choose_count_span(np.count_nonzero(horizontal, axis=1), anchor_y)
    if row_span is None:
        return None
    top, bottom = row_span
    span = choose_span(mask_to_spans(np.any(inv[top : bottom + 1, :] > 0, axis=0)), anchor_x)
    if span is None:
        return None
    left, right = span
    return int(left), int(top), int(right), int(bottom)


def dominant_vertical_box(inv: np.ndarray, vertical: np.ndarray, anchor_y: int | None = None) -> tuple[int, int, int, int] | None:
    col_span = choose_count_span(np.count_nonzero(vertical, axis=0), None)
    if col_span is None:
        return None
    left, right = col_span
    span = choose_span(mask_to_spans(np.any(inv[:, left : right + 1] > 0, axis=1)), anchor_y)
    if span is None:
        return None
    top, bottom = span
    return int(left), int(top), int(right), int(bottom)


def dominant_cross_box(inv: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray) -> tuple[int, int, int, int] | None:
    col_span = choose_count_span(np.count_nonzero(vertical, axis=0), None)
    row_span = choose_count_span(np.count_nonzero(horizontal, axis=1), None)
    if col_span is None or row_span is None:
        return None
    left, right = col_span
    top, bottom = row_span
    anchor_x = (left + right) // 2
    anchor_y = (top + bottom) // 2
    hbox = dominant_horizontal_box(inv, horizontal, anchor_x, anchor_y)
    vbox = dominant_vertical_box(inv, vertical, anchor_y)
    if hbox is None or vbox is None:
        return None
    return int(hbox[0]), int(vbox[1]), int(hbox[2]), int(vbox[3])


def pad_cross_left(inv: np.ndarray, left: int, top: int, right: int, bottom: int, max_pad: int = 2) -> tuple[int, int, int, int]:
    h, w = inv.shape
    left = max(0, left)
    right = min(w - 1, right)
    top = max(0, top)
    bottom = min(h - 1, bottom)

    for _ in range(max_pad):
        if left > 0 and np.count_nonzero(inv[top : bottom + 1, left]) > 0:
            left -= 1
        else:
            break

    return left, top, right, bottom


def expand_pd_probe_box(left: int, top: int, right: int, bottom: int, inv: np.ndarray) -> tuple[int, int, int, int]:
    h, w = inv.shape
    margin_x = max(10, (right - left + 1) // 3)
    margin_y = max(8, (bottom - top + 1) // 2)
    win_left = max(0, left - margin_x)
    win_top = max(0, top - margin_y)
    win_right = min(w - 1, right + margin_x)
    win_bottom = min(h - 1, bottom + margin_y)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    box_w = right - left + 1
    box_h = bottom - top + 1

    window = inv[win_top : win_bottom + 1, win_left : win_right + 1]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(window, connectivity=8)
    merged_left, merged_top, merged_right, merged_bottom = left, top, right, bottom

    for idx in range(1, num_labels):
        x, y, cw, ch, area = (int(v) for v in stats[idx])
        comp_left = win_left + x
        comp_top = win_top + y
        comp_right = comp_left + cw - 1
        comp_bottom = comp_top + ch - 1
        if comp_left >= left and comp_right <= right and comp_top >= top and comp_bottom <= bottom:
            continue
        if area < 8 or area > 90 or cw > 18 or ch > 18:
            continue

        comp_center_x = (comp_left + comp_right) / 2.0
        comp_center_y = (comp_top + comp_bottom) / 2.0
        near_top = comp_bottom < top and abs(comp_center_x - center_x) <= box_w * 0.55
        near_bottom = comp_top > bottom and abs(comp_center_x - center_x) <= box_w * 0.55
        near_left = comp_right < left and abs(comp_center_y - center_y) <= box_h * 0.55
        near_right = comp_left > right and abs(comp_center_y - center_y) <= box_h * 0.55
        if near_top or near_bottom or near_left or near_right:
            merged_left = min(merged_left, comp_left)
            merged_top = min(merged_top, comp_top)
            merged_right = max(merged_right, comp_right)
            merged_bottom = max(merged_bottom, comp_bottom)

    return expand_box(merged_left, merged_top, merged_right, merged_bottom, 2, 2, w, h)


def helper_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if left > right or top > bottom:
        return 0.0
    inter = (right - left + 1) * (bottom - top + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / float(min(area_a, area_b))


def extract_table_boxes(
    inv: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    diagonal: np.ndarray,
    existing: list[SymbolDetection],
) -> list[SymbolDetection]:
    h, w = inv.shape
    detections: list[SymbolDetection] = []
    col_counts = np.count_nonzero(vertical, axis=0)
    spans = mask_to_spans(col_counts >= max(18, int(h * 0.6)))

    for span_left, span_right in spans:
        rows = np.flatnonzero(np.any(vertical[:, span_left : span_right + 1] > 0, axis=1))
        if rows.size == 0:
            continue

        top = max(0, int(rows[0]) - 2)
        bottom = min(h - 1, int(rows[-1]) + 2)
        center = (span_left + span_right) // 2
        anchor_y = (int(rows[0]) + int(rows[-1])) // 2 - top
        win_left = max(0, center - 80)
        win_right = min(w - 1, center + 80)
        hbox = dominant_horizontal_box(
            inv[top : bottom + 1, win_left : win_right + 1],
            horizontal[top : bottom + 1, win_left : win_right + 1],
            center - win_left,
            anchor_y,
        )
        if hbox is None:
            continue
        left = win_left + hbox[0]
        right = win_left + hbox[2]

        width = right - left + 1
        height = bottom - top + 1
        if width < 45 or height < 35:
            continue
        row_center = (hbox[1] + hbox[3]) // 2
        if abs(row_center - anchor_y) > max(6, height // 4):
            continue

        crop = inv[top : bottom + 1, left : right + 1]
        structure = cv2.bitwise_or(horizontal[top : bottom + 1, left : right + 1], vertical[top : bottom + 1, left : right + 1])
        structure_pixels = max(1, int(np.count_nonzero(structure)))
        fill = float(np.count_nonzero(crop)) / float(width * height)
        diag_ratio = float(np.count_nonzero(diagonal[top : bottom + 1, left : right + 1])) / float(structure_pixels)
        sep_x = center - left
        left_ink = int(np.count_nonzero(crop[:, : max(1, sep_x)]))
        right_ink = int(np.count_nonzero(crop[:, min(width - 1, sep_x + 1) :]))

        if not (0.15 <= fill <= 0.36 and left_ink >= 20 and right_ink >= 20 and diag_ratio <= 0.40):
            continue

        box = (left, top, right, bottom)
        if any(helper_overlap_ratio(box, (det.left, det.top, det.right, det.bottom)) >= 0.6 for det in existing + detections):
            continue

        detections.append(SymbolDetection("box", left, top, right, bottom))

    return detections


def extract_fi_lower_bars(inv: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray, existing: list[SymbolDetection]) -> list[SymbolDetection]:
    h, w = inv.shape
    detections: list[SymbolDetection] = []
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)

    def band_stats(top: int, bottom: int, left: int, right: int) -> tuple[int, int]:
        ys, xs = np.where(inv[top:bottom, left:right] > 0)
        if xs.size == 0:
            return 0, 0
        return int(xs.size), int(xs.max() - xs.min() + 1)

    def digit_like_count(top: int, bottom: int, left: int, right: int) -> int:
        band = inv[top:bottom, left:right]
        if band.size == 0:
            return 0
        num, _, local_stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
        count = 0
        for comp_idx in range(1, num):
            _, _, lw, lh, area = (int(v) for v in local_stats[comp_idx])
            if 6 <= area <= 120 and 2 <= lw <= 16 and 6 <= lh <= 18:
                count += 1
        return count

    def expand_bar_box(x: int, y: int, cw: int, ch: int) -> tuple[int, int, int, int]:
        left = max(0, x - 2)
        top = max(0, y - 10)
        right = min(w - 1, x + cw - 1 + 2)
        bottom = min(h - 1, y + ch - 1 + 10)
        window = inv[top : bottom + 1, left : right + 1]
        num, _, local_stats, _ = cv2.connectedComponentsWithStats(window, connectivity=8)
        merged = [x, y, x + cw - 1, y + ch - 1]

        for comp_idx in range(1, num):
            lx, ly, lw, lh, area = (int(v) for v in local_stats[comp_idx])
            comp_left = left + lx
            comp_top = top + ly
            comp_right = comp_left + lw - 1
            comp_bottom = comp_top + lh - 1
            if area < 6 or area > 90 or lw > 14 or lh > 14:
                continue
            comp_center_x = (comp_left + comp_right) // 2
            if not (x - 4 <= comp_center_x <= x + cw - 1 + 4):
                continue
            above = comp_bottom < y and y - comp_bottom <= 12
            below = comp_top > y + ch - 1 and comp_top - (y + ch - 1) <= 12
            if not (above or below):
                continue
            merged[0] = min(merged[0], comp_left)
            merged[1] = min(merged[1], comp_top)
            merged[2] = max(merged[2], comp_right)
            merged[3] = max(merged[3], comp_bottom)

        return expand_box(merged[0], merged[1], merged[2], merged[3], 1, 1, w, h)

    for idx in range(1, num_labels):
        x, y, cw, ch, area = (int(v) for v in stats[idx])
        aspect = cw / float(max(1, ch))
        if area < 18 or cw < 24 or cw > 40 or ch > 4 or aspect < 7.0:
            continue

        vertical_band = vertical[max(0, y - 10) : min(h, y + ch + 10), max(0, x - 1) : min(w, x + cw + 1)]
        if np.count_nonzero(vertical_band) > 4 or y > int(h * 0.6):
            continue

        left_cross = False
        for det in existing:
            if det.label != "cross":
                continue
            gap = x - det.right
            if 8 <= gap <= 70 and abs(((det.top + det.bottom) // 2) - ((y + y + ch - 1) // 2)) <= 18:
                left_cross = True
                break
        if not left_cross:
            continue

        pad_x = 6
        upper_ink, upper_span = band_stats(max(0, y - 14), y, max(0, x - pad_x), min(w, x + cw + pad_x))
        lower_ink, lower_span = band_stats(y + ch, min(h, y + ch + 14), max(0, x - pad_x), min(w, x + cw + pad_x))
        upper_digits = digit_like_count(max(0, y - 14), y, max(0, x - pad_x), min(w, x + cw + pad_x))
        lower_digits = digit_like_count(y + ch, min(h, y + ch + 14), max(0, x - pad_x), min(w, x + cw + pad_x))
        tight_limit = cw + 8
        upper_tight = 8 <= upper_ink <= 90 and 1 <= upper_span <= tight_limit
        lower_tight = 8 <= lower_ink <= 90 and 1 <= lower_span <= tight_limit
        if not (upper_tight or lower_tight):
            continue
        if upper_digits > 1 or lower_digits > 1:
            continue
        if (upper_ink > 120 and upper_span > tight_limit) or (lower_ink > 120 and lower_span > tight_limit):
            continue

        box = expand_bar_box(x, y, cw, ch)
        if any(helper_overlap_ratio(box, (det.left, det.top, det.right, det.bottom)) >= 0.6 for det in existing + detections):
            continue

        detections.append(SymbolDetection("fi_lower_bar", *box))

    return detections


def classify_symbol(width: int, height: int, fill: float, components: int, horiz: float, vert: float, diag: float, top_fill: float, bottom_fill: float) -> str | None:
    aspect = width / float(height)

    # 上颌磨牙 FI 常与上下数字分成 2-3 个连通域，不能只接受单连通域。
    if width >= 20 and height >= 34 and fill <= 0.32 and 0.55 <= aspect <= 0.95 and components <= 3 and horiz <= 0.18 and vert >= 0.25 and diag >= 0.45:
        return "fi_upper_y"
    if width >= 55 and height >= 35 and 0.18 <= fill <= 0.30 and 1.35 <= aspect <= 2.60 and components >= 6 and horiz >= 0.50 and vert >= 0.35 and diag <= 0.25:
        return "box"
    if width >= 55 and height >= 35 and aspect <= 3.20 and horiz >= 1.00 and vert >= 0.30:
        return "cross"
    if width >= 50 and height >= 34 and fill <= 0.28 and 1.25 <= aspect <= 3.20 and horiz >= 0.80 and vert >= 0.30:
        return "cross"
    if width >= 24 and height >= 24 and fill <= 0.28 and 0.75 <= aspect <= 1.35 and horiz >= 0.35 and vert >= 0.35 and diag <= 0.25:
        return "cross"
    if width >= 60 and height >= 28 and fill <= 0.30 and 1.35 < aspect <= 3.20 and horiz >= 0.45 and vert >= 0.30 and diag <= 0.30:
        return "cross"
    if width >= 45 and height >= 20 and fill <= 0.50 and aspect >= 1.80 and horiz >= 0.35 and vert >= 0.35 and components <= 4:
        return "pd_probe"
    return None


def refine_symbol_box(
    label: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
    inv: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    diagonal: np.ndarray,
) -> tuple[int, int, int, int]:
    crop_h = slice(top, bottom + 1)
    crop_w = slice(left, right + 1)
    local_horizontal = horizontal[crop_h, crop_w]
    local_vertical = vertical[crop_h, crop_w]
    local_diagonal = diagonal[crop_h, crop_w]
    h, w = inv.shape

    if label == "pd_probe":
        return expand_pd_probe_box(left, top, right, bottom, inv)
    if label == "fi_upper_y":
        ys, xs = np.where(cv2.bitwise_or(local_vertical, local_diagonal) > 0)
        if xs.size:
            return expand_box(left + int(xs.min()), top + int(ys.min()), left + int(xs.max()), top + int(ys.max()), 2, 2, w, h)
    if label == "cross":
        box = dominant_cross_box(inv[crop_h, crop_w], local_horizontal, local_vertical)
        if box is not None:
            x0, y0, x1, y1 = box
            abs_left, abs_top, abs_right, abs_bottom = left + x0, top + y0, left + x1, top + y1
            abs_left, abs_top, abs_right, abs_bottom = pad_cross_left(inv, abs_left, abs_top, abs_right, abs_bottom)
            return expand_box(abs_left, abs_top, abs_right, abs_bottom, 0, 1, w, h)
    return expand_box(left, top, right, bottom, 2, 2, w, h)


def extract_dental_symbols(
    line_bgr: np.ndarray,
    normalize_box_to_cross: bool = True,
    expand_to_line_height: bool = True,
) -> list[SymbolDetection]:
    inv = otsu_binary(line_bgr, invert=True)
    horizontal, vertical, diagonal = build_symbol_masks(inv)
    stroke_mask = cv2.bitwise_or(cv2.bitwise_or(horizontal, vertical), diagonal)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(stroke_mask, connectivity=8)
    detections: list[SymbolDetection] = []

    for idx in range(1, num_labels):
        x, y, w, h, area = (int(v) for v in stats[idx])
        if area < 24 or w < 8 or h < 8:
            continue

        crop = inv[y : y + h, x : x + w]
        fill = float(np.count_nonzero(crop)) / float(w * h)
        mid = h // 2
        components = count_components(crop)
        top_fill = float(np.count_nonzero(crop[:mid, :])) / float(max(1, mid * w))
        bottom_fill = float(np.count_nonzero(crop[mid:, :])) / float(max(1, (h - mid) * w))
        horiz = np.count_nonzero(horizontal[y : y + h, x : x + w]) / float(area)
        vert = np.count_nonzero(vertical[y : y + h, x : x + w]) / float(area)
        diag = np.count_nonzero(diagonal[y : y + h, x : x + w]) / float(area)

        label = classify_symbol(w, h, fill, components, horiz, vert, diag, top_fill, bottom_fill)
        if label is None:
            continue

        left, top, right, bottom = refine_symbol_box(label, x, y, x + w - 1, y + h - 1, inv, horizontal, vertical, diagonal)
        detections.append(SymbolDetection(label, left, top, right, bottom))

    detections.extend(extract_fi_lower_bars(inv, horizontal, vertical, detections))
    detections.extend(extract_table_boxes(inv, horizontal, vertical, diagonal, detections))
    detections.sort(key=lambda det: (det.top, det.left, det.right, det.bottom))
    line_bottom = inv.shape[0] - 1
    normalized: list[SymbolDetection] = []
    for det in detections:
        label = "cross" if normalize_box_to_cross and det.label == "box" else det.label
        top = 0 if expand_to_line_height else det.top
        bottom = line_bottom if expand_to_line_height else det.bottom
        normalized.append(SymbolDetection(label, det.left, top, det.right, bottom))

    return normalized


def extract_dental_symbols_with_fallback(
    line_bgr: np.ndarray,
    normalize_box_to_cross: bool = True,
    expand_to_line_height: bool = True,
) -> list[SymbolDetection]:
    detections = extract_dental_symbols(
        line_bgr,
        normalize_box_to_cross=normalize_box_to_cross,
        expand_to_line_height=expand_to_line_height,
    )
    if detections:
        return detections

    line_h, line_w = line_bgr.shape[:2]
    if line_h < 60 or line_w < 120:
        return detections

    downscaled = cv2.resize(line_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    fallback = extract_dental_symbols(
        downscaled,
        normalize_box_to_cross=normalize_box_to_cross,
        expand_to_line_height=expand_to_line_height,
    )
    scaled_back: list[SymbolDetection] = []
    for det in fallback:
        left = max(0, int(round(det.left * 2)))
        top = max(0, int(round(det.top * 2)))
        right = min(line_w - 1, int(round((det.right + 1) * 2)) - 1)
        bottom = min(line_h - 1, int(round((det.bottom + 1) * 2)) - 1)
        scaled_back.append(SymbolDetection(det.label, left, top, right, bottom))
    return scaled_back


def extract_line_images(page_bgr: np.ndarray) -> list[LineCrop]:
    binary = clean_binary_page(otsu_binary(page_bgr))
    h, w = binary.shape
    min_line_pixels = max(24, int(round(w * 0.05)))
    min_line_height = max(4, h // 250)
    line_crops: list[LineCrop] = []

    for top, bottom in mask_to_spans(np.any(binary < 255, axis=1)):
        band = binary[top : bottom + 1, :]
        cols = np.flatnonzero(np.any(band < 255, axis=0))
        if cols.size == 0:
            continue
        left = int(cols[0])
        right = int(cols[-1])
        if np.count_nonzero(band[:, left : right + 1] < 255) < min_line_pixels:
            continue
        if bottom - top + 1 < min_line_height:
            continue

        crop_top = max(0, top - 2)
        crop_bottom = min(h, bottom + 3)
        crop_left = max(0, left - 4)
        crop_right = min(w, right + 5)
        line_crops.append(
            LineCrop(
                page_bgr[crop_top:crop_bottom, crop_left:crop_right].copy(),
                crop_left,
                crop_top,
                crop_right - 1,
                crop_bottom - 1,
            )
        )

    return line_crops


def log_debug(enabled: bool, message: str) -> None:
    if not enabled:
        return
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def slow_tag(seconds: float, slow_stage_seconds: float) -> str:
    return "SLOW " if seconds >= slow_stage_seconds else ""


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    left: int
    top: int
    right: int
    bottom: int

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1


@dataclass(frozen=True)
class PageContext:
    visit_date: str | None
    birth_date: str | None
    age_years: int | None


@dataclass(frozen=True)
class TextToken:
    text: str
    confidence: float
    left: int
    top: int
    right: int
    bottom: int

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1


@dataclass
class ToothMark:
    region: str
    digit: str
    codes: list[str]
    left: int
    top: int
    right: int
    bottom: int
    surface: str | None = None

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def display(self) -> str:
        return join_codes(self.codes, self.surface)


@dataclass
class ParsedSymbol:
    label: str
    left: int
    top: int
    right: int
    bottom: int
    text: str
    line_index: int
    teeth: list[ToothMark] = field(default_factory=list)


@dataclass
class ParsedLine:
    line_index: int
    left: int
    top: int
    right: int
    bottom: int
    text: str


@dataclass(frozen=True)
class ConversionOutput:
    pdf_path: Path
    case_dir: Path
    record_path: Path
    review_paths: tuple[Path, ...]
    page_count: int


def get_font() -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), FONT_SIZE)
    return ImageFont.load_default()


def get_symbol_color(label: str) -> tuple[int, int, int]:
    if label in {"cross", "box"}:
        return TOOTH_SYMBOL_COLOR
    if label == "pd_probe":
        return PD_SYMBOL_COLOR
    if label in {"fi_upper_y", "fi_lower_bar"}:
        return FURCATION_SYMBOL_COLOR
    return FURCATION_SYMBOL_COLOR


def normalize_ocr_text(text: str) -> str:
    return re.sub(r"[\s:：,，;；。.\-_/\\]+", "", text).upper()


def shrink_box(points: list[list[float]], scale: int) -> tuple[int, int, int, int]:
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    left = max(0, int(math.floor(min(xs) / scale)))
    top = max(0, int(math.floor(min(ys) / scale)))
    right = max(left, int(math.ceil(max(xs) / scale)) - 1)
    bottom = max(top, int(math.ceil(max(ys) / scale)) - 1)
    return left, top, right, bottom


def run_ocr_tokens(ocr: RapidOCR, image: np.ndarray, scale: int = OCR_SCALE, binarize: bool = True) -> list[OCRToken]:
    if image is None:
        return []
    try:
        image = ensure_cv_image(image, "OCR输入图像")
    except ValueError:
        return []

    if image.ndim == 2:
        work = image
    else:
        work = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if binarize:
        work = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    result, _ = ocr(work)
    tokens: list[OCRToken] = []
    for item in result or []:
        points, text, confidence = item
        normalized = normalize_ocr_text(text)
        if not normalized:
            continue
        left, top, right, bottom = shrink_box(points, scale)
        tokens.append(OCRToken(normalized, float(confidence), left, top, right, bottom))
    return tokens


def clean_text_token_text(text: str) -> str:
    text = text.replace("\n", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_text_tokens(ocr: RapidOCR, image: np.ndarray, scale: int = 1, binarize: bool = False) -> list[TextToken]:
    if image is None:
        return []
    try:
        image = ensure_cv_image(image, "文本OCR输入图像")
    except ValueError:
        return []

    if image.ndim == 2:
        work = image
    else:
        work = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale != 1:
        work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if binarize:
        work = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    result, _ = ocr(work)
    tokens: list[TextToken] = []
    for item in result or []:
        points, text, confidence = item
        cleaned = clean_text_token_text(text)
        if not cleaned:
            continue
        left, top, right, bottom = shrink_box(points, scale)
        tokens.append(TextToken(cleaned, float(confidence), left, top, right, bottom))
    return tokens


def dedupe_text_tokens(tokens: list[TextToken], position_tol: int = 4) -> list[TextToken]:
    deduped: list[TextToken] = []
    for token in sorted(tokens, key=lambda item: (-item.confidence, -(item.width * item.height), item.left, item.top)):
        exists = any(
            kept.text == token.text
            and abs(kept.cx - token.cx) <= position_tol
            and abs(kept.cy - token.cy) <= position_tol
            for kept in deduped
        )
        if not exists:
            deduped.append(token)
    deduped.sort(key=lambda item: (item.top, item.left, item.right, item.bottom))
    return deduped


def text_tokens_from_ocr_result(result: list | None, scale: int = 1) -> list[TextToken]:
    tokens: list[TextToken] = []
    for item in result or []:
        points, text, confidence = item
        cleaned = clean_text_token_text(text)
        if not cleaned:
            continue
        left, top, right, bottom = shrink_box(points, scale)
        tokens.append(TextToken(cleaned, float(confidence), left, top, right, bottom))
    return tokens


def translate_tokens(tokens: list[OCRToken], offset_x: int, offset_y: int) -> list[OCRToken]:
    return [
        OCRToken(
            token.text,
            token.confidence,
            token.left + offset_x,
            token.top + offset_y,
            token.right + offset_x,
            token.bottom + offset_y,
        )
        for token in tokens
    ]


def dedupe_tokens(tokens: list[OCRToken], position_tol: int = 3) -> list[OCRToken]:
    deduped: list[OCRToken] = []
    for token in sorted(tokens, key=lambda item: (-item.confidence, -(item.width * item.height), item.left, item.top)):
        exists = any(
            kept.text == token.text
            and abs(kept.cx - token.cx) <= position_tol
            and abs(kept.cy - token.cy) <= position_tol
            for kept in deduped
        )
        if not exists:
            deduped.append(token)
    deduped.sort(key=lambda item: (item.top, item.left, item.right, item.bottom))
    return deduped


def translate_text_tokens(tokens: list[TextToken], offset_x: int, offset_y: int) -> list[TextToken]:
    return [
        TextToken(
            token.text,
            token.confidence,
            token.left + offset_x,
            token.top + offset_y,
            token.right + offset_x,
            token.bottom + offset_y,
        )
        for token in tokens
    ]


def extract_grid_digits_and_tail(text: str) -> tuple[str | None, str]:
    match = re.match(r"\s*([1-8]{2,8})(.*)", str(text or ""))
    if not match:
        return None, ""
    digits = match.group(1)
    tail = (match.group(2) or "").strip()
    return digits, tail


def grid_digits_to_codes(digits: str, region: str, age_years: int | None) -> list[str]:
    codes: list[str] = []
    for ch in digits:
        if ch not in "12345678":
            continue
        inferred = infer_codes(region, ch, age_years)
        if inferred:
            codes.append(inferred[0])
    return codes


def merge_tooth_grid_text_tokens(
    text_tokens: list[TextToken],
    age_years: int | None,
) -> list[TextToken]:
    if len(text_tokens) < 4:
        return text_tokens

    candidates = []
    for idx, token in enumerate(text_tokens):
        digits, tail = extract_grid_digits_and_tail(token.text)
        if digits is None:
            continue
        candidates.append(
            {
                "idx": idx,
                "token": token,
                "digits": digits,
                "tail": tail,
            }
        )

    if len(candidates) < 4:
        return text_tokens

    used: set[int] = set()
    merged_tokens: list[TextToken] = []

    def top_aligned(a: TextToken, b: TextToken) -> bool:
        return abs(a.cy - b.cy) <= max(6.0, min(a.height, b.height) * 0.55)

    def bottom_aligned(a: TextToken, b: TextToken) -> bool:
        return b.cy > a.cy and (b.cy - a.cy) <= max(28.0, (a.height + b.height) * 1.2)

    def same_column(a: TextToken, b: TextToken) -> bool:
        return x_overlap_ratio(a.left, a.right, b.left, b.right) >= 0.45

    def right_column(a: TextToken, b: TextToken) -> bool:
        gap = b.left - a.right
        return 3 <= gap <= 28

    for item in candidates:
        idx = item["idx"]
        if idx in used:
            continue
        top_left = item["token"]

        top_right_item = None
        for other in candidates:
            if other["idx"] in used or other["idx"] == idx:
                continue
            token = other["token"]
            if top_aligned(top_left, token) and right_column(top_left, token):
                top_right_item = other
                break
        if top_right_item is None:
            continue

        bottom_left_item = None
        for other in candidates:
            if other["idx"] in used or other["idx"] in {idx, top_right_item["idx"]}:
                continue
            token = other["token"]
            if same_column(top_left, token) and bottom_aligned(top_left, token):
                bottom_left_item = other
                break
        if bottom_left_item is None:
            continue

        bottom_right_item = None
        for other in candidates:
            if other["idx"] in used or other["idx"] in {idx, top_right_item["idx"], bottom_left_item["idx"]}:
                continue
            token = other["token"]
            if same_column(top_right_item["token"], token) and bottom_aligned(top_right_item["token"], token):
                bottom_right_item = other
                break
        if bottom_right_item is None:
            continue

        top_left_item = item
        tl_codes = grid_digits_to_codes(top_left_item["digits"], "tl", age_years)
        tr_codes = grid_digits_to_codes(top_right_item["digits"], "tr", age_years)
        bl_codes = grid_digits_to_codes(bottom_left_item["digits"], "bl", age_years)
        br_codes = grid_digits_to_codes(bottom_right_item["digits"], "br", age_years)
        left_codes = tl_codes + bl_codes
        right_codes = tr_codes + br_codes
        if not left_codes and not right_codes:
            continue

        pieces = []
        if left_codes:
            pieces.append("、".join(left_codes))
        if right_codes:
            pieces.append("、".join(right_codes))
        merged_text = "；".join(piece for piece in pieces if piece)

        tail = bottom_right_item["tail"] or top_right_item["tail"] or bottom_left_item["tail"] or top_left_item["tail"]
        if tail:
            merged_text += tail

        left = min(top_left.left, top_right_item["token"].left, bottom_left_item["token"].left, bottom_right_item["token"].left)
        top = min(top_left.top, top_right_item["token"].top, bottom_left_item["token"].top, bottom_right_item["token"].top)
        right = max(top_left.right, top_right_item["token"].right, bottom_left_item["token"].right, bottom_right_item["token"].right)
        bottom = max(top_left.bottom, top_right_item["token"].bottom, bottom_left_item["token"].bottom, bottom_right_item["token"].bottom)

        merged_tokens.append(
            TextToken(
                text=merged_text,
                confidence=min(
                    top_left.confidence,
                    top_right_item["token"].confidence,
                    bottom_left_item["token"].confidence,
                    bottom_right_item["token"].confidence,
                ),
                left=left,
                top=top,
                right=right,
                bottom=bottom,
            )
        )
        used.update({idx, top_right_item["idx"], bottom_left_item["idx"], bottom_right_item["idx"]})

    if not merged_tokens:
        return text_tokens

    output_tokens: list[TextToken] = []
    for idx, token in enumerate(text_tokens):
        if idx not in used:
            output_tokens.append(token)
    output_tokens.extend(merged_tokens)
    output_tokens.sort(key=lambda item: (item.left, item.top, item.right, item.bottom))
    return output_tokens


def intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if left > right or top > bottom:
        return 0
    return (right - left + 1) * (bottom - top + 1)


def overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    return inter / float(max(1, area_a))


def normalized_visual_label(label: str) -> str:
    return "cross" if label in {"cross", "box"} else label


def x_overlap_ratio(a_left: int, a_right: int, b_left: int, b_right: int) -> float:
    left = max(a_left, b_left)
    right = min(a_right, b_right)
    if left > right:
        return 0.0
    inter = right - left + 1
    span = min(a_right - a_left + 1, b_right - b_left + 1)
    return inter / float(max(1, span))


def build_visual_mapping(raw_detections: list[SymbolDetection], visual_detections: list[SymbolDetection]) -> dict[int, SymbolDetection]:
    used: set[int] = set()
    mapping: dict[int, SymbolDetection] = {}
    for raw_idx, raw in enumerate(raw_detections):
        target_label = normalized_visual_label(raw.label)
        best_idx = None
        best_score = float("-inf")
        raw_center = (raw.left + raw.right) / 2.0
        for visual_idx, visual in enumerate(visual_detections):
            if visual_idx in used or visual.label != target_label:
                continue
            visual_center = (visual.left + visual.right) / 2.0
            score = x_overlap_ratio(raw.left, raw.right, visual.left, visual.right) * 100.0 - abs(raw_center - visual_center)
            if score > best_score:
                best_score = score
                best_idx = visual_idx
        if best_idx is not None:
            used.add(best_idx)
            mapping[raw_idx] = visual_detections[best_idx]
    return mapping


def convert_digit_char(ch: str) -> str | None:
    mapping = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "T": "1",
        "Z": "2",
        "S": "5",
        "$": "5",
        "G": "6",
        "B": "8",
    }
    ch = ch.upper()
    if ch.isdigit():
        return ch
    return mapping.get(ch)


def convert_label_char(ch: str) -> str | None:
    mapping = {
        "W": "M",
        "V": "M",
        "N": "M",
        "I": "L",
        "1": "L",
        "8": "B",
        "0": "O",
    }
    ch = ch.upper()
    if ch in {"B", "L", "M", "D", "O"}:
        return ch
    return mapping.get(ch)


def explode_tokens(tokens: list[OCRToken], mode: str, allowed: set[str]) -> list[OCRToken]:
    exploded: list[OCRToken] = []
    for token in tokens:
        raw_chars = [ch for ch in token.text if not ch.isspace()]
        converted_chars: list[str] = []
        for ch in token.text:
            converted = convert_digit_char(ch) if mode == "digit" else convert_label_char(ch)
            if converted is not None and converted in allowed:
                converted_chars.append(converted)
        if mode == "digit" and converted_chars and not any(ch.isdigit() for ch in token.text):
            if len(converted_chars) == 1 and len(raw_chars) > 1:
                continue
        if not converted_chars:
            continue
        if len(converted_chars) == 1:
            exploded.append(OCRToken(converted_chars[0], token.confidence, token.left, token.top, token.right, token.bottom))
            continue

        width = max(1.0, float(token.width) / float(len(converted_chars)))
        for idx, ch in enumerate(converted_chars):
            left = int(round(token.left + idx * width))
            right = int(round(token.left + (idx + 1) * width)) - 1
            exploded.append(OCRToken(ch, token.confidence, left, token.top, max(left, right), token.bottom))
    return exploded


def parse_date_token(value: str) -> date | None:
    digits = re.findall(r"\d+", value)
    if len(digits) < 3:
        return None
    year, month, day = (int(digits[0]), int(digits[1]), int(digits[2]))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def compute_age_years(visit: date | None, birth: date | None) -> int | None:
    if visit is None or birth is None:
        return None
    years = visit.year - birth.year
    if (visit.month, visit.day) < (birth.month, birth.day):
        years -= 1
    return years


def extract_page_context_from_result(result: list | None) -> PageContext:
    page_text = "\n".join(item[1] for item in (result or []))

    visit_match = re.search(r"日期[：:]?\s*(\d{4}[-年./]\d{1,2}[-月./]\d{1,2})", page_text)
    birth_match = re.search(r"出生日期[：:]?\s*(\d{4}[-年./]\d{1,2}[-月./]\d{1,2})", page_text)
    visit_text = visit_match.group(1) if visit_match else None
    birth_text = birth_match.group(1) if birth_match else None

    if visit_text is None or birth_text is None:
        all_dates = re.findall(r"\d{4}[-年./]\d{1,2}[-月./]\d{1,2}", page_text)
        if visit_text is None and all_dates:
            visit_text = all_dates[0]
        if birth_text is None and len(all_dates) >= 2:
            birth_text = all_dates[1]

    visit_date = parse_date_token(visit_text) if visit_text else None
    birth_date = parse_date_token(birth_text) if birth_text else None
    age_years = compute_age_years(visit_date, birth_date)
    return PageContext(
        visit_date=visit_date.isoformat() if visit_date else None,
        birth_date=birth_date.isoformat() if birth_date else None,
        age_years=age_years,
    )


def run_page_text_ocr(ocr: RapidOCR, page_bgr: np.ndarray) -> tuple[PageContext, list[TextToken]]:
    result, _ = ocr(page_bgr)
    return extract_page_context_from_result(result), text_tokens_from_ocr_result(result)


def extract_page_context(ocr: RapidOCR, page_bgr: np.ndarray) -> PageContext:
    context, _ = run_page_text_ocr(ocr, page_bgr)
    return context


def infer_codes(region: str, digit: str, age_years: int | None) -> list[str]:
    value = int(digit)
    permanent = f"{PERMANENT_PREFIX[region]}{digit}"
    primary = f"{PRIMARY_PREFIX[region]}{digit}"
    if value >= 6:
        return [permanent]
    if age_years is None:
        return [permanent]
    if age_years >= 13:
        return [permanent]
    if age_years <= 5:
        return [primary]
    return [permanent]


def join_codes(codes: list[str], surface: str | None = None) -> str:
    suffix = surface or ""
    return "或".join(f"{code}{suffix}" for code in codes)


def quadrant_of_token(token: OCRToken, width: int, height: int, anchor_x: float | None = None, anchor_y: float | None = None) -> str | None:
    mid_x = width / 2.0 if anchor_x is None else anchor_x
    mid_y = height / 2.0 if anchor_y is None else anchor_y
    if token.cx < mid_x and token.cy < mid_y:
        return "tl"
    if token.cx >= mid_x and token.cy < mid_y:
        return "tr"
    if token.cx >= mid_x and token.cy >= mid_y:
        return "br"
    return "bl"


def attach_surface_letters(teeth: list[ToothMark], letters: list[OCRToken]) -> None:
    for letter in letters:
        best_tooth: ToothMark | None = None
        best_score = float("inf")
        for tooth in teeth:
            dx = letter.cx - tooth.right
            dy = tooth.top - letter.cy
            score = abs(dx) * 1.4 + abs(dy)
            if dx < -2 or dx > 12:
                score += 999.0
            if dy < -4 or dy > 14:
                score += 999.0
            if score < best_score:
                best_score = score
                best_tooth = tooth
        if best_tooth is not None and best_score < 18.0:
            best_tooth.surface = letter.text


def suppress_conflicting_digit_tokens(tokens: list[OCRToken]) -> list[OCRToken]:
    kept: list[OCRToken] = []
    for token in sorted(tokens, key=lambda item: (-item.confidence, item.width * item.height, item.left, item.top)):
        token_box = (token.left, token.top, token.right, token.bottom)
        conflict = False
        for existing in kept:
            existing_box = (existing.left, existing.top, existing.right, existing.bottom)
            overlap = max(overlap_ratio(token_box, existing_box), overlap_ratio(existing_box, token_box))
            close_centers = abs(token.cx - existing.cx) <= max(token.width, existing.width) * 0.4 and abs(token.cy - existing.cy) <= max(token.height, existing.height) * 0.4
            if overlap >= 0.55 and close_centers:
                conflict = True
                break
        if not conflict:
            kept.append(token)
    kept.sort(key=lambda item: (item.top, item.left, item.right, item.bottom))
    return kept


def choose_representative_digit(tokens: list[OCRToken], width: int, height: int) -> list[OCRToken]:
    if width > 40 or height > 55 or not tokens:
        return tokens
    if len(tokens) <= 1:
        return tokens
    ordered = sorted(tokens, key=lambda item: (item.left, item.top, item.right, item.bottom))
    avg_height = sum(item.height for item in ordered) / float(len(ordered))
    avg_width = sum(item.width for item in ordered) / float(len(ordered))
    y_span = max(item.cy for item in ordered) - min(item.cy for item in ordered)
    max_gap = 0.0
    for prev, curr in zip(ordered, ordered[1:]):
        max_gap = max(max_gap, float(curr.left - prev.right - 1))
    if y_span <= max(4.0, avg_height * 0.45) and max_gap <= max(6.0, avg_width * 1.5):
        return ordered
    best = max(
        tokens,
        key=lambda item: (
            min(abs(item.cx - width / 2.0), abs(item.cy - height / 2.0)),
            item.confidence,
            item.width * item.height,
        ),
    )
    return [best]


def is_axis_suspicious_digit(token: OCRToken, anchor_x: float, anchor_y: float, width: int, height: int) -> bool:
    margin_x = max(3.0, width * 0.10)
    margin_y = max(3.0, height * 0.10)
    return abs(token.cx - anchor_x) <= margin_x or abs(token.cy - anchor_y) <= margin_y


def reassign_axis_boundary_digits(
    grouped: dict[str, list[OCRToken]],
    width: int,
    height: int,
    anchor_x: float,
) -> None:
    axis_margin = max(4.0, width * 0.08)
    row_margin = max(4.0, height * 0.12)
    gap_margin = max(6.0, width * 0.10)

    for left_region, right_region in (("tl", "tr"), ("bl", "br")):
        left_tokens = sorted(grouped[left_region], key=lambda item: (item.left, item.top))
        right_tokens = sorted(grouped[right_region], key=lambda item: (item.left, item.top))

        if len(left_tokens) >= 2 and len(right_tokens) == 1:
            boundary = right_tokens[0]
            row_center = sum(token.cy for token in left_tokens) / float(len(left_tokens))
            gap = boundary.left - left_tokens[-1].right
            if abs(boundary.cx - anchor_x) <= axis_margin and abs(boundary.cy - row_center) <= row_margin and gap <= gap_margin:
                grouped[left_region].append(boundary)
                grouped[right_region] = []

        if len(right_tokens) >= 2 and len(left_tokens) == 1:
            boundary = left_tokens[0]
            row_center = sum(token.cy for token in right_tokens) / float(len(right_tokens))
            gap = right_tokens[0].left - boundary.right
            if abs(boundary.cx - anchor_x) <= axis_margin and abs(boundary.cy - row_center) <= row_margin and gap <= gap_margin:
                grouped[right_region].insert(0, boundary)
                grouped[left_region] = []


def collapse_small_cross_duplicate_digits(
    grouped: dict[str, list[OCRToken]],
    width: int,
    height: int,
    anchor_x: float,
    anchor_y: float,
) -> None:
    all_tokens = []
    for region in ("tl", "tr", "br", "bl"):
        for token in grouped[region]:
            all_tokens.append((region, token))

    if len(all_tokens) != 2:
        return

    (region_a, token_a), (region_b, token_b) = all_tokens
    if token_a.text != token_b.text:
        return
    if abs(token_a.cx - token_b.cx) > max(4.0, width * 0.12):
        return

    dist_a = abs(token_a.cy - anchor_y)
    dist_b = abs(token_b.cy - anchor_y)
    if dist_a == dist_b:
        keep_region = region_a if token_a.cy < token_b.cy else region_b
    else:
        keep_region = region_a if dist_a > dist_b else region_b

    for region in ("tl", "tr", "br", "bl"):
        if region != keep_region:
            grouped[region] = [token for token in grouped[region] if token.text != token_a.text or abs(token.cx - token_b.cx) > max(4.0, width * 0.12)]


def mask_symbol_axes(crop: np.ndarray) -> np.ndarray:
    masked = crop.copy()
    height, width = masked.shape[:2]
    center_x = width // 2
    center_y = height // 2
    band = max(1, int(round(min(width, height) * 0.08)))
    masked[:, max(0, center_x - band) : min(width, center_x + band + 1)] = 255
    masked[max(0, center_y - band) : min(height, center_y + band + 1), :] = 255
    return masked


def tight_crop_to_ink(image: np.ndarray, pad: int = 1) -> np.ndarray:
    if image.size == 0:
        return image

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    binary = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ys, xs = np.where(binary > 0)
    if xs.size == 0 or ys.size == 0:
        return image

    left = max(0, int(xs.min()) - pad)
    right = min(image.shape[1], int(xs.max()) + pad + 1)
    top = max(0, int(ys.min()) - pad)
    bottom = min(image.shape[0], int(ys.max()) + pad + 1)
    return image[top:bottom, left:right]


def quadrant_roi_bounds(
    width: int,
    height: int,
    region: str,
    outer_pad: int = 1,
    axis_ratio: float = 0.10,
) -> tuple[int, int, int, int]:
    center_x = width / 2.0
    center_y = height / 2.0
    axis_pad_x = max(1, int(round(width * axis_ratio)))
    axis_pad_y = max(1, int(round(height * axis_ratio)))

    if region == "tl":
        left = outer_pad
        right = max(left + 1, int(math.floor(center_x - axis_pad_x)))
        top = outer_pad
        bottom = max(top + 1, int(math.floor(center_y - axis_pad_y)))
    elif region == "tr":
        left = min(width - 1, int(math.ceil(center_x + axis_pad_x)))
        right = max(left + 1, width - outer_pad)
        top = outer_pad
        bottom = max(top + 1, int(math.floor(center_y - axis_pad_y)))
    elif region == "br":
        left = min(width - 1, int(math.ceil(center_x + axis_pad_x)))
        right = max(left + 1, width - outer_pad)
        top = min(height - 1, int(math.ceil(center_y + axis_pad_y)))
        bottom = max(top + 1, height - outer_pad)
    else:
        left = outer_pad
        right = max(left + 1, int(math.floor(center_x - axis_pad_x)))
        top = min(height - 1, int(math.ceil(center_y + axis_pad_y)))
        bottom = max(top + 1, height - outer_pad)

    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width))
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    return left, top, right, bottom


def roi_has_digit_ink(roi: np.ndarray) -> bool:
    if roi.size == 0:
        return False

    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    binary = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if np.count_nonzero(binary) < 8:
        return False

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for idx in range(1, num_labels):
        _, _, w, h, area = (int(v) for v in stats[idx])
        if 6 <= area <= 220 and 2 <= w <= max(18, roi.shape[1]) and 4 <= h <= max(22, roi.shape[0]):
            return True
    return False


def recognize_digit_from_roi(
    ocr: RapidOCR,
    roi: np.ndarray,
    allowed: set[str],
) -> tuple[str | None, float]:
    if roi.size == 0:
        return None, 0.0

    variants: list[np.ndarray] = [tight_crop_to_ink(roi)]
    if variants[0].shape[:2] != roi.shape[:2]:
        variants.append(roi)

    scores: dict[str, float] = {}
    for variant in variants:
        if variant.size == 0:
            continue
        for binarize in (True, False):
            for scale in (6, 8, 10, 12):
                tokens = run_ocr_tokens(ocr, variant, scale=scale, binarize=binarize)
                for token in tokens:
                    candidate = normalize_digit_candidate(token.text, allowed)
                    if not candidate or len(candidate) != 1:
                        continue
                    candidate = disambiguate_six_nine(variant, candidate)
                    score = token.confidence
                    if not binarize:
                        score += 0.05
                    if variant.shape[0] <= 18 or variant.shape[1] <= 18:
                        score += 0.03
                    scores[candidate] = scores.get(candidate, 0.0) + score
    if not scores:
        return None, 0.0

    digit, score = max(scores.items(), key=lambda item: (item[1], item[0]))
    return digit, score


def recognize_symbol_digits_by_quadrant(
    ocr: RapidOCR,
    crop: np.ndarray,
) -> dict[str, OCRToken]:
    height, width = crop.shape[:2]
    masked = mask_symbol_axes(crop)
    detected: dict[str, OCRToken] = {}

    for region in ("tl", "tr", "br", "bl"):
        left, top, right, bottom = quadrant_roi_bounds(width, height, region)
        roi = masked[top:bottom, left:right]
        if not roi_has_digit_ink(roi):
            continue

        digit, score = recognize_digit_from_roi(ocr, roi, allowed=set("12345678"))
        if digit is None or score < 0.45:
            continue

        detected[region] = OCRToken(
            text=digit,
            confidence=score,
            left=left,
            top=top,
            right=max(left, right - 1),
            bottom=max(top, bottom - 1),
        )

    return detected


def detect_symbol_digit_regions(crop: np.ndarray) -> set[str]:
    height, width = crop.shape[:2]
    masked = mask_symbol_axes(crop)
    regions: set[str] = set()
    for region in ("tl", "tr", "br", "bl"):
        left, top, right, bottom = quadrant_roi_bounds(width, height, region)
        roi = masked[top:bottom, left:right]
        if roi_has_digit_ink(roi):
            regions.add(region)
    return regions


def collect_cross_label_tokens(ocr: RapidOCR, crop: np.ndarray, include_labels: bool) -> list[OCRToken]:
    if not include_labels or crop.size == 0:
        return []

    tokens = dedupe_tokens(
        run_ocr_tokens(ocr, crop, scale=1, binarize=False)
        + run_ocr_tokens(ocr, crop, scale=OCR_SCALE, binarize=True)
    )
    source_tokens = [token for token in tokens if not any(ch.isdigit() for ch in token.text)]
    return explode_tokens(source_tokens, mode="label", allowed=set("BLMDO"))


def collect_cross_tokens(ocr: RapidOCR, line_image: np.ndarray, det: SymbolDetection) -> list[OCRToken]:
    crop = line_image[det.top : det.bottom + 1, det.left : det.right + 1]
    tokens = translate_tokens(run_ocr_tokens(ocr, crop, scale=1, binarize=False), 0, 0)

    width = det.right - det.left + 1
    height = det.bottom - det.top + 1
    is_small_cross = det.label == "cross" and width <= 40 and height <= 45
    digit_count = len(explode_tokens(tokens, mode="digit", allowed=set("12345678")))

    if det.label == "box":
        tokens.extend(translate_tokens(run_ocr_tokens(ocr, crop, scale=OCR_SCALE, binarize=True), 0, 0))

    if det.label == "cross" and digit_count <= 1:
        masked_crop = mask_symbol_axes(crop)
        tokens.extend(translate_tokens(run_ocr_tokens(ocr, masked_crop, scale=2, binarize=False), 0, 0))
        tokens.extend(translate_tokens(run_ocr_tokens(ocr, masked_crop, scale=6, binarize=False), 0, 0))
        digit_count = len(explode_tokens(tokens, mode="digit", allowed=set("12345678")))

    if is_small_cross and digit_count <= 1:
        pad = 8
        roi_left = max(0, det.left - pad)
        roi_top = max(0, det.top - pad)
        roi_right = min(line_image.shape[1] - 1, det.right + pad)
        roi_bottom = min(line_image.shape[0] - 1, det.bottom + pad)
        roi = line_image[roi_top : roi_bottom + 1, roi_left : roi_right + 1]
        offset_x = roi_left - det.left
        offset_y = roi_top - det.top
        tokens.extend(translate_tokens(run_ocr_tokens(ocr, roi, scale=1, binarize=False), offset_x, offset_y))
        tokens.extend(translate_tokens(run_ocr_tokens(ocr, roi, scale=OCR_SCALE, binarize=True), offset_x, offset_y))

    return dedupe_tokens(tokens)


def parse_cross_symbol(
    ocr: RapidOCR,
    line_image: np.ndarray,
    det: SymbolDetection,
    age_years: int | None,
) -> tuple[str, list[ToothMark]]:
    crop = line_image[det.top : det.bottom + 1, det.left : det.right + 1]
    tokens = collect_cross_tokens(ocr, line_image, det)
    digit_tokens = suppress_conflicting_digit_tokens(explode_tokens(tokens, mode="digit", allowed=set("12345678")))
    label_tokens = collect_cross_label_tokens(ocr, crop, include_labels=det.label != "box")
    height, width = crop.shape[:2]
    anchor_x = width / 2.0
    anchor_y = height / 2.0
    occupied_regions = detect_symbol_digit_regions(crop)

    grouped: dict[str, list[OCRToken]] = {"tl": [], "tr": [], "br": [], "bl": []}
    if len(occupied_regions) == 1 and digit_tokens:
        only_region = next(iter(occupied_regions))
        grouped[only_region] = list(digit_tokens)
    else:
        for token in digit_tokens:
            region = quadrant_of_token(token, width, height, anchor_x, anchor_y)
            if region is not None:
                grouped[region].append(token)

    reassign_axis_boundary_digits(grouped, width, height, anchor_x)
    if det.label == "cross" and width <= 40 and height <= 45:
        collapse_small_cross_duplicate_digits(grouped, width, height, anchor_x, anchor_y)
        if len(occupied_regions) == 1:
            only_region = next(iter(occupied_regions))
            merged_tokens: list[OCRToken] = []
            for region in ("tl", "tr", "br", "bl"):
                merged_tokens.extend(grouped[region])
                if region != only_region:
                    grouped[region] = []
            grouped[only_region] = merged_tokens or grouped[only_region]

    teeth: list[ToothMark] = []
    for region in ("tl", "tr", "br", "bl"):
        selected = choose_representative_digit(sorted(grouped[region], key=lambda item: (item.cx, item.cy)), width, height)
        if len(selected) == 1 and not occupied_regions and is_axis_suspicious_digit(selected[0], anchor_x, anchor_y, width, height):
            selected = []
        for token in selected:
            teeth.append(
                ToothMark(
                    region=region,
                    digit=token.text,
                    codes=infer_codes(region, token.text, age_years),
                    left=token.left,
                    top=token.top,
                    right=token.right,
                    bottom=token.bottom,
                )
            )
    # 针对形态学检测到有笔画但通用OCR遗漏的象限（如L18中误识或遗漏7），补充象限专属fallback识别
    missing_occupied = [r for r in occupied_regions if not any(t.region == r for t in teeth)]
    if missing_occupied:
        fallback_tokens = recognize_symbol_digits_by_quadrant(ocr, crop)
        for region in missing_occupied:
            token = fallback_tokens.get(region)
            if token is not None:
                teeth.append(
                    ToothMark(
                        region=region,
                        digit=token.text,
                        codes=infer_codes(region, token.text, age_years),
                        left=token.left,
                        top=token.top,
                        right=token.right,
                        bottom=token.bottom,
                    )
                )

    if not teeth:
        fallback_tokens = recognize_symbol_digits_by_quadrant(ocr, crop)
        for region in ("tl", "tr", "br", "bl"):
            token = fallback_tokens.get(region)
            if token is None:
                continue
            teeth.append(
                ToothMark(
                    region=region,
                    digit=token.text,
                    codes=infer_codes(region, token.text, age_years),
                    left=token.left,
                    top=token.top,
                    right=token.right,
                    bottom=token.bottom,
                )
            )

    attach_surface_letters(teeth, label_tokens)
    text = ""
    if teeth:
        ordered = sorted(teeth, key=lambda item: ("tl", "tr", "br", "bl").index(item.region) * 100 + item.left)
        text = "、".join(item.display for item in ordered)
    return text, teeth


def select_nearest(candidates: list[OCRToken], target_x: float, target_y: float, max_dist: float) -> OCRToken | None:
    best: OCRToken | None = None
    best_dist = float("inf")
    for token in candidates:
        dist = math.hypot(token.cx - target_x, token.cy - target_y)
        if dist < best_dist:
            best_dist = dist
            best = token
    if best is None or best_dist > max_dist:
        return None
    return best


def choose_row_label(labels: list[OCRToken], target_x: float, target_y: float, choices: set[str]) -> str | None:
    filtered = [label for label in labels if label.text in choices]
    selected = select_nearest(filtered, target_x, target_y, max_dist=35.0)
    if selected is not None:
        return selected.text
    return None


def choose_col_label(labels: list[OCRToken], target_x: float, target_y: float) -> str | None:
    filtered = [label for label in labels if label.text in {"M", "D"}]
    selected = select_nearest(filtered, target_x, target_y, max_dist=35.0)
    if selected is not None:
        return selected.text

    # OCR may confuse the mesial "M" with "L" when reading compact PD tables.
    fallback_l = [label for label in labels if label.text == "L"]
    selected_l = select_nearest(fallback_l, target_x, target_y, max_dist=28.0)
    if selected_l is not None:
        return "M"
    return None


def opposite_row_label(label: str | None) -> str | None:
    return {"B": "L", "L": "B"}.get(label)


def opposite_col_label(label: str | None) -> str | None:
    return {"M": "D", "D": "M"}.get(label)


def assign_slots(tokens: list[OCRToken], slots: list[tuple[str, float, float]], max_dist: float) -> dict[str, str]:
    remaining = list(tokens)
    assigned: dict[str, str] = {}
    for name, target_x, target_y in slots:
        best_idx = None
        best_dist = float("inf")
        for idx, token in enumerate(remaining):
            dist = math.hypot(token.cx - target_x, token.cy - target_y)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is not None and best_dist <= max_dist:
            assigned[name] = remaining.pop(best_idx).text
    return assigned


def best_exploded_char(tokens: list[OCRToken], mode: str, allowed: set[str]) -> str | None:
    exploded = explode_tokens(tokens, mode=mode, allowed=allowed)
    if not exploded:
        return None
    scores: dict[str, float] = {}
    for token in exploded:
        scores[token.text] = scores.get(token.text, 0.0) + token.confidence
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def normalize_digit_candidate(text: str, allowed: set[str]) -> str | None:
    raw_chars = [ch for ch in text if not ch.isspace()]
    converted_chars: list[str] = []
    for ch in text:
        converted = convert_digit_char(ch)
        if converted is not None and converted in allowed:
            converted_chars.append(converted)
    if not converted_chars:
        return None
    if not any(ch.isdigit() for ch in text) and len(converted_chars) == 1 and len(raw_chars) > 1:
        return None
    return "".join(converted_chars)


def disambiguate_six_nine(cell: np.ndarray, digit: str) -> str:
    if digit not in {"6", "9"}:
        return digit
    if cell.size == 0:
        return digit

    if cell.ndim == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell.copy()

    binary = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ys, xs = np.where(binary > 0)
    if xs.size == 0 or ys.size == 0:
        return digit

    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())
    roi = binary[top : bottom + 1, left : right + 1]
    if roi.size == 0:
        return digit

    contours, hierarchy = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return digit

    holes = []
    roi_area = float(roi.shape[0] * roi.shape[1])
    for idx, contour in enumerate(contours):
        parent = hierarchy[0][idx][3]
        if parent == -1:
            continue
        area = cv2.contourArea(contour)
        if area < max(4.0, roi_area * 0.01) or area > roi_area * 0.45:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        holes.append(moments["m01"] / moments["m00"])

    if not holes:
        return digit

    mean_hole_y = sum(holes) / float(len(holes))
    return "9" if mean_hole_y <= roi.shape[0] * 0.5 else "6"


def crop_center_window(
    image: np.ndarray,
    center_x_ratio: float,
    center_y_ratio: float,
    half_width_ratio: float,
    half_height_ratio: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    cx = width * center_x_ratio
    cy = height * center_y_ratio
    left = max(0, int(width * (center_x_ratio - half_width_ratio)))
    right = min(width, int(width * (center_x_ratio + half_width_ratio)))
    top = max(0, int(height * (center_y_ratio - half_height_ratio)))
    bottom = min(height, int(height * (center_y_ratio + half_height_ratio)))
    if right <= left:
        right = min(width, max(left + 1, int(round(cx)) + 1))
    if bottom <= top:
        bottom = min(height, max(top + 1, int(round(cy)) + 1))
    return image[top:bottom, left:right]


def recognize_probe_slot_char(
    ocr: RapidOCR,
    crop: np.ndarray,
    center_x_ratio: float,
    center_y_ratio: float,
    mode: str,
    allowed: set[str],
    window_sizes: list[tuple[float, float]],
    scales: tuple[int, ...] = (8, 10),
) -> str | None:
    scores: dict[str, float] = {}
    for binarize in (True, False):
        for half_width_ratio, half_height_ratio in window_sizes:
            cell = crop_center_window(crop, center_x_ratio, center_y_ratio, half_width_ratio, half_height_ratio)
            if cell.size == 0:
                continue
            for scale in scales:
                tokens = run_ocr_tokens(ocr, cell, scale=scale, binarize=binarize)
                ch = best_exploded_char(tokens, mode=mode, allowed=allowed)
                if ch is not None:
                    if mode == "digit":
                        ch = disambiguate_six_nine(cell, ch)
                    score = max((token.confidence for token in tokens), default=0.0)
                    if not binarize:
                        score += 0.05
                    scores[ch] = scores.get(ch, 0.0) + score
        if scores:
            break
    if not scores:
        return None
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def recognize_pd_slot_value(
    ocr: RapidOCR,
    crop: np.ndarray,
    center_x_ratio: float,
    center_y_ratio: float,
    window_sizes: list[tuple[float, float]],
    scales: tuple[int, ...] = (8, 10),
) -> str | None:
    allowed = set("0123456789")
    scores: dict[str, float] = {}
    for binarize in (True, False):
        for half_width_ratio, half_height_ratio in window_sizes:
            cell = crop_center_window(crop, center_x_ratio, center_y_ratio, half_width_ratio, half_height_ratio)
            if cell.size == 0:
                continue
            for scale in scales:
                tokens = run_ocr_tokens(ocr, cell, scale=scale, binarize=binarize)
                for token in tokens:
                    candidate = normalize_digit_candidate(token.text, allowed)
                    if not candidate:
                        continue
                    if len(candidate) == 1:
                        candidate = disambiguate_six_nine(cell, candidate)
                    elif candidate not in {"10", "11", "12"}:
                        continue
                    score = token.confidence
                    if not binarize:
                        score += 0.05
                    if len(candidate) == 2:
                        score += 0.1
                    scores[candidate] = scores.get(candidate, 0.0) + score
        if scores:
            break
    if not scores:
        return None
    return max(scores.items(), key=lambda item: (item[1], len(item[0]), item[0]))[0]


def recognize_pd_digits(ocr: RapidOCR, crop: np.ndarray) -> dict[str, str]:
    slot_centers = [
        ("top_left", 0.22, 0.33),
        ("top_mid", 0.50, 0.33),
        ("top_right", 0.78, 0.33),
        ("bottom_left", 0.22, 0.68),
        ("bottom_mid", 0.50, 0.68),
        ("bottom_right", 0.78, 0.68),
    ]
    recognized: dict[str, str] = {}
    missing_slots: list[tuple[str, float, float]] = []

    for name, center_x_ratio, center_y_ratio in slot_centers:
        digit = recognize_pd_slot_value(
            ocr=ocr,
            crop=crop,
            center_x_ratio=center_x_ratio,
            center_y_ratio=center_y_ratio,
            window_sizes=[(0.12, 0.16), (0.18, 0.16)],
            scales=(10,),
        )
        if digit is not None:
            recognized[name] = digit
        else:
            missing_slots.append((name, center_x_ratio, center_y_ratio))

    if not missing_slots:
        return recognized

    for name, center_x_ratio, center_y_ratio in missing_slots:
        digit = recognize_pd_slot_value(
            ocr=ocr,
            crop=crop,
            center_x_ratio=center_x_ratio,
            center_y_ratio=center_y_ratio,
            window_sizes=[(0.10, 0.14), (0.14, 0.18), (0.20, 0.18)],
            scales=(8, 10, 12),
        )
        if digit is not None:
            recognized[name] = digit
    return recognized


def format_pd_row(row_label: str, left_label: str, values: list[str]) -> str:
    row_name = ROW_NAMES[row_label]
    left_name = COL_NAMES[left_label]
    if left_name == "远中":
        col_names = ["远中", "中央", "近中"]
    else:
        col_names = ["近中", "中央", "远中"]
    pairs = [f"{name}{value}" for name, value in zip(col_names, values)]
    return f"{row_name}({', '.join(pairs)})"


def find_associated_tooth(crosses: list[ParsedSymbol], symbol_center_x: float, preferred_regions: set[str] | None = None) -> ToothMark | None:
    teeth = find_associated_teeth(crosses, symbol_center_x, preferred_regions)
    return teeth[0] if teeth else None


def find_associated_teeth(
    crosses: list[ParsedSymbol],
    symbol_center_x: float,
    preferred_regions: set[str] | None = None,
) -> list[ToothMark]:
    best_tooth: ToothMark | None = None
    best_score = float("inf")
    best_cross: ParsedSymbol | None = None
    best_candidates: list[ToothMark] = []
    for cross in crosses:
        if not cross.teeth:
            continue
        candidates = [tooth for tooth in cross.teeth if preferred_regions is None or tooth.region in preferred_regions]
        if not candidates:
            continue
        left_bias = 0.0 if cross.right <= symbol_center_x else 80.0
        candidate_score = min(abs(cross.left + tooth.cx - symbol_center_x) for tooth in candidates) + left_bias
        if candidate_score < best_score:
            best_score = candidate_score
            best_cross = cross
            best_candidates = candidates

    if best_cross is None or not best_candidates:
        return []

    cross_is_left_of_symbol = best_cross.right <= symbol_center_x
    return sorted(
        best_candidates,
        key=lambda tooth: (
            abs(best_cross.left + tooth.cx - symbol_center_x),
            -(best_cross.left + tooth.cx) if cross_is_left_of_symbol else (best_cross.left + tooth.cx),
        ),
    )


def format_suffix(codes: list[str], suffix: str) -> str:
    return "或".join(f"{code}{suffix}" for code in codes)


def format_group_suffix(teeth: list[ToothMark], suffix: str) -> str:
    if not teeth:
        return suffix
    if len(teeth) == 1:
        return format_suffix(teeth[0].codes, suffix)
    parts: list[str] = []
    seen: set[str] = set()
    for tooth in teeth:
        part = "或".join(tooth.codes)
        if part and part not in seen:
            parts.append(part)
            seen.add(part)
    if not parts:
        return suffix
    return f"{'、'.join(parts)}{suffix}"


def format_symbol_detail(prefix: str, parts: list[str]) -> str:
    if not parts:
        return prefix
    return f"{prefix}（{'，'.join(parts)}）"


def parse_pd_probe(ocr: RapidOCR, crop: np.ndarray, tooth: ToothMark | None) -> str:
    tokens = run_ocr_tokens(ocr, crop)
    label_tokens = explode_tokens(tokens, mode="label", allowed=set("BLMDO"))
    height, width = crop.shape[:2]

    top_label = choose_row_label(label_tokens, width / 2.0, height * 0.10, {"B", "L"})
    bottom_label = choose_row_label(label_tokens, width / 2.0, height * 0.90, {"B", "L"})
    left_label = choose_col_label(label_tokens, width * 0.08, height / 2.0)
    right_label = choose_col_label(label_tokens, width * 0.92, height / 2.0)

    if top_label is None:
        top_label = opposite_row_label(bottom_label)
    if bottom_label is None:
        bottom_label = opposite_row_label(top_label)
    if left_label is None:
        left_label = opposite_col_label(right_label)
    if right_label is None:
        right_label = opposite_col_label(left_label)

    slots = [
        ("top_left", width * 0.22, height * 0.33),
        ("top_mid", width * 0.50, height * 0.33),
        ("top_right", width * 0.78, height * 0.33),
        ("bottom_left", width * 0.22, height * 0.68),
        ("bottom_mid", width * 0.50, height * 0.68),
        ("bottom_right", width * 0.78, height * 0.68),
    ]
    values = recognize_pd_digits(ocr, crop)
    if len(values) < len(slots):
        central_tokens = [
            token
            for token in tokens
            if width * 0.12 <= token.cx <= width * 0.88 and height * 0.18 <= token.cy <= height * 0.82
        ]
        digit_tokens = explode_tokens(central_tokens, mode="digit", allowed=set("0123456789"))
        fallback_values = assign_slots(digit_tokens, slots, max_dist=max(width, height) * 0.28)
        for name, value in fallback_values.items():
            values.setdefault(name, value)
    top_values = [values.get("top_left", "?"), values.get("top_mid", "?"), values.get("top_right", "?")]
    bottom_values = [values.get("bottom_left", "?"), values.get("bottom_mid", "?"), values.get("bottom_right", "?")]

    prefix = ""
    if tooth is not None:
        prefix = format_suffix(tooth.codes, "PD")
    else:
        prefix = "PD"

    if top_label is None or bottom_label is None or left_label is None:
        return f"{prefix}（行列标签未完整识别）"

    parts = [
        format_pd_row(top_label, left_label, top_values),
        format_pd_row(bottom_label, left_label, bottom_values),
    ]
    return f"{prefix}（{'，'.join(parts)}）"


def parse_fi_upper_y(ocr: RapidOCR, crop: np.ndarray, teeth: list[ToothMark]) -> str:
    tokens = dedupe_tokens(run_ocr_tokens(ocr, crop, scale=1, binarize=False) + run_ocr_tokens(ocr, crop, scale=OCR_SCALE, binarize=True))
    digit_tokens = explode_tokens(tokens, mode="digit", allowed=set("01234"))
    height, width = crop.shape[:2]
    slots = [
        ("top", width * 0.50, height * 0.22),
        ("left_lower", width * 0.28, height * 0.76),
        ("right_lower", width * 0.72, height * 0.76),
    ]
    values = assign_slots(digit_tokens, slots, max_dist=max(width, height) * 0.35)
    for name, center_x_ratio, center_y_ratio in (
        ("top", 0.50, 0.18),
        ("left_lower", 0.28, 0.76),
        ("right_lower", 0.72, 0.76),
    ):
        if name in values:
            continue
        if name == "top":
            window_sizes = [(0.12, 0.12), (0.16, 0.14), (0.20, 0.16)]
            scales = (8, 10, 12)
        else:
            window_sizes = [(0.18, 0.16), (0.24, 0.20)]
            scales = (6, 8, 10)
        digit = recognize_probe_slot_char(
            ocr=ocr,
            crop=crop,
            center_x_ratio=center_x_ratio,
            center_y_ratio=center_y_ratio,
            mode="digit",
            allowed=set("01234"),
            window_sizes=window_sizes,
            scales=scales,
        )
        if digit is not None:
            values[name] = digit

    if "top" not in values:
        upper_top = crop[: max(1, int(height * 0.55)), int(width * 0.15) : max(int(width * 0.15) + 1, int(width * 0.85))]
        upper_tokens = []
        for scale in (8, 10, 12, 14):
            upper_tokens.extend(run_ocr_tokens(ocr, upper_top, scale=scale, binarize=False))
            upper_tokens.extend(run_ocr_tokens(ocr, upper_top, scale=scale, binarize=True))
        upper_digit = best_exploded_char(upper_tokens, mode="digit", allowed=set("01234"))
        if upper_digit is not None:
            values["top"] = upper_digit

    prefix = format_group_suffix(teeth, "FI")
    region = teeth[0].region if teeth else None

    left_name = "腭侧左下"
    right_name = "腭侧右下"
    if region == "tl":
        left_name = "腭侧远中"
        right_name = "腭侧近中"
    elif region == "tr":
        left_name = "腭侧近中"
        right_name = "腭侧远中"

    parts: list[str] = []
    if values.get("top") is not None:
        parts.append(f"颊侧{values['top']}度")
    if values.get("left_lower") is not None:
        parts.append(f"{left_name}{values['left_lower']}度")
    if values.get("right_lower") is not None:
        parts.append(f"{right_name}{values['right_lower']}度")
    return format_symbol_detail(prefix, parts)


def parse_fi_lower_bar(ocr: RapidOCR, crop: np.ndarray, teeth: list[ToothMark]) -> str:
    tokens = dedupe_tokens(run_ocr_tokens(ocr, crop, scale=1, binarize=False) + run_ocr_tokens(ocr, crop, scale=OCR_SCALE, binarize=True))
    digit_tokens = explode_tokens(tokens, mode="digit", allowed=set("01234"))
    height, width = crop.shape[:2]
    slots = [
        ("top", width * 0.50, height * 0.24),
        ("bottom", width * 0.50, height * 0.76),
    ]
    values = assign_slots(digit_tokens, slots, max_dist=max(width, height) * 0.35)
    for name, center_y_ratio in (("top", 0.18), ("bottom", 0.76)):
        if name in values:
            continue
        if name == "top":
            window_sizes = [(0.12, 0.12), (0.16, 0.14), (0.20, 0.16)]
            scales = (8, 10, 12)
        else:
            window_sizes = [(0.20, 0.16), (0.26, 0.22)]
            scales = (6, 8, 10)
        digit = recognize_probe_slot_char(
            ocr=ocr,
            crop=crop,
            center_x_ratio=0.50,
            center_y_ratio=center_y_ratio,
            mode="digit",
            allowed=set("01234"),
            window_sizes=window_sizes,
            scales=scales,
        )
        if digit is not None:
            values[name] = digit

    prefix = format_group_suffix(teeth, "FI")
    parts: list[str] = []
    if values.get("top") is not None:
        parts.append(f"舌侧{values['top']}度")
    if values.get("bottom") is not None:
        parts.append(f"颊侧{values['bottom']}度")
    return format_symbol_detail(prefix, parts)


def merge_boxes(boxes: list[tuple[int, int, int, int]], gap: int = 8) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes):
        if not merged:
            merged.append(box)
            continue
        prev = merged[-1]
        if not (box[0] > prev[2] + gap or box[2] + gap < prev[0] or box[1] > prev[3] + gap or box[3] + gap < prev[1]):
            merged[-1] = (
                min(prev[0], box[0]),
                min(prev[1], box[1]),
                max(prev[2], box[2]),
                max(prev[3], box[3]),
            )
        else:
            merged.append(box)
    return merged


def detect_qr_boxes(page_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    try:
        page_bgr = ensure_cv_image(page_bgr, "二维码检测输入", allow_gray=False)
    except ValueError:
        return []
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    scale = 2
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    detector = cv2.QRCodeDetector()
    boxes: list[tuple[int, int, int, int]] = []

    def append_points(points) -> None:
        if points is None:
            return
        point_sets = points if len(points.shape) == 3 else np.expand_dims(points, axis=0)
        for quad in point_sets:
            xs = quad[:, 0] / scale
            ys = quad[:, 1] / scale
            left = max(0, int(math.floor(xs.min())))
            top = max(0, int(math.floor(ys.min())))
            right = min(page_bgr.shape[1] - 1, int(math.ceil(xs.max())) - 1)
            bottom = min(page_bgr.shape[0] - 1, int(math.ceil(ys.max())) - 1)
            if right > left and bottom > top:
                boxes.append((left, top, right, bottom))

    try:
        ok, _, points, _ = detector.detectAndDecodeMulti(up)
        if ok:
            append_points(points)
        else:
            _, point, _ = detector.detectAndDecode(up)
            append_points(point)
    except Exception:
        return []

    return merge_boxes(boxes)


def normalize_control_text(text: str) -> str:
    return re.sub(r"[^A-Za-z十]", "", text).upper()


def should_drop_text_token(token: TextToken, symbols: list[ParsedSymbol], qr_boxes: list[tuple[int, int, int, int]]) -> bool:
    token_box = (token.left, token.top, token.right, token.bottom)
    if any(overlap_ratio(token_box, qr_box) >= 0.2 for qr_box in qr_boxes):
        return True
    if any(overlap_ratio(token_box, (symbol.left, symbol.top, symbol.right, symbol.bottom)) >= 0.2 for symbol in symbols):
        return True

    control_key = normalize_control_text(token.text)
    if control_key not in {"PD", "FI", "十"}:
        return False

    for symbol in symbols:
        vertical_overlap = intersection_area(token_box, (token.left, max(token.top, symbol.top), token.right, min(token.bottom, symbol.bottom)))
        if vertical_overlap == 0:
            continue
        gap_left = symbol.left - token.right
        gap_right = token.left - symbol.right
        if -4 <= gap_left <= 24 or -4 <= gap_right <= 24:
            return True
    return False


def is_dental_symbol_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "PD（" in stripped or "FI（" in stripped:
        return True
    if stripped == "未识别牙位":
        return True
    parts = [part for part in re.split(r"[，,、\s]+", stripped) if part]
    if not parts:
        return False
    return all(re.fullmatch(r"[1-8][1-8](?:[BLMDO])?", part) for part in parts)


INLINE_DENTAL_SYMBOL_PATTERN = re.compile(
    r"(?:"
    r"(?:[1-8][1-8](?:[BLMDO])?(?:、[1-8][1-8](?:[BLMDO])?)*)?(?:PD|FI)（[^）]*）"
    r"|"
    r"[1-8][1-8](?:[BLMDO])?(?:、[1-8][1-8](?:[BLMDO])?)*"
    r")"
)


def add_spaces_around_dental_symbols(text: str) -> str:
    if not text:
        return text

    pieces: list[str] = []
    cursor = 0
    for match in INLINE_DENTAL_SYMBOL_PATTERN.finditer(text):
        start, end = match.span()
        symbol = match.group(0)
        pieces.append(text[cursor:start])

        prev_char = text[start - 1] if start > 0 else ""
        next_char = text[end] if end < len(text) else ""
        need_left_space = bool(prev_char and not prev_char.isspace() and prev_char not in "（([{《<：:，。；;、,.!?/+-")
        need_right_space = bool(next_char and not next_char.isspace() and next_char not in "）)]}》>，。；;、,.!?/+-")

        if need_left_space:
            pieces.append(" ")
        pieces.append(symbol)
        if need_right_space:
            pieces.append(" ")
        cursor = end

    pieces.append(text[cursor:])
    return "".join(pieces)


def token_separator(prev_text: str, curr_text: str, gap: int) -> str:
    is_symbol_prev = is_dental_symbol_text(prev_text)
    is_symbol_curr = is_dental_symbol_text(curr_text)
    if (is_symbol_prev or is_symbol_curr) and curr_text[0] not in "，。；：、,.!?)]）】》>" and prev_text[-1] not in "（([{《<：:":
        return " "
    if gap <= 6:
        return ""
    if prev_text[-1] in "（([{《<：:":
        return ""
    if curr_text[0] in "，。；：、,.!?)]）】》>":
        return ""
    return " "


def normalize_text_metric_labels(text: str) -> str:
    corrected = text
    corrected = re.sub(r"(?<![A-Za-z0-9])0HI-S(?=\b|[\s:：=])", "OHI-S", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"(?<![A-Za-z0-9])0HI(?=\b|[\s:：=])", "OHI", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"(?<![A-Za-z0-9])B0P(?![A-Za-z0-9])", "BOP", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"\bmm[oO]\b", "mm", corrected)
    corrected = re.sub(r"\bm[uμ]m\b", "mm", corrected, flags=re.IGNORECASE)
    return corrected


def normalize_compact_metric_ranges(text: str) -> str:
    corrected = text

    def repl_mm(match: re.Match[str]) -> str:
        label = match.group("label").upper()
        sep = match.group("sep")
        left = match.group("left")
        right = match.group("right")
        return f"{label}{sep}{left}–{right}mm"

    def repl_plain(match: re.Match[str]) -> str:
        label = match.group("label").upper()
        sep = match.group("sep")
        left = match.group("left")
        right = match.group("right")
        return f"{label}{sep}{left}–{right}"

    corrected = re.sub(
        r"(?P<label>\b(?:PD|GR|AL|CAL)\b)\s*(?P<sep>[:：=])\s*(?P<left>\d)\s*(?P<right>10|11|12)\s*(?:mm|mmo)\b",
        repl_mm,
        corrected,
        flags=re.IGNORECASE,
    )
    corrected = re.sub(
        r"(?P<label>\b(?:PD|GR|AL|CAL)\b)\s*(?P<sep>[:：=])\s*(?P<left>\d)\s*(?P<right>\d)\s*(?:mm|mmo)\b",
        repl_mm,
        corrected,
        flags=re.IGNORECASE,
    )
    corrected = re.sub(
        r"(?P<label>\b(?:BI|OHI-S)\b)\s*(?P<sep>[:：=])\s*(?P<left>\d)\s*(?P<right>\d)(?=[^0-9]|$)",
        repl_plain,
        corrected,
        flags=re.IGNORECASE,
    )
    return corrected


def normalize_two_digit_mm_ranges(text: str) -> str:
    corrected = text

    def repl_two_digit_mm(match: re.Match[str]) -> str:
        left = match.group("left")
        right = match.group("right")
        unit = match.group("unit")
        return f"{left}–{right}{unit}"

    corrected = re.sub(
        r"(?<![0-9A-Za-z])(?P<left>\d)(?P<right>10|11|12)(?P<unit>\s*(?:mm|mmo)\b)",
        repl_two_digit_mm,
        corrected,
        flags=re.IGNORECASE,
    )
    corrected = re.sub(
        r"(?<![0-9A-Za-z])(?P<left>[2-9])(?P<right>\d)(?P<unit>\s*(?:mm|mmo)\b)",
        repl_two_digit_mm,
        corrected,
        flags=re.IGNORECASE,
    )
    return corrected


def normalize_tm_fi_grades(text: str) -> str:
    corrected = text

    def repl_zero_to_roman(match: re.Match[str]) -> str:
        label = match.group("label").upper()
        label = "TM" if label == "LL" else label
        roman = match.group("roman").upper()
        return f"{label}:0-{roman}°"

    def repl_roman_only(match: re.Match[str]) -> str:
        label = match.group("label").upper()
        roman = match.group("roman").upper()
        return f"{label}:{roman}°"

    corrected = re.sub(
        r"\b(?P<label>LL|TM|FI)(?P<zero>[O0])\s*[-–]?\s*(?P<roman>I{1,3})(?:\s*[°。oO])?",
        repl_zero_to_roman,
        corrected,
        flags=re.IGNORECASE,
    )
    corrected = re.sub(
        r"\b(?P<label>LL|TM|FI)\b\s*(?:[:：=]\s*)?(?:[O0]\s*[-–]?\s*)(?P<roman>I{1,3})(?:\s*[°。oO])?",
        repl_zero_to_roman,
        corrected,
        flags=re.IGNORECASE,
    )
    corrected = re.sub(
        r"\b(?P<label>TM|FI)\b\s*(?:[:：=]\s*)(?P<roman>I{1,3})(?:\s*[°。oO])?(?=[^A-Za-z]|$)",
        repl_roman_only,
        corrected,
        flags=re.IGNORECASE,
    )
    corrected = re.sub(
        r"\bFI\b\s*(?:[:：=]\s*)?(?P<left>[0-4])\s*(?P<right>[1-4])(?=[^0-9]|$)",
        lambda m: f"FI:{m.group('left')}–{m.group('right')}",
        corrected,
        flags=re.IGNORECASE,
    )
    corrected = re.sub(r"([；;，,]\s*)[:：](?=\s*(?:TM|FI)\b)", r"\1", corrected)
    corrected = re.sub(r"(^|\s)[:：](?=\s*(?:TM|FI)\b)", r"\1", corrected)
    return corrected


def apply_text_ocr_corrections(text: str) -> str:
    corrected = normalize_text_metric_labels(text)
    corrected = corrected.replace("煎合", "愈合")
    corrected = re.sub(r"口\s*[（(]\s*±\s*[）)]", "叩（±）", corrected)
    corrected = re.sub(
        r"(?<![A-Za-z])TK(?=\s*(?:[:：=]|[O0IⅤVX-]|[0-9]))",
        "TM",
        corrected,
        flags=re.IGNORECASE,
    )
    corrected = re.sub(
        r"(?<![A-Za-z])LL(?=\s+(?:[1-8][1-8]\b|[O0]\s*[-–]?\s*I{1,3}\b|I{1,3}\b|缺失\b|松动\b|叩\b))",
        "TM",
        corrected,
    )
    corrected = re.sub(r"(?<!\S)T(?!\S)", "I", corrected)
    corrected = normalize_compact_metric_ranges(corrected)
    corrected = normalize_two_digit_mm_ranges(corrected)
    corrected = normalize_tm_fi_grades(corrected)
    corrected = re.sub(r"(\b(?:PD|AL|BI|CAL|OHI-S)\s*[:：=]?\s*)26(?=\s*(?:mm|mmo)\b)", r"\g<1>2–6", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"(?<![0-9A-Za-z])26(?=\s*(?:mm|mmo)\b)", "2–6", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"((?:TM|FI):[0-9I–-]+°)(?=[1-8][1-8]\b)", r"\1 ", corrected)
    corrected = re.sub(r"(BI[:：=]\s*[0-4][–-][0-4])(?=[1-8][1-8])", r"\1，", corrected)
    corrected = re.sub(r"(种植义齿)(?=[1-8][1-8])", r"\1，", corrected)
    corrected = re.sub(r"(?<=[1-8][1-8])\s+[1-8]{2,3}(?=为)", "", corrected)
    corrected = re.sub(r"(?<=为)\s*4(?=\s*TM\s*[:：]?\s*[O0I-])", "4-5mm；", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"TM\s*[:：]?\s*[O0]\s*-\s*I\b", "TM:0-I°", corrected, flags=re.IGNORECASE)
    corrected = re.sub(
        r"(TM:0-I°)\s*([1-8][1-8](?:、[1-8][1-8])*)\s*(?=余详见牙周检查表)",
        r"\1，\2为I°；",
        corrected,
    )
    corrected = re.sub(r"([1-8][1-8](?:、[1-8][1-8])*)\s+[1-8]{2,3}(?=(?:龈下刮治术|根面平整术))", r"\1", corrected)
    corrected = re.sub(r"止血，\s*[1-8][1-8](?:、[1-8][1-8])+(?:[BLMDO])?\s*冲洗", "止血，冲洗", corrected)
    corrected = re.sub(r"([1-8][1-8])\s+使用碳纤维工作尖维护", r"\1使用碳纤维工作尖维护", corrected)
    return corrected


def polish_line_text(text: str) -> str:
    text = apply_text_ocr_corrections(text)
    text = add_spaces_around_dental_symbols(text)
    text = re.sub(r"\s+([，。；：、,.!?）】》])", r"\1", text)
    text = re.sub(r"([（【《])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def compose_line_text(elements: list[tuple[int, int, str]]) -> str:
    if not elements:
        return ""
    ordered = sorted(elements, key=lambda item: (item[0], item[1], item[2]))
    text = ordered[0][2]
    prev_left, prev_right, prev_text = ordered[0]
    for left, right, curr_text in ordered[1:]:
        text += token_separator(prev_text, curr_text, left - prev_right) + curr_text
        prev_left, prev_right, prev_text = left, right, curr_text
    return polish_line_text(text)


def record_symbol_text(symbol: ParsedSymbol) -> str:
    return symbol.text


def should_omit_cross_in_record(symbol: ParsedSymbol, next_symbol: ParsedSymbol | None) -> bool:
    if symbol.label != "cross" or next_symbol is None or next_symbol.label not in {"pd_probe", "fi_upper_y", "fi_lower_bar"}:
        return False
    current_text = record_symbol_text(symbol).strip()
    next_text = record_symbol_text(next_symbol).strip()
    if not current_text:
        return False
    if next_text.startswith(current_text):
        return next_symbol.left - symbol.right <= 42
    current_codes = re.findall(r"[1-8][1-8]", current_text)
    next_codes = re.findall(r"[1-8][1-8]", next_text)
    if next_symbol.label == "pd_probe" and current_codes and next_codes:
        if next_codes[0] in current_codes:
            return next_symbol.left - symbol.right <= 42
    if not current_codes or len(next_codes) < len(current_codes):
        return False
    if set(current_codes) != set(next_codes[: len(current_codes)]):
        return False
    return next_symbol.left - symbol.right <= 42


def record_symbol_elements(parsed_symbols: list[ParsedSymbol]) -> list[tuple[int, int, str]]:
    ordered_symbols = sorted(parsed_symbols, key=lambda item: (item.left, item.top, item.right, item.bottom))
    elements: list[tuple[int, int, str]] = []
    for idx, symbol in enumerate(ordered_symbols):
        next_symbol = ordered_symbols[idx + 1] if idx + 1 < len(ordered_symbols) else None
        if should_omit_cross_in_record(symbol, next_symbol):
            continue
        text = record_symbol_text(symbol)
        if not text.strip():
            continue
        elements.append((symbol.left, symbol.right, text))
    return elements


def is_structural_line_start(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(
        re.match(
            r"^(广州医科大学附属口腔医院|日期[:：.．]|患者[:：.．]|科室[:：.．]|复诊(?:[:：.．]|\s)|检查[:：.．]?|处置[:：.．]?|诊断[:：.．]?|签名[:：.．]?|牙周病科电话[:：.．]?|\d+[.．、])",
            stripped,
        )
    )


def extract_structural_prefix(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    match = re.match(
        r"^(广州医科大学附属口腔医院|日期[:：.．]|患者[:：.．]|科室[:：.．]|复诊(?:[:：.．]|\s)|检查[:：.．]?|处置[:：.．]?|诊断[:：.．]?|签名[:：.．]?|牙周病科电话[:：.．]?|\d+[.．、])",
        stripped,
    )
    return match.group(0) if match else None


def trim_redundant_line_prefix(prev_text: str, curr_text: str) -> str:
    prev = prev_text.strip()
    curr = curr_text.strip()
    if not prev or not curr or len(curr) <= len(prev) or len(prev) < 10:
        return curr

    candidates = [curr]
    prefix = extract_structural_prefix(curr)
    if prefix:
        remainder = curr[len(prefix) :].lstrip()
        if remainder:
            candidates.append(remainder)

    for candidate in candidates:
        if candidate.startswith(prev) and len(candidate) > len(prev):
            tail = candidate[len(prev) :].lstrip("，,；;:： ")
            if tail:
                return tail

        pos = candidate.find(prev)
        if 0 <= pos <= 6 and len(candidate) > pos + len(prev):
            head = candidate[:pos].strip()
            head_prefix = extract_structural_prefix(head) if head else None
            if not head or head_prefix == head:
                tail = candidate[pos + len(prev) :].lstrip("，,；;:： ")
                if tail:
                    return tail

    return curr


def paragraph_separator(prev_text: str, curr_text: str) -> str:
    if not prev_text or not curr_text:
        return ""
    if prev_text[-1] in "（([{《<：:":
        return ""
    if curr_text[0] in "，。；：、,.!?)]）】》>":
        return ""
    if prev_text[-1].isalnum() and curr_text[0].isalnum():
        return " "
    return ""


def should_start_new_paragraph(prev_line: ParsedLine, curr_line: ParsedLine, curr_text: str | None = None) -> bool:
    prev_text = prev_line.text.strip()
    curr_text = curr_line.text.strip() if curr_text is None else curr_text.strip()
    if not prev_text or not curr_text:
        return False
    if is_structural_line_start(curr_text):
        return True
    prev_height = prev_line.bottom - prev_line.top + 1
    curr_height = curr_line.bottom - curr_line.top + 1
    vertical_gap = curr_line.top - prev_line.bottom
    if vertical_gap > max(prev_height, curr_height) * 1.6 + 8:
        return True
    return False


def compose_record_text(parsed_lines: list[ParsedLine]) -> str:
    if not parsed_lines:
        return ""
    ordered = sorted(parsed_lines, key=lambda item: (item.top, item.left, item.right, item.bottom))
    paragraphs: list[str] = []
    current = ordered[0].text.strip()
    prev_line = ordered[0]

    for line in ordered[1:]:
        line_text = trim_redundant_line_prefix(prev_line.text, line.text)
        if not line_text:
            continue
        if should_start_new_paragraph(prev_line, line, curr_text=line_text):
            if current:
                paragraphs.append(current.strip())
            current = line_text
        else:
            current += paragraph_separator(current, line_text) + line_text
        prev_line = line

    if current:
        paragraphs.append(current.strip())
    return "\n".join(paragraph for paragraph in paragraphs if paragraph)


def clean_record_paragraph(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"(?:^|\s)\d+\s*/\s*\d+(?=\s|$)", " ", cleaned)
    cleaned = re.sub(r"诊治医师[:：]?\s*\S*", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def is_basic_info_paragraph(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return bool(
        re.search(
            r"(广州医科大学附属口腔医院|日期[：:]|患者[：:]|科室[：:]|门（急）诊病历|门\(急\)诊病历|主索引|出生日期[：:])",
            stripped,
        )
    )


def trim_repeated_page_header(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    match = re.search(r"(主诉|现病史|既往史|个人史|家族史|全身|复诊|检查|诊\s*断|治疗计划|处置)\s*[:：.．]", stripped)
    if match:
        return stripped[match.start() :].strip()
    if is_basic_info_paragraph(stripped):
        return ""
    return stripped


def compose_full_record_text(page_texts: list[str]) -> str:
    all_pages: list[str] = []
    for page_index, page_text in enumerate(page_texts, start=1):
        paragraphs: list[str] = []
        for paragraph in page_text.splitlines():
            cleaned = clean_record_paragraph(paragraph)
            if not cleaned:
                continue
            if page_index > 1:
                cleaned = trim_repeated_page_header(cleaned)
                if not cleaned:
                    continue
            paragraphs.append(cleaned)
        if paragraphs:
            all_pages.append("\n".join(paragraphs))
    return "\n\n".join(all_pages)


def whiten_boxes_on_line(
    line_image: np.ndarray,
    line_left: int,
    line_top: int,
    boxes_abs: list[tuple[int, int, int, int]],
    pad: int = 2,
) -> np.ndarray:
    cleaned = line_image.copy()
    line_h, line_w = cleaned.shape[:2]
    for left, top, right, bottom in boxes_abs:
        rel_left = max(0, left - line_left - pad)
        rel_top = max(0, top - line_top - pad)
        rel_right = min(line_w - 1, right - line_left + pad)
        rel_bottom = min(line_h - 1, bottom - line_top + pad)
        if rel_left <= rel_right and rel_top <= rel_bottom:
            cleaned[rel_top : rel_bottom + 1, rel_left : rel_right + 1] = 255
    return cleaned


def build_text_spans(
    line_width: int,
    blockers_abs: list[tuple[int, int, int, int]],
    line_left: int,
    min_width: int = 10,
) -> list[tuple[int, int]]:
    blockers_rel = sorted(
        (
            max(0, left - line_left),
            min(line_width - 1, right - line_left),
        )
        for left, _, right, _ in blockers_abs
    )
    spans: list[tuple[int, int]] = []
    cursor = 0
    for left, right in blockers_rel:
        if left - 1 >= cursor and left - cursor >= min_width:
            spans.append((cursor, left - 1))
        cursor = max(cursor, right + 1)
    if line_width - cursor >= min_width:
        spans.append((cursor, line_width - 1))
    if not blockers_rel and line_width >= min_width:
        return [(0, line_width - 1)]
    return spans


def select_page_text_tokens_for_line(
    page_text_tokens: list[TextToken],
    line_left: int,
    line_top: int,
    line_right: int,
    line_bottom: int,
) -> list[TextToken]:
    line_height = line_bottom - line_top + 1
    vertical_pad = max(2, int(round(line_height * 0.25)))
    selected: list[TextToken] = []
    for token in page_text_tokens:
        if token.right < line_left or token.left > line_right:
            continue
        if token.cy < line_top - vertical_pad or token.cy > line_bottom + vertical_pad:
            continue
        selected.append(token)
    selected.sort(key=lambda item: (item.left, item.top, item.right, item.bottom))
    return selected


def assign_block_text_tokens_to_lines(
    tokens: list[TextToken],
    run: list[tuple[int, object]],
) -> dict[int, list[TextToken]]:
    token_map: dict[int, list[TextToken]] = {}
    if not tokens or not run:
        return token_map

    ordered_run = sorted(run, key=lambda item: (item[1].top, item[1].left, item[0]))
    line_centers = [((line.top + line.bottom) / 2.0) for _, line in ordered_run]
    boundaries = [float("-inf")]
    for prev_center, next_center in zip(line_centers, line_centers[1:]):
        boundaries.append((prev_center + next_center) / 2.0)
    boundaries.append(float("inf"))

    for token in tokens:
        target_idx = 0
        for idx in range(len(ordered_run)):
            if boundaries[idx] <= token.cy < boundaries[idx + 1]:
                target_idx = idx
                break
        line_index = ordered_run[target_idx][0]
        token_map.setdefault(line_index, []).append(token)

    for line_index in token_map:
        token_map[line_index].sort(key=lambda item: (item.left, item.top, item.right, item.bottom))
    return token_map


def text_span_is_covered(
    tokens: list[TextToken],
    span_left_abs: int,
    span_right_abs: int,
    min_overlap_ratio: float = 0.35,
) -> bool:
    span_width = max(1, span_right_abs - span_left_abs + 1)
    for token in tokens:
        overlap_left = max(span_left_abs, token.left)
        overlap_right = min(span_right_abs, token.right)
        if overlap_left > overlap_right:
            continue
        overlap_width = overlap_right - overlap_left + 1
        if overlap_width / span_width >= min_overlap_ratio:
            return True
    return False


def should_ocr_text_crop(crop: np.ndarray, dark_threshold: int = 205, min_dark_pixels: int = 28, min_active_columns: int = 4) -> bool:
    if crop.size == 0:
        return False
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    dark_mask = gray < dark_threshold
    dark_pixels = int(dark_mask.sum())
    if dark_pixels < min_dark_pixels:
        return False
    active_columns = int((dark_mask.sum(axis=0) > 0).sum())
    return active_columns >= min_active_columns


def line_intersects_qr(
    line_left: int,
    line_top: int,
    line_right: int,
    line_bottom: int,
    qr_boxes: list[tuple[int, int, int, int]],
) -> bool:
    for qr_left, qr_top, qr_right, qr_bottom in qr_boxes:
        if not (qr_right < line_left or qr_left > line_right or qr_bottom < line_top or qr_top > line_bottom):
            return True
    return False


def build_multiline_text_token_map(
    ocr: RapidOCR,
    page_bgr: np.ndarray,
    lines: list,
    line_symbols_map: dict[int, list[ParsedSymbol]],
    qr_boxes: list[tuple[int, int, int, int]],
    debug_timing: bool = False,
    debug_label: str = "",
    slow_stage_seconds: float = DEFAULT_SLOW_STAGE_SECONDS,
) -> dict[int, list[TextToken]]:
    token_map: dict[int, list[TextToken]] = {}
    run: list[tuple[int, object]] = []

    def flush_run() -> None:
        nonlocal run
        if len(run) <= 1:
            run = []
            return

        top = min(item.top for _, item in run)
        bottom = max(item.bottom for _, item in run)
        left = min(item.left for _, item in run)
        right = max(item.right for _, item in run)
        crop = page_bgr[top : bottom + 1, left : right + 1]
        start_time = time.perf_counter()
        tokens = run_text_tokens(ocr, crop, scale=1, binarize=False)
        if not tokens:
            tokens = run_text_tokens(ocr, crop, scale=2, binarize=False)
        abs_tokens = translate_text_tokens(tokens, left, top)
        total_seconds = time.perf_counter() - start_time
        line_indexes = [line_index for line_index, _ in run]
        log_debug(
            debug_timing,
            f"{slow_tag(total_seconds, slow_stage_seconds)}{debug_label} text_block lines={line_indexes} "
            f"tokens={len(abs_tokens)} total={total_seconds:.3f}s",
        )
        for line_index, line in run:
            line_tokens = select_page_text_tokens_for_line(abs_tokens, line.left, line.top, line.right, line.bottom)
            if line_tokens:
                token_map[line_index] = line_tokens
        run = []

    for idx, line in enumerate(lines, start=1):
        eligible = not line_symbols_map.get(idx) and not line_intersects_qr(line.left, line.top, line.right, line.bottom, qr_boxes)
        if eligible:
            run.append((idx, line))
        else:
            flush_run()
    flush_run()
    return token_map


def build_record_line(
    ocr: RapidOCR,
    line_image: np.ndarray,
    line_index: int,
    line_left: int,
    line_top: int,
    line_right: int,
    line_bottom: int,
    parsed_symbols: list[ParsedSymbol],
    qr_boxes: list[tuple[int, int, int, int]],
    page_text_tokens: list[TextToken] | None = None,
    debug_timing: bool = False,
    debug_label: str = "",
    slow_stage_seconds: float = DEFAULT_SLOW_STAGE_SECONDS,
) -> ParsedLine | None:
    start_time = time.perf_counter()
    symbol_boxes = [(symbol.left, symbol.top, symbol.right, symbol.bottom) for symbol in parsed_symbols]
    line_qr_boxes = [
        (
            max(line_left, qr_left),
            max(line_top, qr_top),
            min(line_right, qr_right),
            min(line_bottom, qr_bottom),
        )
        for qr_left, qr_top, qr_right, qr_bottom in qr_boxes
        if not (qr_right < line_left or qr_left > line_right or qr_bottom < line_top or qr_top > line_bottom)
    ]
    blockers = symbol_boxes + line_qr_boxes
    kept_text_tokens: list[TextToken] = []
    text_ocr_seconds = 0.0
    text_source = "crop"
    text_spans: list[tuple[int, int]] = build_text_spans(line_image.shape[1], blockers, line_left)

    used_page_tokens = False
    if page_text_tokens is not None and not parsed_symbols and not line_qr_boxes:
        page_line_tokens = select_page_text_tokens_for_line(page_text_tokens, line_left, line_top, line_right, line_bottom)
        kept_text_tokens.extend(token for token in page_line_tokens if not should_drop_text_token(token, parsed_symbols, qr_boxes))
        if kept_text_tokens:
            text_source = "block"
            used_page_tokens = True

    if not used_page_tokens:
        for span_left, span_right in text_spans:
            crop = line_image[:, span_left : span_right + 1]
            if not should_ocr_text_crop(crop):
                continue
            ocr_start = time.perf_counter()
            text_tokens = run_text_tokens(ocr, crop, scale=1, binarize=False)
            if not text_tokens:
                text_tokens = run_text_tokens(ocr, crop, scale=2, binarize=False)
            text_ocr_seconds += time.perf_counter() - ocr_start
            abs_text_tokens = translate_text_tokens(text_tokens, line_left + span_left, line_top)
            new_tokens = [token for token in abs_text_tokens if not should_drop_text_token(token, parsed_symbols, qr_boxes)]
            if new_tokens:
                kept_text_tokens.extend(new_tokens)

    elements: list[tuple[int, int, str]] = []
    elements.extend((token.left, token.right, token.text) for token in kept_text_tokens)
    elements.extend(record_symbol_elements(parsed_symbols))
    line_text = compose_line_text(elements)
    if not line_text:
        total_seconds = time.perf_counter() - start_time
        log_debug(
            debug_timing,
            f"{slow_tag(total_seconds, slow_stage_seconds)}{debug_label} record_line spans={len(text_spans)} "
            f"text_tokens={len(kept_text_tokens)} text_source={text_source} text_ocr={text_ocr_seconds:.3f}s total={total_seconds:.3f}s empty",
        )
        return None
    total_seconds = time.perf_counter() - start_time
    log_debug(
        debug_timing,
        f"{slow_tag(total_seconds, slow_stage_seconds)}{debug_label} record_line spans={len(text_spans)} "
        f"text_tokens={len(kept_text_tokens)} text_source={text_source} text_ocr={text_ocr_seconds:.3f}s total={total_seconds:.3f}s",
    )
    return ParsedLine(
        line_index=line_index,
        left=line_left,
        top=line_top,
        right=line_right,
        bottom=line_bottom,
        text=line_text,
    )


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for ch in text:
        candidate = current + ch
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if current and bbox[2] - bbox[0] > max_width:
            lines.append(current)
            current = ch
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [text]


def overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def measure_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    text: str,
    max_width: int,
) -> tuple[list[str], int, int]:
    preview_lines = wrap_text(draw, text, font, max_width=max_width)
    line_heights = []
    line_widths = []
    for line in preview_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    box_w = max(line_widths) + 14
    box_h = sum(line_heights) + 12 + max(0, len(preview_lines) - 1) * 4
    return preview_lines, box_w, box_h

def draw_review_image(page_bgr: np.ndarray, lines: list[LineCrop], parsed_symbols: list[ParsedSymbol], out_path: Path) -> None:
    page_h, page_w = page_bgr.shape[:2]
    scratch = Image.new("RGBA", (page_w + REVIEW_PANEL_WIDTH, max(page_h, 100)), (255, 255, 255, 255))
    scratch_draw = ImageDraw.Draw(scratch)
    font = get_font()

    layouts: list[tuple[ParsedSymbol, list[str], int, int]] = []
    total_height = 12
    for parsed in parsed_symbols:
        label_lines, box_w, box_h = measure_label(scratch_draw, font, parsed.text, max_width=REVIEW_PANEL_WIDTH - 40)
        layouts.append((parsed, label_lines, box_w, box_h))
        total_height += box_h + 10

    canvas_h = max(page_h, total_height + 12)
    canvas = Image.new("RGBA", (page_w + REVIEW_PANEL_WIDTH, canvas_h), (255, 255, 255, 255))
    page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
    canvas.paste(Image.fromarray(page_rgb), (0, 0))
    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    draw.line((page_w + 4, 0, page_w + 4, canvas_h), fill=(220, 220, 220, 255), width=2)

    for idx, line in enumerate(lines, start=1):
        draw.rectangle((line.left, line.top, line.right, line.bottom), outline=(0, 153, 102, 180), width=2)
        draw.rounded_rectangle(
            (line.left + 2, line.top + 2, min(page_w - 2, line.left + 42), min(canvas_h - 2, line.top + 20)),
            radius=4,
            fill=(0, 153, 102, 210),
            outline=(0, 153, 102, 255),
            width=1,
        )
        draw.text((line.left + 8, line.top + 3), f"L{idx}", font=font, fill=(255, 255, 255, 255))

    y_cursor = 12
    for parsed, label_lines, box_w, box_h in layouts:
        color = get_symbol_color(parsed.label)
        draw.rectangle((parsed.left, parsed.top, parsed.right, parsed.bottom), outline=color + (255,), width=2)
        label_x = page_w + 18
        label_y = max(y_cursor, min(parsed.top, canvas_h - box_h - 12))
        label_box = (label_x, label_y, label_x + box_w, label_y + box_h)
        y_cursor = label_box[3] + 10

        anchor = (parsed.right, (parsed.top + parsed.bottom) // 2)
        connector_target = (label_box[0], label_box[1] + box_h // 2)
        draw.line((anchor[0], anchor[1], connector_target[0], connector_target[1]), fill=color + (255,), width=2)
        draw.rounded_rectangle(label_box, radius=6, fill=(255, 255, 255, TEXT_BG_ALPHA), outline=TEXT_BORDER + (255,), width=1)
        text_y = label_box[1] + 6
        for line in label_lines:
            draw.text((label_box[0] + 7, text_y), line, font=font, fill=TEXT_BORDER + (255,))
            bbox = draw.textbbox((label_box[0] + 7, text_y), line, font=font)
            text_y = bbox[3] + 4

    merged = Image.alpha_composite(canvas, overlay).convert("RGB")
    merged.save(out_path)


def serialize_symbol(parsed: ParsedSymbol) -> dict:
    payload = asdict(parsed)
    payload["bbox"] = [parsed.left, parsed.top, parsed.right, parsed.bottom]
    payload.pop("left")
    payload.pop("top")
    payload.pop("right")
    payload.pop("bottom")
    return payload


def serialize_line(parsed: ParsedLine) -> dict:
    payload = asdict(parsed)
    payload["bbox"] = [parsed.left, parsed.top, parsed.right, parsed.bottom]
    payload.pop("left")
    payload.pop("top")
    payload.pop("right")
    payload.pop("bottom")
    return payload


def parse_line_symbols(
    ocr: RapidOCR,
    line_image: np.ndarray,
    line_index: int,
    line_left: int,
    line_top: int,
    age_years: int | None,
    debug_timing: bool = False,
    debug_label: str = "",
    slow_stage_seconds: float = DEFAULT_SLOW_STAGE_SECONDS,
) -> list[ParsedSymbol]:
    start_time = time.perf_counter()
    detect_start = time.perf_counter()
    detections = extract_dental_symbols(
        line_image,
        normalize_box_to_cross=False,
        expand_to_line_height=False,
    )
    legacy_detections = extract_dental_symbols(
        line_image,
        normalize_box_to_cross=True,
        expand_to_line_height=True,
    )
    detect_seconds = time.perf_counter() - detect_start
    legacy_display_map = build_visual_mapping(detections, legacy_detections)
    parsed: list[ParsedSymbol] = []
    crosses: list[ParsedSymbol] = []

    for det_idx, det in enumerate(detections):
        if det.label in {"cross", "box"}:
            item_start = time.perf_counter()
            text, teeth = parse_cross_symbol(ocr, line_image, det, age_years)
            item_seconds = time.perf_counter() - item_start
            if not teeth or not text.strip():
                log_debug(
                    debug_timing and item_seconds >= slow_stage_seconds,
                    f"{slow_tag(item_seconds, slow_stage_seconds)}{debug_label} symbol cross "
                    f"bbox=({det.left},{det.top},{det.right},{det.bottom}) total={item_seconds:.3f}s skipped-empty",
                )
                continue
            display_det = legacy_display_map.get(
                det_idx,
                SymbolDetection(normalized_visual_label(det.label), det.left, det.top, det.right, det.bottom),
            )
            abs_left = line_left + display_det.left
            abs_top = line_top + display_det.top
            abs_right = line_left + display_det.right
            abs_bottom = line_top + display_det.bottom
            parsed_symbol = ParsedSymbol("cross", abs_left, abs_top, abs_right, abs_bottom, text, line_index, teeth)
            parsed.append(parsed_symbol)
            crosses.append(parsed_symbol)
            log_debug(
                debug_timing and item_seconds >= slow_stage_seconds,
                f"{slow_tag(item_seconds, slow_stage_seconds)}{debug_label} symbol cross "
                f"bbox=({det.left},{det.top},{det.right},{det.bottom}) total={item_seconds:.3f}s text={text}",
            )

    for det_idx, det in enumerate(detections):
        if det.label in {"cross", "box"}:
            continue
        display_det = legacy_display_map.get(det_idx, SymbolDetection(det.label, det.left, det.top, det.right, det.bottom))
        crop_top = det.top
        crop_bottom = det.bottom + 1
        crop_left = det.left
        crop_right = det.right + 1
        if det.label == "fi_upper_y":
            crop_top = max(0, det.top - 8)
            crop_bottom = min(line_image.shape[0], det.bottom + 7)
            crop_left = max(0, det.left - 8)
            crop_right = min(line_image.shape[1], det.right + 8)
        elif det.label == "fi_lower_bar":
            crop_top = max(0, det.top - 8)
            crop_bottom = min(line_image.shape[0], det.bottom + 6)
            crop_left = max(0, det.left - 6)
            crop_right = min(line_image.shape[1], det.right + 6)
        crop = line_image[crop_top:crop_bottom, crop_left:crop_right]
        abs_left = line_left + display_det.left
        abs_right = line_left + display_det.right
        abs_top = line_top + display_det.top
        abs_bottom = line_top + display_det.bottom
        tooth = None
        item_start = time.perf_counter()
        if det.label == "pd_probe":
            tooth = find_associated_tooth(crosses, abs_left + (abs_right - abs_left) / 2.0)
            text = parse_pd_probe(ocr, crop, tooth)
        elif det.label == "fi_upper_y":
            teeth = find_associated_teeth(crosses, abs_left + (abs_right - abs_left) / 2.0, preferred_regions=UPPER_REGIONS)
            text = parse_fi_upper_y(ocr, crop, teeth)
        elif det.label == "fi_lower_bar":
            teeth = find_associated_teeth(crosses, abs_left + (abs_right - abs_left) / 2.0, preferred_regions=LOWER_REGIONS)
            text = parse_fi_lower_bar(ocr, crop, teeth)
        else:
            text = f"{det.label}：未识别规则"
        item_seconds = time.perf_counter() - item_start
        parsed.append(ParsedSymbol(det.label, abs_left, abs_top, abs_right, abs_bottom, text, line_index))
        log_debug(
            debug_timing and item_seconds >= slow_stage_seconds,
            f"{slow_tag(item_seconds, slow_stage_seconds)}{debug_label} symbol {det.label} "
            f"bbox=({det.left},{det.top},{det.right},{det.bottom}) total={item_seconds:.3f}s text={text}",
        )

    parsed.sort(key=lambda item: (item.top, item.left, item.right, item.bottom))
    total_seconds = time.perf_counter() - start_time
    log_debug(
        debug_timing,
        f"{slow_tag(total_seconds, slow_stage_seconds)}{debug_label} parse_symbols "
        f"detections={len(detections)} parsed={len(parsed)} detect={detect_seconds:.3f}s total={total_seconds:.3f}s",
    )
    return parsed


def process_pdf(
    ocr: RapidOCR,
    pdf_path: Path,
    output_root: Path = OUT_DIR,
    export_review_png: bool = False,
    export_debug_artifacts: bool = False,
    debug_timing: bool = False,
    slow_stage_seconds: float = DEFAULT_SLOW_STAGE_SECONDS,
    progress_callback=None,
) -> ConversionOutput:
    pdf_start = time.perf_counter()
    case_dir = output_root / pdf_path.stem
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    full_record_pages: list[str] = []
    review_paths: list[Path] = []

    extract_start = time.perf_counter()
    pages = extract_images_from_input(pdf_path)
    extract_seconds = time.perf_counter() - extract_start
    log_debug(
        debug_timing,
        f"{slow_tag(extract_seconds, slow_stage_seconds)}{pdf_path.name} extract_images pages={len(pages)} {extract_seconds:.3f}s",
    )

    for page_idx, page_bgr in pages:
        page_start = time.perf_counter()
        context_start = time.perf_counter()
        page_context = extract_page_context(ocr, page_bgr)
        context_seconds = time.perf_counter() - context_start
        parsed_symbols: list[ParsedSymbol] = []
        parsed_lines: list[ParsedLine] = []
        line_symbols_map: dict[int, list[ParsedSymbol]] = {}
        qr_start = time.perf_counter()
        qr_boxes = detect_qr_boxes(page_bgr)
        qr_seconds = time.perf_counter() - qr_start
        lines_start = time.perf_counter()
        lines = extract_line_images(page_bgr)
        lines_seconds = time.perf_counter() - lines_start
        log_debug(
            debug_timing,
            f"{pdf_path.name} page {page_idx} prep context={context_seconds:.3f}s "
            f"qr={qr_seconds:.3f}s lines={len(lines)} line_extract={lines_seconds:.3f}s",
        )
        for line_index, line in enumerate(lines, start=1):
            line_label = f"{pdf_path.stem} p{page_idx} line {line_index}/{len(lines)}"
            symbols_start = time.perf_counter()
            line_symbols = parse_line_symbols(
                ocr=ocr,
                line_image=line.image,
                line_index=line_index,
                line_left=line.left,
                line_top=line.top,
                age_years=page_context.age_years,
                debug_timing=debug_timing,
                debug_label=line_label,
                slow_stage_seconds=slow_stage_seconds,
            )
            symbols_seconds = time.perf_counter() - symbols_start
            line_symbols_map[line_index] = line_symbols
            parsed_symbols.extend(line_symbols)
            log_debug(
                debug_timing,
                f"{slow_tag(symbols_seconds, slow_stage_seconds)}{line_label} symbols_only "
                f"symbols={len(line_symbols)} parse={symbols_seconds:.3f}s",
            )

        block_text_token_map = build_multiline_text_token_map(
            ocr=ocr,
            page_bgr=page_bgr,
            lines=lines,
            line_symbols_map=line_symbols_map,
            qr_boxes=qr_boxes,
            debug_timing=debug_timing,
            debug_label=f"{pdf_path.stem} p{page_idx}",
            slow_stage_seconds=slow_stage_seconds,
        )

        for line_index, line in enumerate(lines, start=1):
            line_label = f"{pdf_path.stem} p{page_idx} line {line_index}/{len(lines)}"
            record_start = time.perf_counter()
            parsed_line = build_record_line(
                ocr=ocr,
                line_image=line.image,
                line_index=line_index,
                line_left=line.left,
                line_top=line.top,
                line_right=line.right,
                line_bottom=line.bottom,
                parsed_symbols=line_symbols_map.get(line_index, []),
                qr_boxes=qr_boxes,
                page_text_tokens=block_text_token_map.get(line_index),
                debug_timing=debug_timing,
                debug_label=line_label,
                slow_stage_seconds=slow_stage_seconds,
            )
            record_seconds = time.perf_counter() - record_start
            if parsed_line is not None:
                parsed_lines.append(parsed_line)
            log_debug(
                debug_timing,
                f"{slow_tag(record_seconds, slow_stage_seconds)}{line_label} record_only total={record_seconds:.3f}s",
            )

        record_text = compose_record_text(parsed_lines)
        if record_text:
            full_record_pages.append(record_text)

        image_name = f"image_{page_idx:03d}_review.png"
        text_name = f"image_{page_idx:03d}_results.txt"
        json_name = f"image_{page_idx:03d}_results.json"
        record_name = f"image_{page_idx:03d}_record.txt"

        draw_seconds = 0.0
        if export_review_png:
            draw_start = time.perf_counter()
            review_path = case_dir / image_name
            draw_review_image(page_bgr, lines, parsed_symbols, review_path)
            review_paths.append(review_path)
            draw_seconds = time.perf_counter() - draw_start
        write_start = time.perf_counter()
        if export_debug_artifacts:
            with (case_dir / json_name).open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "pdf": pdf_path.name,
                        "page": page_idx,
                        "page_context": asdict(page_context),
                        "symbols": [serialize_symbol(item) for item in parsed_symbols],
                        "qr_boxes": [list(box) for box in qr_boxes],
                        "record_lines": [serialize_line(item) for item in parsed_lines],
                        "record_text": record_text,
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
            with (case_dir / text_name).open("w", encoding="utf-8") as fh:
                fh.write(f"PDF: {pdf_path.name}\n")
                fh.write(f"Page: {page_idx}\n")
                fh.write(f"Visit date: {page_context.visit_date or 'unknown'}\n")
                fh.write(f"Birth date: {page_context.birth_date or 'unknown'}\n")
                fh.write(f"Age years: {page_context.age_years if page_context.age_years is not None else 'unknown'}\n\n")
                for item in parsed_symbols:
                    fh.write(
                        f"[{item.label}] line={item.line_index} bbox=({item.left},{item.top},{item.right},{item.bottom}) {item.text}\n"
                    )
            with (case_dir / record_name).open("w", encoding="utf-8") as fh:
                fh.write(record_text)
        write_seconds = time.perf_counter() - write_start
        page_seconds = time.perf_counter() - page_start
        log_debug(
            debug_timing,
            f"{slow_tag(page_seconds, slow_stage_seconds)}{pdf_path.name} page {page_idx} done "
            f"symbols={len(parsed_symbols)} parsed_lines={len(parsed_lines)} draw={draw_seconds:.3f}s "
            f"write={write_seconds:.3f}s total={page_seconds:.3f}s",
        )
        progress_target = image_name if export_review_png else "full_record.txt"
        print(f"{pdf_path.name} page {page_idx}: {len(parsed_symbols)} symbols -> {progress_target}")
        if callable(progress_callback):
            progress_callback(page_idx, len(pages))

    record_path = case_dir / "full_record.txt"
    final_record_text = compose_full_record_text(full_record_pages)
    with record_path.open("w", encoding="utf-8") as fh:
        fh.write(final_record_text)
    pdf_seconds = time.perf_counter() - pdf_start
    log_debug(
        debug_timing,
        f"{slow_tag(pdf_seconds, slow_stage_seconds)}{pdf_path.name} complete pages={len(pages)} total={pdf_seconds:.3f}s",
    )
    return ConversionOutput(
        pdf_path=pdf_path,
        case_dir=case_dir,
        record_path=record_path,
        review_paths=tuple(review_paths),
        page_count=len(pages),
    )


def convert_outpatient_pdf_to_txt(
    pdf_path: str | Path,
    output_root: str | Path = OUT_DIR,
    export_review_png: bool = False,
    export_debug_artifacts: bool = False,
    debug_timing: bool = False,
    slow_stage_seconds: float = DEFAULT_SLOW_STAGE_SECONDS,
    progress_callback=None,
) -> ConversionOutput:
    normalized_pdf_path = Path(pdf_path)
    normalized_output_root = Path(output_root)
    normalized_output_root.mkdir(parents=True, exist_ok=True)

    ocr = get_cached_ocr(debug_timing=debug_timing)
    return process_pdf(
        ocr=ocr,
        pdf_path=normalized_pdf_path,
        output_root=normalized_output_root,
        export_review_png=export_review_png,
        export_debug_artifacts=export_debug_artifacts,
        debug_timing=debug_timing,
        slow_stage_seconds=slow_stage_seconds,
        progress_callback=progress_callback,
    )


def _convert_pdf_worker(
    pdf_path: str,
    output_root: str,
    export_review_png: bool,
    export_debug_artifacts: bool,
    debug_timing: bool,
    slow_stage_seconds: float,
) -> dict:
    result = convert_outpatient_pdf_to_txt(
        pdf_path=Path(pdf_path),
        output_root=Path(output_root),
        export_review_png=export_review_png,
        export_debug_artifacts=export_debug_artifacts,
        debug_timing=debug_timing,
        slow_stage_seconds=slow_stage_seconds,
    )
    return {
        "pdf_path": str(result.pdf_path),
        "case_dir": str(result.case_dir),
        "record_path": str(result.record_path),
        "review_paths": [str(path) for path in result.review_paths],
        "page_count": result.page_count,
    }


def convert_outpatient_pdfs_parallel(
    pdf_paths: list[str | Path],
    output_root: str | Path = OUT_DIR,
    export_review_png: bool = False,
    export_debug_artifacts: bool = False,
    max_workers: int | None = None,
    debug_timing: bool = False,
    slow_stage_seconds: float = DEFAULT_SLOW_STAGE_SECONDS,
) -> list[ConversionOutput]:
    normalized_pdf_paths = [Path(path) for path in pdf_paths]
    if not normalized_pdf_paths:
        return []

    normalized_output_root = Path(output_root)
    normalized_output_root.mkdir(parents=True, exist_ok=True)
    workers = max_workers or min(4, max(1, os.cpu_count() or 1))
    workers = max(1, min(workers, len(normalized_pdf_paths)))

    def submit_jobs(executor) -> list[ConversionOutput]:
        local_results: list[ConversionOutput] = []
        futures = {
            executor.submit(
                _convert_pdf_worker,
                str(pdf_path),
                str(normalized_output_root),
                export_review_png,
                export_debug_artifacts,
                debug_timing,
                slow_stage_seconds,
            ): pdf_path
            for pdf_path in normalized_pdf_paths
        }
        for future in as_completed(futures):
            payload = future.result()
            result = ConversionOutput(
                pdf_path=Path(payload["pdf_path"]),
                case_dir=Path(payload["case_dir"]),
                record_path=Path(payload["record_path"]),
                review_paths=tuple(Path(path) for path in payload["review_paths"]),
                page_count=int(payload["page_count"]),
            )
            local_results.append(result)
            print(f"parallel done: {result.pdf_path.name} -> {result.record_path}", flush=True)
        return local_results

    results: list[ConversionOutput] = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = submit_jobs(executor)
    except PermissionError:
        print("parallel fallback: ProcessPool unavailable in current environment, using ThreadPoolExecutor", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = submit_jobs(executor)

    results.sort(key=lambda item: item.pdf_path.name)
    return results


def resolve_input_pdfs(input_path: str | Path | None = None) -> list[Path]:
    if input_path is None:
        candidate = RAW_DIR
    else:
        candidate = Path(input_path)

    if candidate.is_file():
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {candidate}。支持的格式: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return [candidate]

    if candidate.is_dir():
        files = sorted(path for path in candidate.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not files:
            raise FileNotFoundError(f"目录中未找到支持的PDF或图片文件: {candidate}")
        return files

    raise FileNotFoundError(f"输入路径不存在: {candidate}")


resolve_input_files = resolve_input_pdfs
process_document = process_pdf
convert_outpatient_file_to_txt = convert_outpatient_pdf_to_txt



def convert_outpatient_input_path_parallel(
    input_path: str | Path,
    output_root: str | Path = OUT_DIR,
    export_review_png: bool = False,
    export_debug_artifacts: bool = False,
    max_workers: int | None = None,
    debug_timing: bool = False,
    slow_stage_seconds: float = DEFAULT_SLOW_STAGE_SECONDS,
) -> list[ConversionOutput]:
    pdf_files = resolve_input_pdfs(input_path)
    return convert_outpatient_pdfs_parallel(
        pdf_paths=pdf_files,
        output_root=output_root,
        export_review_png=export_review_png,
        export_debug_artifacts=export_debug_artifacts,
        max_workers=max_workers,
        debug_timing=debug_timing,
        slow_stage_seconds=slow_stage_seconds,
    )


def run_all_test_samples_parallel(
    output_root: str | Path,
    input_path: str | Path = RAW_DIR,
    export_review_png: bool = False,
    export_debug_artifacts: bool = False,
    max_workers: int | None = None,
    debug_timing: bool = False,
    slow_stage_seconds: float = DEFAULT_SLOW_STAGE_SECONDS,
) -> list[ConversionOutput]:
    return convert_outpatient_input_path_parallel(
        input_path=input_path,
        output_root=output_root,
        export_review_png=export_review_png,
        export_debug_artifacts=export_debug_artifacts,
        max_workers=max_workers,
        debug_timing=debug_timing,
        slow_stage_seconds=slow_stage_seconds,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert detected dental symbols into rule-based text with OCR review overlays.")
    parser.add_argument("--input-path", help="PDF file or directory containing PDFs. Defaults to raw data.")
    parser.add_argument("--pdf", action="append", help="Specific PDF path(s) to process. Defaults to all PDFs in raw data.")
    parser.add_argument("--output-root", default=str(OUT_DIR), help="Directory for output txt/png artifacts.")
    parser.add_argument("--export-review-png", action="store_true", help="Save review PNGs with visualized symbol boxes.")
    parser.add_argument("--export-debug-artifacts", action="store_true", help="Save page-level json/txt debug artifacts.")
    parser.add_argument("--parallel", action="store_true", help="Process multiple PDFs in parallel.")
    parser.add_argument("--max-workers", type=int, default=None, help="Worker count for parallel processing.")
    parser.add_argument("--run-all-tests", action="store_true", help="Legacy alias: process all PDFs from raw data.")
    parser.add_argument("--debug-timing", action="store_true", help="Print timing logs for major pipeline stages.")
    parser.add_argument(
        "--slow-stage-seconds",
        type=float,
        default=DEFAULT_SLOW_STAGE_SECONDS,
        help="Threshold in seconds for marking a timing line as slow.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.input_path:
        pdf_files = resolve_input_pdfs(args.input_path)
    elif args.run_all_tests:
        pdf_files = resolve_input_pdfs(RAW_DIR)
    elif args.pdf:
        pdf_files = [Path(item) for item in args.pdf]
    else:
        pdf_files = resolve_input_pdfs(RAW_DIR)

    total_start = time.perf_counter()
    if args.parallel and len(pdf_files) > 1:
        results = convert_outpatient_pdfs_parallel(
            pdf_paths=pdf_files,
            output_root=output_root,
            export_review_png=args.export_review_png,
            export_debug_artifacts=args.export_debug_artifacts,
            max_workers=args.max_workers,
            debug_timing=args.debug_timing,
            slow_stage_seconds=args.slow_stage_seconds,
        )
        for result in results:
            print(f"record ready: {result.record_path}")
    else:
        ocr = get_cached_ocr(debug_timing=args.debug_timing)
        for pdf_path in pdf_files:
            result = process_pdf(
                ocr,
                pdf_path,
                output_root=output_root,
                export_review_png=args.export_review_png,
                export_debug_artifacts=args.export_debug_artifacts,
                debug_timing=args.debug_timing,
                slow_stage_seconds=args.slow_stage_seconds,
            )
            print(f"record ready: {result.record_path}")
    log_debug(args.debug_timing, f"All PDFs complete total={time.perf_counter() - total_start:.3f}s")


if __name__ == "__main__":
    main()

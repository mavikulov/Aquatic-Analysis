from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MaskOptions:
    method: str = "NDVI-порог"
    ndvi_threshold: float = 0.20
    exg_auto: bool = True
    exg_threshold: float = 0.10
    smooth: int = 5
    min_area: int = 200
    fill_holes: bool = False

    def key(self) -> tuple:
        return tuple(sorted(self.__dict__.items()))


def excess_green(bands) -> np.ndarray:
    total = bands.R + bands.G + bands.B + 1e-6
    r, g, b = bands.R / total, bands.G / total, bands.B / total
    return 2.0 * g - r - b


def otsu_threshold(x: np.ndarray) -> float:
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return float(np.median(x))

    scaled = np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return lo + (t / 255.0) * (hi - lo)


def ndvi(bands) -> np.ndarray:
    return (bands.NIR - bands.R) / np.clip(bands.NIR + bands.R, 1e-6, None)


def postprocess_mask(
    mask: np.ndarray, smooth: int, min_area: int, fill: bool
) -> np.ndarray:
    mask = mask.astype(np.uint8)
    if smooth >= 3:
        k = np.ones((smooth, smooth), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        mask = keep[labels].astype(np.uint8)

    if fill:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)

    return mask.astype(bool)


def segment(bands, opts: MaskOptions) -> np.ndarray:
    if opts.method == "Избыток зелёного (ExG)":
        exg = excess_green(bands)
        thr = otsu_threshold(exg) if opts.exg_auto else opts.exg_threshold
        raw = exg > thr
    else:  # NDVI-порог
        raw = ndvi(bands) > opts.ndvi_threshold

    return postprocess_mask(raw, opts.smooth, opts.min_area, opts.fill_holes)

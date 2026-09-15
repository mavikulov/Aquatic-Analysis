from dataclasses import dataclass, field

import cv2
import numpy as np

from veg.segmentation import otsu_threshold, postprocess_mask

THRESHOLD_METHODS = ("Оцу (авто)", "Ручной порог", "Адаптивный (локальный)")


@dataclass
class CoverOptions:
    flatfield: bool = False
    flatfield_sigma_frac: float = 0.12
    method: str = "Оцу (авто)"
    threshold: float = 0.5
    adaptive_block_frac: float = 0.12
    adaptive_offset: float = 0.02
    invert: bool = False
    exclude_bright: bool = False
    bright_percentile: float = 99.5
    smooth: int = 5
    min_area: int = 200
    fill_holes: bool = False

    def key(self) -> tuple:
        return tuple(sorted(self.__dict__.items()))


@dataclass
class CoverResult:
    mask: np.ndarray
    roi: np.ndarray
    percent: float
    class_px: int
    roi_px: int
    extra: dict = field(default_factory=dict)


def round_odd(n: int) -> int:
    n = int(round(n))
    return n + 1 if n % 2 == 0 else max(3, n)


def flatfield_correct(band: np.ndarray, sigma_frac: float) -> np.ndarray:
    sigma = max(3.0, sigma_frac * max(band.shape))
    k = round_odd(sigma * 3)
    bg = cv2.GaussianBlur(band.astype(np.float32), (k, k), sigma)
    bg = np.clip(bg, 1e-3, None)
    out = band / bg
    return np.clip(out / max(float(np.percentile(out, 99)), 1e-6), 0.0, 1.0)


def _threshold_mask(band: np.ndarray, opts: CoverOptions) -> np.ndarray:
    if opts.method == "Ручной порог":
        raw = band > opts.threshold
    elif opts.method == "Адаптивный (локальный)":
        block = round_odd(opts.adaptive_block_frac * max(band.shape))
        u8 = np.clip(band * 255.0, 0, 255).astype(np.uint8)
        c = opts.adaptive_offset * 255.0
        raw = (
            cv2.adaptiveThreshold(
                u8,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                -c,
            )
            > 0
        )
    else:
        raw = band > otsu_threshold(band)
    return ~raw if opts.invert else raw


def cover_mask(
    band: np.ndarray, opts: CoverOptions, roi: np.ndarray | None = None
) -> np.ndarray:
    work = (
        flatfield_correct(band, opts.flatfield_sigma_frac) if opts.flatfield else band
    )
    raw = _threshold_mask(work, opts)

    if opts.exclude_bright:
        raw &= band <= np.percentile(band, opts.bright_percentile)
    if roi is not None:
        raw &= roi

    mask = postprocess_mask(raw, opts.smooth, opts.min_area, opts.fill_holes)
    if roi is not None:
        mask &= roi
    return mask


def resolve_roi(shape: tuple[int, int], roi: np.ndarray | None) -> np.ndarray:
    if roi is None:
        return np.ones(shape, dtype=bool)
    return roi.astype(bool)


def rect_roi(
    shape: tuple[int, int], x0: float, x1: float, y0: float, y1: float
) -> np.ndarray:
    h, w = shape
    m = np.zeros(shape, dtype=bool)
    xa, xb = sorted((int(x0 * w), int(x1 * w)))
    ya, yb = sorted((int(y0 * h), int(y1 * h)))
    m[ya:yb, xa:xb] = True
    return m


def ground_cover(
    band: np.ndarray, opts: CoverOptions, roi: np.ndarray | None = None
) -> CoverResult:
    roi_m = resolve_roi(band.shape, roi)
    mask = cover_mask(band, opts, roi_m)
    roi_px = int(roi_m.sum())
    class_px = int(mask.sum())
    pct = 100.0 * class_px / roi_px if roi_px else 0.0
    return CoverResult(mask, roi_m, pct, class_px, roi_px)


def excess_green(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    total = f.sum(axis=2) + 1e-6
    r, g, b = (f[:, :, i] / total for i in range(3))
    return 2.0 * g - r - b


def lily_mask(
    rgb: np.ndarray,
    opts: CoverOptions,
    roi: np.ndarray | None = None,
    exg_threshold: float | None = None,
) -> np.ndarray:
    exg = excess_green(rgb)
    thr = otsu_threshold(exg) if exg_threshold is None else float(exg_threshold)
    raw = exg > thr
    if opts.exclude_bright:
        raw &= rgb.max(axis=2) <= np.percentile(rgb.max(axis=2), opts.bright_percentile)

    if roi is not None:
        raw &= roi

    mask = postprocess_mask(raw, opts.smooth, opts.min_area, opts.fill_holes)
    return mask & roi if roi is not None else mask


def water_cover(
    rgb: np.ndarray,
    nir: np.ndarray,
    lily_opts: CoverOptions,
    submerged_opts: CoverOptions,
    roi: np.ndarray | None = None,
    lily_exg_threshold: float | None = None,
    lily_dilate_px: int = 25,
) -> tuple[CoverResult, CoverResult]:
    roi_m = resolve_roi(nir.shape, roi)
    roi_px = int(roi_m.sum())
    lilies = lily_mask(rgb, lily_opts, roi_m, lily_exg_threshold)
    submerged_raw = cover_mask(nir, submerged_opts, roi_m)

    if lily_dilate_px > 0:
        k = np.ones((round_odd(lily_dilate_px), round_odd(lily_dilate_px)), np.uint8)
        lily_guard = cv2.dilate(lilies.astype(np.uint8), k) > 0
    else:
        lily_guard = lilies

    submerged = submerged_raw & ~lily_guard & roi_m

    def res(m: np.ndarray) -> CoverResult:
        px = int(m.sum())
        return CoverResult(m, roi_m, 100.0 * px / roi_px if roi_px else 0.0, px, roi_px)

    lily_res = res(lilies)
    sub_res = res(submerged)
    sub_res.extra["submerged_before_guard"] = submerged_raw
    return lily_res, sub_res


def overlay(
    base_rgb: np.ndarray,
    layers: list[tuple[np.ndarray, tuple[int, int, int]]],
    roi: np.ndarray | None = None,
    dim_outside: float = 0.35,
) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    if roi is not None:
        out[~roi] *= dim_outside
    for mask, color in layers:
        col = np.array(color, np.float32)
        out[mask] = 0.45 * out[mask] + 0.55 * col

    return np.clip(out, 0, 255).astype(np.uint8)

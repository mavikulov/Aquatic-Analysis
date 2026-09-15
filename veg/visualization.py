import matplotlib
import numpy as np
import pandas as pd


def mask_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = arr[mask]
    return values[np.isfinite(values)]


def colorize(
    arr: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    mask: np.ndarray | None = None,
    background: str = "Затемнить фон",
) -> np.ndarray:
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0)
    rgb = (matplotlib.colormaps[cmap](norm)[:, :, :3] * 255).astype(np.uint8)

    if mask is not None and background != "Показать всё":
        factor = 0.0 if background == "Скрыть фон" else 0.18
        rgb[~mask] = (rgb[~mask] * factor).astype(np.uint8)

    return rgb


def colorbar_strip(cmap: str, width: int = 480, height: int = 24) -> np.ndarray:
    grad = np.linspace(0.0, 1.0, width)
    row = (matplotlib.colormaps[cmap](grad)[:, :3] * 255).astype(np.uint8)
    return np.tile(row, (height, 1, 1))


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[~mask] = (out[~mask] * 0.30).astype(np.uint8)
    veg = out[mask].astype(np.int16)
    veg[:, 1] = np.minimum(255, veg[:, 1] + 45)
    out[mask] = veg.astype(np.uint8)
    return out


def histogram_df(arr: np.ndarray, mask: np.ndarray, bins: int = 50) -> pd.DataFrame:
    masked_values = mask_values(arr, mask)
    if masked_values.size == 0:
        return pd.DataFrame({"value": [], "share": []})

    low, high = np.percentile(masked_values, [0.5, 99.5])
    if high <= low:
        low, high = float(masked_values.min()), float(masked_values.max()) + 1e-6

    counts, edges = np.histogram(masked_values, bins=bins, range=(low, high))
    centers = (edges[:-1] + edges[1:]) / 2.0
    share = counts / max(counts.sum(), 1) * 100.0
    return pd.DataFrame({"value": np.round(centers, 4), "share": np.round(share, 3)})


def distribution_long_df(
    results: dict[str, np.ndarray], mask: np.ndarray, per_index: int = 6000
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for name, arr in results.items():
        v = mask_values(arr, mask)
        if v.size == 0:
            continue
        if v.size > per_index:
            v = rng.choice(v, per_index, replace=False)
        frames.append(pd.DataFrame({"index": name, "value": v}))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

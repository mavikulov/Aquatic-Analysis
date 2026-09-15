import io
import zipfile

import cv2
import numpy as np
import pandas as pd
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

NODATA = -9999.0


def _geotiff_bytes(arr: np.ndarray, dtype: str, nodata) -> bytes:
    h, w = arr.shape
    with MemoryFile() as mf:
        with mf.open(
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=dtype,
            crs="EPSG:3857",
            transform=from_origin(0, h, 1, 1),
            nodata=nodata,
            compress="lzw",
            predictor=3 if dtype == "float32" else 2,
        ) as dst:
            dst.write(arr.astype(dtype), 1)
        return mf.read()


def index_geotiff(arr: np.ndarray) -> bytes:
    return _geotiff_bytes(
        np.nan_to_num(arr, nan=NODATA).astype(np.float32), "float32", NODATA
    )


def mask_geotiff(mask: np.ndarray) -> bytes:
    return _geotiff_bytes(mask.astype(np.uint8), "uint8", 0)


def preview_png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Не удалось закодировать PNG.")
    return buf.tobytes()


def build_cover_zip(
    overlay_rgb: np.ndarray,
    layers: dict[str, np.ndarray],
    roi: np.ndarray,
    summary: pd.DataFrame,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("overlay.png", preview_png(overlay_rgb))
        z.writestr("roi_mask.tif", mask_geotiff(roi.astype(bool)))
        for name, mask in layers.items():
            z.writestr(f"masks/{name}.tif", mask_geotiff(mask.astype(bool)))
            z.writestr(
                f"masks/{name}.png",
                preview_png(
                    np.repeat((mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
                ),
            )
        z.writestr("projective_cover.csv", summary.to_csv(index=False))

    return buf.getvalue()


def build_zip(
    results: dict[str, np.ndarray],
    mask: np.ndarray,
    stats: pd.DataFrame,
    previews: dict[str, np.ndarray] | None = None,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, arr in results.items():
            z.writestr(f"indices/{name.lower()}.tif", index_geotiff(arr))
        if previews:
            for name, rgb in previews.items():
                z.writestr(f"previews/{name.lower()}.png", preview_png(rgb))
        z.writestr("vegetation_mask.tif", mask_geotiff(mask))
        z.writestr("statistics.csv", stats.to_csv(index=False))

    return buf.getvalue()

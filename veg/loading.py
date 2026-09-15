from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Bands:
    rgb: np.ndarray
    R: np.ndarray
    G: np.ndarray
    B: np.ndarray
    NIR: np.ndarray
    nir_preview: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.R.shape


def decode_image(data: bytes, flags: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, flags)
    if img is None:
        raise ValueError("Не удалось декодировать изображение.")

    return img


def load_color(data: bytes) -> np.ndarray:
    img = decode_image(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_band(data: bytes) -> np.ndarray:
    img = decode_image(data, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = img[:, :, :3].mean(axis=2)

    info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
    scale = info.max if info else float(img.max() or 1.0)
    return (img.astype(np.float32) / scale).clip(0.0, 1.0)


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if img.shape[1] == size[0] and img.shape[0] == size[1]:
        return img

    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def target_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max_side and max(width, height) > max_side:
        s = max_side / max(width, height)
        return round(width * s), round(height * s)
    return width, height


def prepare_from_rgb_nir(
    rgb_bytes: bytes,
    nir_bytes: bytes,
) -> Bands:
    rgb = load_color(rgb_bytes)
    nir_src = load_color(nir_bytes)
    h, w = rgb.shape[:2]
    rgb = resize(rgb, (w, h))
    nir_src = resize(nir_src, (w, h))
    idx = 0
    nir = nir_src[:, :, idx] if idx is not None else nir_src.mean(axis=2)
    rgb_f = rgb.astype(np.float32) / 255.0
    return Bands(
        rgb=rgb,
        R=rgb_f[:, :, 0],
        G=rgb_f[:, :, 1],
        B=rgb_f[:, :, 2],
        NIR=(nir.astype(np.float32) / 255.0),
        nir_preview=nir_src,
    )


def prepare_single_band(
    data: bytes, channel: str = "Среднее", max_side: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    src = load_color(data)
    h, w = src.shape[:2]
    tw, th = target_size(w, h, max_side)
    src = resize(src, (tw, th))
    idx = 0
    band = src[:, :, idx] if idx is not None else src.mean(axis=2)
    band = (band.astype(np.float32) / 255.0).clip(0.0, 1.0)
    return band, src


def load_roi_mask(data: bytes, size: tuple[int, int]) -> np.ndarray:
    img = decode_image(data, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = img[:, :, :3].mean(axis=2)

    img = resize(img.astype(np.float32), size)
    return img > (0.5 * float(img.max() or 1.0))


def prepare_from_channels(
    r_bytes: bytes,
    g_bytes: bytes,
    b_bytes: bytes,
    nir_bytes: bytes,
    max_side: int = 0,
) -> Bands:
    R = load_band(r_bytes)
    G = load_band(g_bytes)
    B = load_band(b_bytes)
    NIR = load_band(nir_bytes)

    h, w = R.shape
    tw, th = target_size(w, h, max_side)
    R, G, B, NIR = (resize(x, (tw, th)) for x in (R, G, B, NIR))

    rgb = (np.dstack([R, G, B]) * 255).clip(0, 255).astype(np.uint8)
    nir_preview = (NIR * 255).clip(0, 255).astype(np.uint8)
    return Bands(rgb=rgb, R=R, G=G, B=B, NIR=NIR, nir_preview=nir_preview)

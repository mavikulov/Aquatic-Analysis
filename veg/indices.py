from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_EPS = 1e-6


@dataclass(frozen=True)
class IndexSpec:
    name: str
    func: Callable[..., np.ndarray]
    needs_nir: bool
    cmap: str
    vmin: float
    vmax: float
    formula: str
    about: str


def safe_division(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    safe = np.where(np.abs(den) < _EPS, np.copysign(_EPS, den) + (den == 0) * _EPS, den)
    return num / safe


def NDVI(R: np.ndarray, G: np.ndarray, B: np.ndarray, NIR: np.ndarray, L: float = 0.5):
    return safe_division(NIR - R, NIR + R)


def SAVI(R: np.ndarray, G: np.ndarray, B: np.ndarray, NIR: np.ndarray, L: float = 0.5):
    return (1.0 + L) * safe_division(NIR - R, NIR + R + L)


def NDAVI(R: np.ndarray, G: np.ndarray, B: np.ndarray, NIR: np.ndarray, L: float = 0.5):
    return safe_division(NIR - B, NIR + B)


def WAVI(R: np.ndarray, G: np.ndarray, B: np.ndarray, NIR: np.ndarray, L: float = 0.5):
    return (1.0 + L) * safe_division(NIR - B, NIR + B + L)


def EVI(R: np.ndarray, G: np.ndarray, B: np.ndarray, NIR: np.ndarray, L: float = 0.5):
    return 2.5 * safe_division(NIR - R, NIR + 6.0 * R - 7.5 * B + 1.0)


def NDWI(R: np.ndarray, G: np.ndarray, B: np.ndarray, NIR: np.ndarray, L: float = 0.5):
    return safe_division(G - NIR, G + NIR)


def VARI(R: np.ndarray, G: np.ndarray, B: np.ndarray, NIR: np.ndarray, L: float = 0.5):
    return safe_division(G - R, G + R - B)


INDEX_SPECS: dict[str, IndexSpec] = {
    "NDVI": IndexSpec(
        "NDVI",
        NDVI,
        True,
        "RdYlGn",
        -1.0,
        1.0,
        r"NDVI = \dfrac{NIR - R}{NIR + R}",
        "Базовый индекс зелёной массы и здоровья растений.",
    ),
    "SAVI": IndexSpec(
        "SAVI",
        SAVI,
        True,
        "RdYlGn",
        -1.2,
        1.2,
        r"SAVI = (1 + L)\,\dfrac{NIR - R}{NIR + R + L}",
        "NDVI с поправкой L на яркость фона (почва). Меньше «завышает» на редком покрове.",
    ),
    "EVI": IndexSpec(
        "EVI",
        EVI,
        True,
        "RdYlGn",
        -1.0,
        1.0,
        r"EVI = 2.5\,\dfrac{NIR - R}{NIR + 6R - 7.5B + 1}",
        "Улучшенный индекс: не насыщается на плотной листве, устойчив к атмосфере.",
    ),
    "NDAVI": IndexSpec(
        "NDAVI",
        NDAVI,
        True,
        "YlGn",
        -1.0,
        1.0,
        r"NDAVI = \dfrac{NIR - B}{NIR + B}",
        "Индекс водной растительности: подавляет фон-воду через синий канал.",
    ),
    "WAVI": IndexSpec(
        "WAVI",
        WAVI,
        True,
        "YlGn",
        -1.2,
        1.2,
        r"WAVI = (1 + L)\,\dfrac{NIR - B}{NIR + B + L}",
        "NDAVI с поправкой L на фон-воду.",
    ),
    "NDWI": IndexSpec(
        "NDWI",
        NDWI,
        True,
        "RdYlBu",
        -1.0,
        1.0,
        r"NDWI = \dfrac{G - NIR}{G + NIR}",
        "Индекс воды. Для растительности значения отрицательные — полезен как контроль.",
    ),
    "VARI": IndexSpec(
        "VARI",
        VARI,
        False,
        "RdYlGn",
        -1.0,
        1.0,
        r"VARI = \dfrac{G - R}{G + R - B}",
        "Работает без NIR, только по RGB.",
    ),
}


def compute_index(name: str, bands, L: float) -> np.ndarray:
    spec = INDEX_SPECS[name]
    arr = spec.func(bands.R, bands.G, bands.B, bands.NIR, L).astype(np.float32)
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=2.0, neginf=-2.0), -2.0, 2.0)

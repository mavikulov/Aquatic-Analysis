import numpy as np
import pandas as pd


def index_stats(name: str, arr: np.ndarray, mask: np.ndarray) -> dict:
    v = arr[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"Индекс": name, "Пикселей": 0}
    return {
        "Индекс": name,
        "Пикселей": int(v.size),
        "Среднее": round(float(v.mean()), 4),
        "Медиана": round(float(np.median(v)), 4),
        "Ст. откл.": round(float(v.std()), 4),
        "Мин": round(float(v.min()), 4),
        "P10": round(float(np.percentile(v, 10)), 4),
        "P90": round(float(np.percentile(v, 90)), 4),
        "Макс": round(float(v.max()), 4),
    }


def get_stats_dataframe(
    results: dict[str, np.ndarray], mask: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame([index_stats(n, a, mask) for n, a in results.items()])

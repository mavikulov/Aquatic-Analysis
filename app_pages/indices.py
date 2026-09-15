from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from veg import loading
from veg.export import build_zip, index_geotiff, mask_geotiff, preview_png
from veg.indices import INDEX_SPECS, compute_index
from veg.segmentation import MaskOptions, segment
from veg.stats import get_stats_dataframe
from veg.visualization import (
    colorbar_strip,
    colorize,
    distribution_long_df,
    histogram_df,
    overlay_mask,
)


@st.cache_data(show_spinner=False, max_entries=6)
def prepare_rbg_nir(rgb: bytes, nir: bytes) -> loading.Bands:
    return loading.prepare_from_rgb_nir(rgb, nir)


@st.cache_data(show_spinner=False, max_entries=6)
def prepare_channels(r: bytes, g: bytes, b: bytes, nir: bytes) -> loading.Bands:
    return loading.prepare_from_channels(r, g, b, nir)


@st.cache_data(show_spinner=False, max_entries=12)
def _segment(bands: loading.Bands, opts_key: tuple) -> np.ndarray:
    return segment(bands, MaskOptions(**dict(opts_key)))


@st.cache_data(show_spinner=False, max_entries=6)
def compute(bands: loading.Bands, names: tuple, L: float) -> dict:
    return {name: compute_index(name, bands, L) for name in names}


class Mode(Enum):
    RGB_NIR = "RGB-снимок + NIR-снимок"
    CHANNELS = "Отдельные каналы R, G, B, NIR"


class BackgroundMode(Enum):
    HIDE = "Скрыть фон"
    SHOW = "Показать всё"
    DARK = "Затемнить фон"


class SegmentationMethod:
    EXG = "Избыток зелёного (ExG)"
    NDVI_THRESHOLD = "NDVI-порог"


@dataclass
class IndicesPageState:
    mode: str = "RGB-снимок + NIR-снимок"
    files: dict = field(default_factory=dict)
    bands: loading.Bands | None = None
    sig: tuple | None = None
    mask: np.ndarray | None = None
    mask_options: MaskOptions | None = None
    selected: list[str] = field(default_factory=list)
    L_const: float = 0.5
    background: Literal["Затемнить фон"] = BackgroundMode.DARK.value
    results: dict[str, np.ndarray] = field(default_factory=dict)
    stats_df: pd.DataFrame | None = None
    previews: dict[str, np.ndarray] = field(default_factory=dict)


class IndicesPage:
    def __init__(self):
        self.state: IndicesPageState = IndicesPageState()

    def __render_sidebar(self):
        state = self.state
        st.sidebar.header("Данные")
        state.mode = st.sidebar.radio(
            "Как заданы снимки",
            ["RGB-снимок + NIR-снимок", "Отдельные каналы R, G, B, NIR"],
        )
        state.files = {}

        if state.mode == "RGB-снимок + NIR-снимок":
            state.files["rgb"] = st.sidebar.file_uploader(
                "RGB-снимок", type=["jpg", "jpeg", "png", "tif", "tiff"]
            )
            state.files["nir"] = st.sidebar.file_uploader(
                "NIR-снимок", type=["jpg", "jpeg", "png", "tif", "tiff"]
            )
        else:
            column_1, column_2 = st.sidebar.columns(2)
            state.files["r"] = column_1.file_uploader(
                "R", type=["png", "jpg", "jpeg", "tif", "tiff"]
            )
            state.files["g"] = column_2.file_uploader(
                "G", type=["png", "jpg", "jpeg", "tif", "tiff"]
            )
            state.files["b"] = column_1.file_uploader(
                "B", type=["png", "jpg", "jpeg", "tif", "tiff"]
            )
            state.files["nir"] = column_2.file_uploader(
                "NIR", type=["png", "jpg", "jpeg", "tif", "tiff"]
            )

    def __load_bands(self) -> bool:
        state = self.state
        if not all(state.files.values()):
            st.info("Загрузите снимки в боковой панели.", icon=":material/upload_file:")
            with st.expander("RGB-снимок", icon=":material/photo_camera:"):
                st.markdown("Фотография в RGB.")
            with st.expander("NIR-снимок", icon=":material/blur_on:"):
                st.markdown("Снимок в инфракрасном диапазоне.")
            with st.expander("Отдельные каналы R, G, B, NIR", icon=":material/layers:"):
                st.markdown("Четыре одноканальных файла R, G, B, NIR")
            return False

        try:
            raw_data = {key: value.getvalue() for key, value in state.files.items()}
            if state.mode == "RGB-снимок + NIR-снимок":
                state.sig = (
                    state.mode,
                    raw_data["rgb"][:64],
                    raw_data["nir"][:64],
                    len(raw_data["rgb"]),
                )
                state.bands = prepare_rbg_nir(raw_data["rgb"], raw_data["nir"])
            else:
                state.sig = (state.mode,) + tuple(
                    raw_data[k][:48] for k in ("r", "g", "b", "nir")
                )
                state.bands = prepare_channels(
                    raw_data["r"], raw_data["g"], raw_data["b"], raw_data["nir"]
                )

        except Exception as exc:
            st.error(f"Не удалось подготовить каналы: {exc}", icon=":material/error:")
            return False
        return True

    def __render_input_section(self) -> None:
        state = self.state
        H, W = state.bands.shape
        st.header("1. Входные данные", divider="gray")
        column_1, column_2 = st.columns(2)
        column_1.image(state.bands.rgb, caption=f"RGB: {W}×{H}", width="stretch")
        nir_cap = (
            "NIR" if state.mode == "Отдельные каналы R, G, B, NIR" else "NIR-снимок"
        )
        column_2.image(state.bands.nir_preview, caption=nir_cap, width="stretch")

        with st.container(border=True):
            column_1, column_2, column_3, column_4 = st.columns(4)
            column_1.metric("Размер", f"{W} × {H}")
            column_2.metric("Средний R", f"{state.bands.R.mean():.3f}")
            column_3.metric("Средний NIR", f"{state.bands.NIR.mean():.3f}")
            contrast = float((state.bands.NIR - state.bands.R).mean())
            column_4.metric(
                "Средн. NIR − R",
                f"{contrast:+.3f}",
            )

    def __render_segmentation_section(self) -> None:
        state = self.state
        st.header("2 · Выделение растительности", divider="gray")
        st.caption("Маска задаёт, по каким пикселям считается статистика индексов.")
        method = (
            st.segmented_control(
                "Метод", ["NDVI-порог", "Избыток зелёного (ExG)"], default="NDVI-порог"
            )
            or "NDVI-порог"
        )

        options = MaskOptions(method=method)
        column_1, column_2 = st.columns([2, 1])

        with column_1:
            if method == "NDVI-порог":
                options.ndvi_threshold = st.slider(
                    "Порог NDVI",
                    -0.2,
                    0.9,
                    0.20,
                    0.01,
                    help="Пиксель считается растением, если NDVI выше порога.",
                )
            else:
                options.exg_auto = st.toggle(
                    "Порог автоматически (метод Оцу)", value=True
                )
                options.exg_threshold = st.slider(
                    "Порог ExG", -0.2, 0.6, 0.10, 0.01, disabled=options.exg_auto
                )

        with column_2, st.popover("Очистка маски", icon=":material/tune:"):
            options.smooth = st.slider("Сглаживание (морфология)", 0, 20, 5, 1)
            options.min_area = st.slider(
                "Удалять пятна мельче, пикс.", 0, 5000, 200, 50
            )
            options.fill_holes = st.toggle("Заливать дырки в листьях", value=False)

        with st.spinner("Считаю маску…"):
            state.mask = _segment(state.bands, options.key())

        state.mask_options = options
        coverage = float(state.mask.mean() * 100)

        first_image, second_image = st.columns(2)
        first_image.image(
            overlay_mask(state.bands.rgb, state.mask),
            caption="Растительность подсвечена",
            width="stretch",
        )
        second_image.image(
            (state.mask * 255).astype(np.uint8),
            caption="Бинарная маска",
            width="stretch",
        )

        first_metric_stats, second_metric_stats = st.columns(2)
        first_metric_stats.metric("Покрытие растительностью", f"{coverage:.1f} %")
        second_metric_stats.metric(
            "Пикселей растительности", f"{int(state.mask.sum()):,}".replace(",", " ")
        )

    def __render_indices_section(self) -> bool:
        state = self.state
        st.header("3 · Вегетационные индексы", divider="gray")

        sel1, sel2 = st.columns([3, 1])
        state.selected = sel1.multiselect(
            "Индексы", list(INDEX_SPECS), default=["NDVI", "SAVI", "EVI", "VARI"]
        )
        state.L_const = sel2.slider("L (SAVI/WAVI)", 0.0, 1.0, 0.5, 0.05)

        state.background = st.radio(
            "Фон на картах индексов",
            [mode.value for mode in BackgroundMode],
            horizontal=True,
        )

        if not state.selected:
            st.info("Выберите хотя бы один индекс.", icon=":material/checklist:")
            return False

        with st.expander("Как считаются индексы", icon=":material/functions:"):
            for name in state.selected:
                spec = INDEX_SPECS[name]
                st.markdown(f"**{name}** — {spec.about}")
                st.latex(spec.formula)

        with st.spinner("Считаю индексы…"):
            state.results = compute(state.bands, tuple(state.selected), state.L_const)

        state.stats_df = get_stats_dataframe(state.results, state.mask)
        self.__render_index_maps()
        self.__render_distributions()

        st.subheader("Сводная статистика")
        st.dataframe(state.stats_df, width="stretch", hide_index=True)
        return True

    def __render_index_maps(self):
        state = self.state
        st.subheader("Карты индексов")
        grid = st.columns(2)
        state.previews = {}
        for i, name in enumerate(state.selected):
            spec = INDEX_SPECS[name]
            img = colorize(
                state.results[name],
                spec.cmap,
                spec.vmin,
                spec.vmax,
                state.mask,
                state.background,
            )
            state.previews[name] = img
            with grid[i % 2].container(border=True):
                st.markdown(f"**{name}**  ·  шкала [{spec.vmin}, {spec.vmax}]")
                st.image(img, width="stretch")
                st.image(colorbar_strip(spec.cmap), width="stretch")
                row = state.stats_df.loc[state.stats_df["Индекс"] == name].iloc[0]
                if row["Пикселей"]:
                    st.caption(
                        f"среднее **{row['Среднее']}** · медиана {row['Медиана']} · "
                        f"P10–P90 [{row['P10']}, {row['P90']}]"
                    )

    def __render_distributions(self):
        state = self.state
        st.subheader("Графики распределения")
        st.caption("По пикселям растительности. Красная линия — среднее.")
        hist_cols = st.columns(2)

        for i, name in enumerate(state.selected):
            hdf = histogram_df(state.results[name], state.mask)
            if hdf.empty:
                continue

            mean_v = float(
                state.stats_df.loc[state.stats_df["Индекс"] == name, "Среднее"].iloc[0]
            )

            bars = (
                alt.Chart(hdf)
                .mark_bar(color="#4c9a2a")
                .encode(
                    x=alt.X("value", type="quantitative", title=name),
                    y=alt.Y("share", type="quantitative", title="доля, %"),
                    tooltip=[
                        alt.Tooltip("value", title="значение"),
                        alt.Tooltip("share", title="доля, %"),
                    ],
                )
            )

            rule = (
                alt.Chart(pd.DataFrame({"m": [mean_v]}))
                .mark_rule(color="#d62728", size=2)
                .encode(x=alt.X("m", type="quantitative"))
            )

            hist_cols[i % 2].altair_chart(
                (bars + rule).properties(height=200), width="stretch"
            )

        long_df = distribution_long_df(state.results, state.mask)
        if not long_df.empty and len(state.selected) > 1:
            st.markdown("**Сравнение индексов**")
            box = (
                alt.Chart(long_df)
                .mark_boxplot(extent="min-max")
                .encode(
                    x=alt.X("index", type="nominal", title=None),
                    y=alt.Y("value", type="quantitative", title="значение"),
                    color=alt.Color("index", type="nominal", legend=None),
                )
            )
            st.altair_chart(box.properties(height=280), width="stretch")

    def __render_export_section(self):
        state = self.state
        st.header("4 · Экспорт без потерь", divider="gray")
        export_sig = (
            state.sig,
            tuple(state.selected),
            state.L_const,
            state.mask_options.key(),
            state.background,
        )

        if st.button("Подготовить файлы", icon=":material/build:"):
            with st.spinner("Кодирую GeoTIFF…"):
                st.session_state["export"] = {
                    "sig": export_sig,
                    "zip": build_zip(
                        state.results, state.mask, state.stats_df, state.previews
                    ),
                    "mask": mask_geotiff(state.mask),
                    "tif": {n: index_geotiff(state.results[n]) for n in state.selected},
                    "png": {n: preview_png(state.previews[n]) for n in state.selected},
                }

        bundle = st.session_state.get("export")
        if not bundle or bundle["sig"] != export_sig:
            st.caption(
                "Нажмите «Подготовить файлы», чтобы собрать GeoTIFF для скачивания."
            )
            return

        e1, e2 = st.columns(2)

        e1.download_button(
            "Скачать всё (ZIP)",
            bundle["zip"],
            "vegetation_indices.zip",
            "application/zip",
            type="primary",
            icon=":material/folder_zip:",
            width="stretch",
        )
        e2.download_button(
            "Маска (GeoTIFF)",
            bundle["mask"],
            "vegetation_mask.tif",
            "image/tiff",
            icon=":material/layers:",
            width="stretch",
        )

        st.markdown("**Отдельные индексы:**")
        dl = st.columns(3)
        for i, name in enumerate(state.selected):
            with dl[i % 3]:
                st.download_button(
                    f"{name}.tif",
                    bundle["tif"][name],
                    f"{name.lower()}.tif",
                    "image/tiff",
                    width="stretch",
                )
                st.download_button(
                    f"{name} · PNG",
                    bundle["png"][name],
                    f"{name.lower()}.png",
                    "image/png",
                    width="stretch",
                )

    def run(self) -> None:
        self.__render_sidebar()
        st.title("Вегетационные индексы")
        st.caption(
            "Данные → выделение растительности → индексы → распределения → "
            "экспорт без потерь."
        )
        if not self.__load_bands():
            return

        self.__render_input_section()
        self.__render_segmentation_section()

        if not self.__render_indices_section():
            return

        self.__render_export_section()


IndicesPage().run()

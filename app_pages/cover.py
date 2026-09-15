import numpy as np
import pandas as pd
import streamlit as st
from veg import loading
from veg.cover import (
    CoverOptions,
    ground_cover,
    overlay,
    rect_roi,
    water_cover,
)
from veg.export import build_cover_zip

CHANNELS = ["Красный канал", "Зелёный канал", "Синий канал", "Среднее"]
TASK_GROUND = "Наземное сообщество · 2 класса"
TASK_WATER = "Водное сообщество · 3 класса"
THRESHOLD_METHODS = ("Оцу (авто)", "Ручной порог", "Адаптивный (локальный)")

LILY_RGB = (80, 200, 120)
SUBMERGED_RGB = (250, 170, 40)
VEG_RGB = (90, 210, 100)


@st.cache_data(show_spinner=False, max_entries=6)
def load_band(data: bytes, channel: str, max_side: int):
    return loading.prepare_single_band(data, channel, max_side)


@st.cache_data(show_spinner=False, max_entries=6)
def load_pair(
    rgb: bytes, nir: bytes, channel: str, max_side: int
) -> tuple[np.ndarray, np.ndarray]:
    bands = loading.prepare_from_rgb_nir(rgb, nir)
    return bands.rgb, bands.NIR


@st.cache_data(show_spinner=False, max_entries=6)
def load_roi(data: bytes, w: int, h: int) -> np.ndarray:
    return loading.load_roi_mask(data, (w, h))


def threshold_control(prefix: str, defaults: CoverOptions) -> CoverOptions:
    cover_options = CoverOptions(**defaults.__dict__)
    cover_options.method = (
        st.segmented_control(
            "Порог", THRESHOLD_METHODS, default=cover_options.method, key=f"{prefix}_m"
        )
        or cover_options.method
    )
    if cover_options.method == "Ручной порог":
        cover_options.threshold = st.slider(
            "Значение порога",
            0.0,
            1.0,
            cover_options.threshold,
            0.01,
            key=f"{prefix}_t",
        )
    elif cover_options.method == "Адаптивный (локальный)":
        cover_options.adaptive_block_frac = st.slider(
            "Размер окна, доля кадра",
            0.02,
            0.4,
            cover_options.adaptive_block_frac,
            0.01,
            key=f"{prefix}_b",
        )
        cover_options.adaptive_offset = st.slider(
            "Смещение C",
            -0.1,
            0.1,
            cover_options.adaptive_offset,
            0.005,
            key=f"{prefix}_c",
        )
    cover_options.invert = st.toggle(
        "Класс — тёмное (инвертировать)",
        value=cover_options.invert,
        key=f"{prefix}_inv",
    )
    with st.popover("Коррекция и очистка", icon=":material/tune:"):
        cover_options.flatfield = st.toggle(
            "Выровнять освещённость (flat-field)",
            value=cover_options.flatfield,
            key=f"{prefix}_ff",
        )
        cover_options.exclude_bright = st.toggle(
            "Отбросить самые яркие пятна (этикетки)",
            value=cover_options.exclude_bright,
            key=f"{prefix}_eb",
        )
        cover_options.smooth = st.slider(
            "Сглаживание (морфология)",
            0,
            25,
            cover_options.smooth,
            1,
            key=f"{prefix}_s",
        )
        cover_options.min_area = st.slider(
            "Удалять пятна мельче, пикс.",
            0,
            8000,
            cover_options.min_area,
            50,
            key=f"{prefix}_a",
        )
        cover_options.fill_holes = st.toggle(
            "Заливать дырки", value=cover_options.fill_holes, key=f"{prefix}_fh"
        )
    return cover_options


st.title("Проективное покрытие")
st.caption(
    "Доля площади под растительностью: сегментация кадра → площадь класса × 100 / "
    "площадь учётной площадки."
)

st.sidebar.header("Задача")
task = st.sidebar.radio(
    "Тип сообщества", [TASK_GROUND, TASK_WATER], label_visibility="collapsed"
)

st.sidebar.header("Данные")
if task == TASK_GROUND:
    nir_file = st.sidebar.file_uploader(
        "ИК-снимок", type=["jpg", "jpeg", "png", "tif", "tiff"]
    )
    rgb_file = None
else:
    rgb_file = st.sidebar.file_uploader(
        "RGB-снимок", type=["jpg", "jpeg", "png", "tif", "tiff"]
    )
    nir_file = st.sidebar.file_uploader(
        "ИК-снимок", type=["jpg", "jpeg", "png", "tif", "tiff"]
    )
channel = st.sidebar.selectbox("Канал ИК-сигнала", CHANNELS)

need = [nir_file] if task == TASK_GROUND else [rgb_file, nir_file]
if not all(need):
    st.info("Загрузите снимок(и) в боковой панели.", icon=":material/upload_file:")
    st.stop()

try:
    if task == TASK_GROUND:
        band, rgb = load_band(nir_file.getvalue(), channel, 0)
    else:
        rgb, band = load_pair(rgb_file.getvalue(), nir_file.getvalue(), channel, 0)
except Exception as exc:  # noqa: BLE001
    st.error(f"Не удалось подготовить снимки: {exc}", icon=":material/error:")
    st.stop()

H, W = band.shape

st.header("1 · Учётная площадка", divider="gray")
st.caption("Знаменатель в формуле покрытия. Всё вне площадки исключается из расчёта.")

roi_mode = st.radio(
    "Как задана площадка",
    ["Весь кадр", "Прямоугольник", "Загрузить маску"],
    horizontal=True,
)
roi: np.ndarray | None = None
if roi_mode == "Прямоугольник":
    cx, cy = st.columns(2)
    x0, x1 = cx.slider("По горизонтали", 0.0, 1.0, (0.0, 1.0), 0.01)
    y0, y1 = cy.slider("По вертикали", 0.0, 1.0, (0.0, 1.0), 0.01)
    roi = rect_roi((H, W), x0, x1, y0, y1)
elif roi_mode == "Загрузить маску":
    roi_file = st.file_uploader(
        "Маска площадки (светлое = внутри)", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )
    if roi_file is None:
        st.warning(
            "Загрузите файл-маску или выберите другой режим.", icon=":material/warning:"
        )
        st.stop()
    roi = load_roi(roi_file.getvalue(), W, H)

roi_px = int(roi.sum()) if roi is not None else H * W
st.metric("Площадь учётной площадки", f"{roi_px:,} пикс.".replace(",", " "))

if task == TASK_GROUND:
    st.header("2 · Растительность", divider="gray")
    defaults = CoverOptions(method="Оцу (авто)", min_area=200)
    opts = threshold_control("veg", defaults)

    with st.spinner("Считаю маску…"):
        res = ground_cover(band, opts, roi)

    over = overlay(rgb, [(res.mask, VEG_RGB)], roi)
    c1, c2 = st.columns(2)
    c1.image(over, caption="Растительность подсвечена", width="stretch")
    c2.image(
        (res.mask * 255).astype(np.uint8), caption="Бинарная маска", width="stretch"
    )

    st.header("3 · Проективное покрытие", divider="gray")
    big = st.container(border=True)
    big.metric("Проективное покрытие", f"{res.percent:.1f} %")
    big.caption(
        f"{res.class_px:,} пикс. растительности / {res.roi_px:,} пикс. площадки".replace(
            ",", " "
        )
    )

    summary = pd.DataFrame(
        [
            {
                "Класс": "Растительность",
                "Пикселей": res.class_px,
                "Покрытие, %": round(res.percent, 2),
            }
        ]
    )
    layers = {"vegetation": res.mask}

else:
    st.header("2 · Кувшинки (по RGB)", divider="gray")
    st.caption("Плавающие листья на поверхности — индекс избытка зелёного ExG.")
    lily_auto = st.toggle("Порог ExG автоматически (Оцу)", value=True)
    lily_exg = st.slider(
        "Порог ExG для листьев", -0.2, 0.6, 0.10, 0.01, disabled=lily_auto
    )
    lily_opts = CoverOptions(smooth=5, min_area=400)
    with st.popover("Очистка маски листьев", icon=":material/tune:"):
        lily_opts.exclude_bright = st.toggle(
            "Отбросить пересветы (блики на воде)", value=False, key="lily_eb"
        )
        lily_opts.smooth = st.slider("Сглаживание", 0, 25, 5, 1, key="lily_s")
        lily_opts.min_area = st.slider(
            "Удалять пятна мельче, пикс.", 0, 8000, 400, 50, key="lily_a"
        )
        lily_opts.fill_holes = st.toggle("Заливать дырки", value=True, key="lily_fh")

    st.header("3 · Погружённая растительность (по ИК)", divider="gray")
    st.caption(
        "Внутри площадки: тёмная вода отсекается порогом, след кувшинок вычитается с запасом."
    )
    sub_defaults = CoverOptions(
        method="Оцу (авто)", flatfield=True, smooth=5, min_area=300
    )
    sub_opts = threshold_control("sub", sub_defaults)
    dilate_px = st.slider(
        "Запас вокруг кувшинок (из-за несовмещения снимков), пикс.", 0, 80, 25, 5
    )

    with st.spinner("Считаю классы…"):
        lily_res, sub_res = water_cover(
            rgb,
            band,
            lily_opts,
            sub_opts,
            roi,
            lily_exg_threshold=None if lily_auto else lily_exg,
            lily_dilate_px=dilate_px,
        )

    over = overlay(rgb, [(lily_res.mask, LILY_RGB), (sub_res.mask, SUBMERGED_RGB)], roi)
    st.image(
        over,
        caption="Зелёный — кувшинки · оранжевый — погружённая растительность",
        width="stretch",
    )
    mc1, mc2 = st.columns(2)
    mc1.image(
        (lily_res.mask * 255).astype(np.uint8), caption="Кувшинки", width="stretch"
    )
    mc2.image(
        (sub_res.mask * 255).astype(np.uint8), caption="Погружённая", width="stretch"
    )

    st.header("4 · Проективное покрытие", divider="gray")
    r1, r2 = st.columns(2)
    r1.container(border=True).metric("Кувшинки", f"{lily_res.percent:.1f} %")
    r2.container(border=True).metric(
        "Погружённая растительность", f"{sub_res.percent:.1f} %"
    )
    st.caption(
        f"Оба процента — от площадки {sub_res.roi_px:,} пикс.".replace(",", " ")
        + ". Результат по погружённой растительности приблизительный: снимки не совмещены."
    )

    summary = pd.DataFrame(
        [
            {
                "Класс": "Кувшинки",
                "Пикселей": lily_res.class_px,
                "Покрытие, %": round(lily_res.percent, 2),
            },
            {
                "Класс": "Погружённая растительность",
                "Пикселей": sub_res.class_px,
                "Покрытие, %": round(sub_res.percent, 2),
            },
        ]
    )
    layers = {"lilies": lily_res.mask, "submerged": sub_res.mask}

st.dataframe(summary, width="stretch", hide_index=True)

st.header("Экспорт", divider="gray")
roi_out = roi if roi is not None else np.ones((H, W), bool)
if st.button("Подготовить файлы", icon=":material/build:"):
    with st.spinner("Кодирую…"):
        st.session_state["cover_zip"] = build_cover_zip(over, layers, roi_out, summary)

if st.session_state.get("cover_zip"):
    st.download_button(
        "Скачать (ZIP)",
        st.session_state["cover_zip"],
        "projective_cover.zip",
        "application/zip",
        type="primary",
        icon=":material/folder_zip:",
    )

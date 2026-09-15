import streamlit as st

st.set_page_config(
    page_title="Анализ растительности", page_icon=":material/eco:", layout="wide"
)

pages = [
    st.Page(
        "app_pages/indices.py",
        title="Вегетационные индексы",
        icon=":material/analytics:",
        default=True,
    ),
    st.Page(
        "app_pages/cover.py",
        title="Проективное покрытие",
        icon=":material/pie_chart:",
    ),
]

st.navigation(pages).run()

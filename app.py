import streamlit as st

st.set_page_config(
    page_title="Workers Compensation: прогнозирование выплат",
    page_icon="💼",
    layout="wide",
)

# Многостраничная навигация на основе st.navigation и st.Page (Streamlit >= 1.36).
pages = [
    st.Page(
        "analysis_and_model.py",
        title="Анализ и модель",
        icon="📊",
        default=True,
    ),
    st.Page(
        "presentation.py",
        title="Презентация",
        icon="🎤",
    ),
]

current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()

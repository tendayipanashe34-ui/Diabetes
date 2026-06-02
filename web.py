import os

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="WIFD Website",
    page_icon="🌐",
    layout="wide",
)

st.title("WIFD Website")
st.markdown(
    "Run the existing `wifd.html` website inside Streamlit. If the page does not load, check that `wifd.html` is present in the same folder."
)

html_path = os.path.join(os.path.dirname(__file__), "wifd.html")

if not os.path.exists(html_path):
    st.error(f"Could not find `wifd.html` at: {html_path}")
else:
    with open(html_path, "r", encoding="utf-8") as html_file:
        html = html_file.read()

    components.html(html, height=1000, scrolling=True)

    st.sidebar.header("WIFD Launcher")
    st.sidebar.markdown(
        "Use this app to display the WIFD website directly from the local `wifd.html` file."
    )
    st.sidebar.markdown(
        "If the embedded page does not render correctly, open `wifd.html` in a browser or review the file for unsupported scripts."
    )

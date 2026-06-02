import os
from pathlib import Path

import streamlit as st

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
    iframe_url = Path(html_path).resolve().as_uri()
    st.iframe(iframe_url, height=1000)

    st.sidebar.header("WIFD Launcher")
    st.sidebar.markdown(
        "Use this app to display the WIFD website directly from the local `wifd.html` file."
    )
    st.sidebar.markdown(
        "If the embedded page does not render correctly, open `wifd.html` in a browser or review the file for unsupported scripts."
    )

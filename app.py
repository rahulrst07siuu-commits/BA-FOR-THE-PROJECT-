
from pathlib import Path
import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

possible_model_paths = [
    BASE_DIR / "artifacts" / "nifty50_model.joblib",
    BASE_DIR / "nifty50_model.joblib"
]

MODEL_PATH = next(
    (path for path in possible_model_paths if path.exists()),
    None
)

if MODEL_PATH is None:
    st.error("Trained model not found.")

    st.write("Files available in app directory:")
    st.write([p.name for p in BASE_DIR.iterdir()])

    st.stop()

model_bundle = joblib.load(MODEL_PATH)

st.success(f"Model loaded successfully: {MODEL_PATH.name}")

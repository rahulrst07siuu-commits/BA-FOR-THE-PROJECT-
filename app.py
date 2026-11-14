import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="NIFTY 50 Next-Session Model", page_icon="📈", layout="wide")
st.title("NIFTY 50 — Next-Session Return Prediction")
st.caption("Real historical index data · chronological train/validation/test · walk-forward cross-validation")

ARTIFACT = Path("artifacts/nifty50_model.joblib")
DATA = Path("Nifty 50 Historical Data.csv")

st.info(
    "This model predicts the next trading session's close-to-close percentage return "
    "from information available through the current session close. It does not guarantee "
    "a profitable trading signal."
)

if not ARTIFACT.exists():
    st.warning("No trained model artifact found. Run `python train_model.py` first, then refresh this app.")
    st.stop()

bundle = joblib.load(ARTIFACT)
model = bundle["model"]
features = bundle["features"]
evaluation = bundle["evaluation"]
last_features = bundle["last_feature_row"]
last_close = float(bundle["last_close"])

st.subheader("Training and testing results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected model", bundle["selected_model"])
c2.metric("Untouched test R²", f"{evaluation['test_r2']:.3f}")
c3.metric("Test MAE", f"{evaluation['test_mae_pct_points']:.3f} pp")
c4.metric("Directional accuracy", f"{evaluation['test_directional_accuracy']*100:.1f}%")

if evaluation["test_r2"] <= 0:
    st.warning(
        "The final holdout R² is zero or negative. The model has not demonstrated "
        "reliable return-prediction skill against the test-set mean. Treat predictions "
        "as experimental; the app will not label this as a BUY or SELL signal."
    )
elif evaluation["test_r2"] < 0.80:
    st.warning(
        "The model has positive but below-80% R² on the untouched test period. "
        "Do not force an 80–85% target by changing the test set or leaking future data."
    )

st.subheader("Change input variables and predict")
st.write("Defaults are the latest available feature values from the uploaded dataset. Change any value to run a what-if prediction.")

help_text = {
    "nifty_return_pct": "Current session close-to-close NIFTY 50 return (%)",
    "return_lag1_pct": "Most recent completed daily return (%)",
    "return_lag2_pct": "Return from two sessions ago (%)",
    "return_lag5_pct": "Return from five sessions ago (%)",
    "return_lag10_pct": "Return from ten sessions ago (%)",
    "rolling_volatility_5d_pct": "5-session standard deviation of daily returns",
    "rolling_volatility_10d_pct": "10-session standard deviation of daily returns",
    "rolling_volatility_20d_pct": "20-session standard deviation of daily returns",
    "intraday_range_pct": "(High - Low) / Open × 100",
    "open_gap_pct": "Today's open relative to previous close (%)",
    "deviation_sma5_pct": "Close deviation from 5-session moving average (%)",
    "deviation_sma20_pct": "Close deviation from 20-session moving average (%)",
    "deviation_sma50_pct": "Close deviation from 50-session moving average (%)",
    "volume_change_pct": "Daily traded-volume percentage change",
    "rsi14": "14-session Relative Strength Index"
}

edited = {}
cols = st.columns(2)
for idx, name in enumerate(features):
    with cols[idx % 2]:
        default = float(last_features[name])
        edited[name] = st.number_input(
            name.replace("_", " ").title(),
            value=default,
            format="%.6f",
            help=help_text.get(name, "")
        )

if st.button("Predict next-session return", type="primary", use_container_width=True):
    input_df = pd.DataFrame([edited], columns=features)
    predicted_return = float(model.predict(input_df)[0])
    implied_close = last_close * (1 + predicted_return / 100)
    st.divider()
    a, b, c = st.columns(3)
    a.metric("Predicted next-session return", f"{predicted_return:+.4f}%")
    b.metric("Implied index close", f"{implied_close:,.2f}")
    c.metric("Reference close", f"{last_close:,.2f}")
    if evaluation["test_r2"] <= 0:
        st.error("Model status: No demonstrated predictive edge on the untouched test set.")
    elif predicted_return > 0:
        st.success("Model direction: Positive estimated return (not a buy recommendation).")
    elif predicted_return < 0:
        st.warning("Model direction: Negative estimated return (not a sell recommendation).")
    else:
        st.info("Model direction: Approximately flat.")
    st.caption(
        "The implied close is a mathematical conversion of the predicted return. "
        "It is not an independently forecast price target."
    )

with st.expander("Full evaluation details"):
    st.json(evaluation)
    for filename in [
        "artifacts/walk_forward_cv_results.csv",
        "artifacts/validation_model_comparison.csv",
        "artifacts/test_evaluation.csv"
    ]:
        if Path(filename).exists():
            st.write(filename)
            st.dataframe(pd.read_csv(filename), use_container_width=True)

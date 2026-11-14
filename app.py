
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NIFTY 50 Prediction Terminal",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# 2. FIND THE TRAINED MODEL
# Supports both root and artifacts folder locations
# =========================================================

MODEL_PATHS = [
    BASE_DIR / "nifty50_model.joblib",
    BASE_DIR / "artifacts" / "nifty50_model.joblib"
]

MODEL_PATH = next(
    (path for path in MODEL_PATHS if path.is_file()),
    None
)


# =========================================================
# 3. APPLICATION HEADER
# =========================================================

st.title("NIFTY 50 — Next-Session Return Prediction")

st.caption(
    "Real historical index data | "
    "Chronological training, validation and testing | "
    "Walk-forward cross-validation"
)

st.info(
    "This model estimates the next trading session's "
    "close-to-close percentage return using information "
    "available through the current session close. "
    "It does not guarantee a profitable trading signal."
)


# =========================================================
# 4. CHECK MODEL AVAILABILITY
# =========================================================

if MODEL_PATH is None:

    st.error("Trained model file was not found.")

    st.write("Expected model locations:")

    for path in MODEL_PATHS:
        st.code(str(path))

    st.write("Files currently available in the repository:")

    st.write([
        p.name for p in BASE_DIR.iterdir()
    ])

    st.warning(
        "Upload nifty50_model.joblib to the same folder "
        "as app.py, or put it inside an artifacts folder."
    )

    st.stop()


# =========================================================
# 5. LOAD MODEL
# =========================================================

try:

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    features = bundle["features"]
    evaluation = bundle["evaluation"]

    last_features = bundle["last_feature_row"]
    last_close = float(bundle["last_close"])

    selected_model = bundle["selected_model"]
    last_date = bundle.get("last_date", "Not available")

except Exception as error:

    st.error("Unable to load the trained model.")

    st.exception(error)

    st.info(
        "Check that the model was trained using compatible "
        "versions of scikit-learn, NumPy and joblib."
    )

    st.stop()


st.success(
    f"Trained model loaded successfully: {MODEL_PATH.name}"
)


# =========================================================
# 6. TRAINING AND TESTING RESULTS
# =========================================================

st.header("Model Training and Testing Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Selected Model",
    selected_model
)

col2.metric(
    "Test R²",
    f"{evaluation['test_r2']:.4f}"
)

col3.metric(
    "Test MAE",
    f"{evaluation['test_mae_pct_points']:.4f}"
)

col4.metric(
    "Directional Accuracy",
    f"{evaluation['test_directional_accuracy'] * 100:.2f}%"
)


# =========================================================
# 7. DISPLAY DATASET AND MODEL INFORMATION
# =========================================================

with st.expander("Dataset and model information"):

    st.write("**Model:**", selected_model)

    st.write(
        "**Historical data period:**",
        f"{evaluation['date_start']} to {evaluation['date_end']}"
    )

    st.write("**Latest available data date:**", last_date)

    st.write("**Total dataset rows:**", evaluation["dataset_rows"])

    st.write("**Modeling observations:**", evaluation["modeling_rows"])

    st.write("**Training observations:**", evaluation["train_rows"])

    st.write("**Validation observations:**", evaluation["validation_rows"])

    st.write("**Testing observations:**", evaluation["test_rows"])

    st.write("**Prediction target:**", evaluation["target"])


# =========================================================
# 8. MODEL PERFORMANCE WARNING
# =========================================================

test_r2 = float(evaluation["test_r2"])

if test_r2 <= 0:

    st.warning(
        "The model has zero or negative R² on the untouched "
        "test dataset. It has not demonstrated a reliable "
        "predictive advantage. Treat predictions as experimental."
    )

elif test_r2 < 0.80:

    st.warning(
        "The model has positive but below 80% test R². "
        "Do not force a higher score by changing the test data "
        "or introducing future information."
    )

else:

    st.success(
        "The model has achieved positive test R². "
        "Continue checking stability and benchmark performance."
    )


# =========================================================
# 9. INPUT VARIABLE EXPLANATIONS
# =========================================================

help_text = {

    "nifty_return_pct":
        "Current session close-to-close NIFTY 50 return (%)",

    "return_lag1_pct":
        "Most recent completed daily return (%)",

    "return_lag2_pct":
        "Return from two sessions ago (%)",

    "return_lag5_pct":
        "Return from five sessions ago (%)",

    "return_lag10_pct":
        "Return from ten sessions ago (%)",

    "rolling_volatility_5d_pct":
        "Five-session standard deviation of daily returns",

    "rolling_volatility_10d_pct":
        "Ten-session standard deviation of daily returns",

    "rolling_volatility_20d_pct":
        "Twenty-session standard deviation of daily returns",

    "intraday_range_pct":
        "Today's high-low range divided by opening price (%)",

    "open_gap_pct":
        "Today's opening price compared with previous close (%)",

    "deviation_sma5_pct":
        "Current close deviation from 5-session moving average (%)",

    "deviation_sma20_pct":
        "Current close deviation from 20-session moving average (%)",

    "deviation_sma50_pct":
        "Current close deviation from 50-session moving average (%)",

    "volume_change_pct":
        "Daily traded-volume percentage change",

    "rsi14":
        "14-session Relative Strength Index"

}


# =========================================================
# 10. USER INPUT SECTION
# =========================================================

st.header("Change Input Variables")

st.write(
    "The default values are taken from the latest available "
    "historical observation used during model preparation. "
    "Change the values to perform a what-if prediction."
)

st.caption(
    f"Reference NIFTY 50 close: {last_close:,.2f} | "
    f"Latest data date: {last_date}"
)

edited = {}

columns = st.columns(2)

for index, feature in enumerate(features):

    with columns[index % 2]:

        default_value = float(last_features[feature])

        edited[feature] = st.number_input(

            label=feature.replace("_", " ").title(),

            value=default_value,

            format="%.6f",

            help=help_text.get(feature, "")

        )


# =========================================================
# 11. PREDICTION BUTTON
# =========================================================

st.divider()

if st.button(
    "Predict Next-Session Return",
    type="primary",
    use_container_width=True
):

    try:

        input_df = pd.DataFrame(
            [edited],
            columns=features
        )

        predicted_return = float(
            model.predict(input_df)[0]
        )

        implied_close = (
            last_close * (1 + predicted_return / 100)
        )

        st.header("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted Next-Session Return",
            f"{predicted_return:+.4f}%"
        )

        col2.metric(
            "Implied Index Close",
            f"{implied_close:,.2f}"
        )

        col3.metric(
            "Reference Index Close",
            f"{last_close:,.2f}"
        )


        # Directional interpretation

        if test_r2 <= 0:

            st.error(
                "Model status: No demonstrated predictive edge "
                "on the untouched test dataset."
            )

        elif predicted_return > 0:

            st.success(
                "Model direction: Positive estimated return. "
                "This is not a BUY recommendation."
            )

        elif predicted_return < 0:

            st.warning(
                "Model direction: Negative estimated return. "
                "This is not a SELL recommendation."
            )

        else:

            st.info(
                "Model direction: Approximately flat."
            )


        st.caption(
            "The implied closing index is calculated from "
            "the predicted percentage return. It is not "
            "an independently forecast price target."
        )

        st.subheader("Your Input Values")

        st.dataframe(
            input_df,
            use_container_width=True
        )

    except Exception as error:

        st.error("Prediction could not be completed.")

        st.exception(error)


# =========================================================
# 12. EVALUATION FILES
# =========================================================

st.divider()

st.header("Detailed Model Evaluation")

evaluation_files = [

    "walk_forward_cv_results.csv",

    "validation_model_comparison.csv",

    "test_evaluation.csv",

    "test_predictions.csv"

]

for filename in evaluation_files:

    possible_paths = [

        BASE_DIR / filename,

        BASE_DIR / "artifacts" / filename

    ]

    file_path = next(
        (p for p in possible_paths if p.is_file()),
        None
    )

    if file_path is not None:

        with st.expander(filename):

            try:

                result_df = pd.read_csv(file_path)

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

            except Exception as error:

                st.error(str(error))


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "NIFTY 50 Machine Learning Project | "
    "Educational and research purposes only. "
    "Predictions are estimates, not investment advice."
)

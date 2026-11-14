# NIFTY 50 Next-Session Return Prediction

## What this project does
- Uses the supplied 10-year NIFTY 50 OHLCV CSV.
- Predicts next-session close-to-close index return (%).
- Creates lagged returns, rolling volatility, intraday range, opening gap,
  moving-average deviations, volume change and RSI.
- Uses a chronological 70% train / 15% validation / 15% final test split.
- Runs 5-fold expanding-window TimeSeriesSplit cross-validation on the training segment.
- Selects the model using validation MAE, then refits on train + validation.
- Evaluates once on the untouched final test period.
- Compares against a zero-return baseline.
- Provides editable input controls in Streamlit.

## Important
Do not force a target R². A target of 80–85% is not guaranteed and is not a valid
acceptance criterion for noisy one-day financial returns. A negative test R² means
the model has not beaten a simple mean-return benchmark on that test period.
Do not randomly shuffle time series, use future information, or tune on the final test set.

## Setup
Use Python 3.13 (runtime.txt) and install dependencies:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The training script creates:
- artifacts/nifty50_model.joblib
- artifacts/walk_forward_cv_results.csv
- artifacts/validation_model_comparison.csv
- artifacts/test_evaluation.csv
- artifacts/test_predictions.csv

## Data and scope
The CSV supplied for this project contains NIFTY 50 index OHLCV history only.
It does not contain India VIX, S&P 500 returns, constituents' fundamentals or
company-level data. Those variables are not fabricated or included.
The original CSV does not identify its upstream provider; independently verify
the source and permitted use before publication.

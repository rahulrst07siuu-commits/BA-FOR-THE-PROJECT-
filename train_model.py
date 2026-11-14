import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from feature_engineering import load_and_build, FEATURES

warnings.filterwarnings("ignore")
DATA_FILE = "Nifty 50 Historical Data.csv"
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

raw, data, latest_features = load_and_build(DATA_FILE)
X = data[FEATURES].copy()
y = data["target_next_return_pct"].copy()

# Chronological 70/15/15 split: never shuffle time-series observations.
n = len(data)
i_train = int(n * 0.70)
i_val = int(n * 0.85)
X_train, y_train = X.iloc[:i_train], y.iloc[:i_train]
X_val, y_val = X.iloc[i_train:i_val], y.iloc[i_train:i_val]
X_test, y_test = X.iloc[i_val:], y.iloc[i_val:]

models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0))
    ]),
    "Lasso Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.01, max_iter=20000))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=15,
            max_features=0.8, random_state=42, n_jobs=-1
        ))
    ]),
    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=300, max_depth=4, min_samples_leaf=15,
            max_features=0.8, random_state=42, n_jobs=-1
        ))
    ]),
    "Hist Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=7, l2_regularization=10,
            learning_rate=0.03, random_state=42
        ))
    ])
}

# Time-series CV is performed only inside the training period.
tscv = TimeSeriesSplit(n_splits=5)
cv_rows = []
val_rows = []
fitted_candidates = {}

for name, model in models.items():
    cv_result = cross_validate(
        model, X_train, y_train, cv=tscv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        },
        return_train_score=True, n_jobs=-1
    )
    for fold, (r2, mae, rmse, train_r2) in enumerate(zip(
        cv_result["test_r2"],
        -cv_result["test_mae"],
        -cv_result["test_rmse"],
        cv_result["train_r2"]
    ), start=1):
        cv_rows.append({
            "Model": name, "Fold": fold,
            "Train_R2": train_r2, "Validation_R2": r2,
            "MAE": mae, "RMSE": rmse
        })

    model.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    val_rows.append({
        "Model": name,
        "Validation_R2": r2_score(y_val, pred_val),
        "Validation_MAE": mean_absolute_error(y_val, pred_val),
        "Validation_RMSE": mean_squared_error(y_val, pred_val) ** 0.5
    })
    fitted_candidates[name] = model

cv_df = pd.DataFrame(cv_rows)
val_df = pd.DataFrame(val_rows)

# Select on validation MAE, not on the final test set.
selected_name = val_df.sort_values("Validation_MAE").iloc[0]["Model"]

# Refit selected model on train + validation; final test remains untouched.
X_dev = pd.concat([X_train, X_val])
y_dev = pd.concat([y_train, y_val])
final_model = models[selected_name]
final_model.fit(X_dev, y_dev)
test_pred = final_model.predict(X_test)

test_r2 = r2_score(y_test, test_pred)
test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
directional_accuracy = np.mean((test_pred >= 0) == (y_test >= 0))

# Zero-return baseline: predicts 0% for every next session.
baseline = np.zeros(len(y_test))
baseline_r2 = r2_score(y_test, baseline)
baseline_mae = mean_absolute_error(y_test, baseline)
baseline_rmse = mean_squared_error(y_test, baseline) ** 0.5

evaluation = {
    "dataset_rows": int(len(raw)),
    "modeling_rows": int(len(data)),
    "date_start": str(data["Date"].min().date()),
    "date_end": str(data["Date"].max().date()),
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "selected_model": selected_name,
    "test_r2": float(test_r2),
    "test_mae_pct_points": float(test_mae),
    "test_rmse_pct_points": float(test_rmse),
    "test_directional_accuracy": float(directional_accuracy),
    "zero_return_baseline_r2": float(baseline_r2),
    "zero_return_baseline_mae_pct_points": float(baseline_mae),
    "zero_return_baseline_rmse_pct_points": float(baseline_rmse),
    "sklearn_version": sklearn.__version__,
    "target": "Next trading session close-to-close NIFTY 50 return (%)",
    "warning": "A negative or weak test R2 means the model has not demonstrated reliable predictive edge."
}

joblib.dump({
    "model": final_model,
    "features": FEATURES,
    "selected_model": selected_name,
    "evaluation": evaluation,
    "last_feature_row": latest_features[FEATURES].iloc[0].to_dict(),
    "last_close": float(raw["Price"].iloc[-1]),
    "last_date": str(raw["Date"].iloc[-1].date())
}, os.path.join(OUT_DIR, "nifty50_model.joblib"))

cv_df.to_csv(os.path.join(OUT_DIR, "walk_forward_cv_results.csv"), index=False)
val_df.to_csv(os.path.join(OUT_DIR, "validation_model_comparison.csv"), index=False)
pd.DataFrame([evaluation]).to_csv(os.path.join(OUT_DIR, "test_evaluation.csv"), index=False)

test_df = pd.DataFrame({
    "Date": data["Date"].iloc[i_val:].values,
    "Actual_Next_Return_Pct": y_test.values,
    "Predicted_Next_Return_Pct": test_pred
})
test_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

print("\n=== DATA SPLIT ===")
print(f"Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")
print("\n=== VALIDATION COMPARISON ===")
print(val_df.sort_values("Validation_MAE").to_string(index=False))
print("\n=== FINAL UNTOUCHED TEST RESULTS ===")
print(json.dumps(evaluation, indent=2))
print("\nArtifacts saved in:", OUT_DIR)

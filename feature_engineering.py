import numpy as np
import pandas as pd

FEATURES = [
    "nifty_return_pct", "return_lag1_pct", "return_lag2_pct",
    "return_lag5_pct", "return_lag10_pct", "rolling_volatility_5d_pct",
    "rolling_volatility_10d_pct", "rolling_volatility_20d_pct",
    "intraday_range_pct", "open_gap_pct", "deviation_sma5_pct",
    "deviation_sma20_pct", "deviation_sma50_pct", "volume_change_pct",
    "rsi14"
]

def load_and_build(path):
    df = pd.read_csv(path)
    required = ["Date", "Price", "Open", "High", "Low", "Vol."]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    for col in ["Price", "Open", "High", "Low"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["Volume"] = pd.to_numeric(
        df["Vol."].astype(str).str.replace("M", "", regex=False),
        errors="coerce"
    ) * 1_000_000

    df = df.dropna(subset=["Date", "Price", "Open", "High", "Low"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    ret = df["Price"].pct_change(fill_method=None) * 100
    f = pd.DataFrame(index=df.index)
    f["nifty_return_pct"] = ret
    f["return_lag1_pct"] = ret
    f["return_lag2_pct"] = ret.shift(1)
    f["return_lag5_pct"] = ret.shift(4)
    f["return_lag10_pct"] = ret.shift(9)

    for w in [5, 10, 20]:
        f[f"rolling_volatility_{w}d_pct"] = ret.rolling(w).std()

    f["intraday_range_pct"] = (df["High"] - df["Low"]) / df["Open"] * 100
    f["open_gap_pct"] = (df["Open"] / df["Price"].shift(1) - 1) * 100

    for w in [5, 20, 50]:
        sma = df["Price"].rolling(w).mean()
        f[f"deviation_sma{w}_pct"] = (df["Price"] / sma - 1) * 100

    f["volume_change_pct"] = df["Volume"].pct_change(fill_method=None) * 100

    delta = df["Price"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi14"] = 100 - (100 / (1 + rs))

    # At date t, features use information available through close t.
    # Target is close-to-close return from t to the next trading session.
    target = (df["Price"].shift(-1) / df["Price"] - 1) * 100
    latest_features = f.replace([np.inf, -np.inf], np.nan).dropna().iloc[[-1]].copy()

    out = f.copy()
    out["target_next_return_pct"] = target
    out["Date"] = df["Date"]
    out["current_close"] = df["Price"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df, out, latest_features

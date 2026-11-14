import unittest
import pandas as pd
import joblib
from pathlib import Path
from feature_engineering import load_and_build, FEATURES

class TestNiftyProject(unittest.TestCase):
    def test_data_and_feature_alignment(self):
        raw, modeling, latest = load_and_build("Nifty 50 Historical Data.csv")
        self.assertEqual(len(raw), 2693)
        self.assertEqual(list(latest[FEATURES].columns), FEATURES)
        self.assertEqual(str(raw["Date"].iloc[-1].date()), "2026-11-13")
        self.assertAlmostEqual(float(raw["Price"].iloc[-1]), 25144.30, places=2)
        self.assertEqual(str(modeling["Date"].iloc[-1].date()), "2026-11-12")

    def test_saved_model_predicts(self):
        b = joblib.load("artifacts/nifty50_model.joblib")
        x = pd.DataFrame([b["last_feature_row"]], columns=b["features"])
        pred = b["model"].predict(x)
        self.assertEqual(len(pred), 1)
        self.assertTrue(pd.notna(pred[0]))

    def test_time_ordered_split_counts(self):
        b = joblib.load("artifacts/nifty50_model.joblib")
        e = b["evaluation"]
        self.assertEqual(e["train_rows"], 1822)
        self.assertEqual(e["validation_rows"], 391)
        self.assertEqual(e["test_rows"], 391)

if __name__ == "__main__":
    unittest.main()

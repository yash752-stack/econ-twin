"""
ml/forecasting.py
ML Macroeconomic Forecasting Models
XGBoost for GDP growth and inflation forecasting.
Feature engineering from lagged economic indicators.

Run: python ml/forecasting.py
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

print("=" * 65)
print("  GLOBAL ECONOMIC DIGITAL TWIN — ML Forecasting v1.0")
print("  yash752-stack | Module 5: XGBoost Macro Forecasting")
print("=" * 65)

os.makedirs("data/processed", exist_ok=True)
os.makedirs("ml/models", exist_ok=True)

econ_df = pd.read_csv("data/processed/economic_data.csv")
comm_df = pd.read_csv("data/processed/commodity_data.csv")

# ── FEATURE ENGINEERING ──────────────────────────────────────────────────────
print("\n[1/4] Engineering features...")

econ_df = econ_df.sort_values(["iso3", "year"])

def create_lag_features(df, target_col, lags=[1, 2, 3]):
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df.groupby("iso3")[target_col].shift(lag)
    return df

# Lag features
for col in ["gdp_growth_pct", "inflation_pct", "unemployment_pct",
            "interest_rate_pct", "exports_usd", "imports_usd"]:
    econ_df = create_lag_features(econ_df, col, lags=[1, 2, 3])

# Rolling features
for col in ["gdp_growth_pct", "inflation_pct"]:
    econ_df[f"{col}_roll3"] = econ_df.groupby("iso3")[col].transform(lambda x: x.rolling(3, min_periods=1).mean())
    econ_df[f"{col}_roll3_std"] = econ_df.groupby("iso3")[col].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))

# Trade balance ratio
econ_df["trade_balance_pct_gdp"] = (econ_df["exports_usd"] - econ_df["imports_usd"]) / econ_df["gdp_usd"] * 100

# Merge commodity prices
econ_df = econ_df.merge(comm_df[["year", "oil_brent", "gold", "copper", "wheat"]], on="year", how="left")

# Commodity YoY change
for comm in ["oil_brent", "gold", "copper"]:
    econ_df[f"{comm}_yoy"] = econ_df[comm].pct_change() * 100

# Region dummies
econ_df = pd.get_dummies(econ_df, columns=["region"], prefix="region", drop_first=True)

# Drop rows with NaN lag features
econ_df = econ_df.dropna(subset=["gdp_growth_pct_lag1", "inflation_pct_lag1"])

print(f"      ✅ Dataset: {len(econ_df)} rows × {len(econ_df.columns)} features")

# ── MODEL 1: GDP GROWTH FORECASTING ─────────────────────────────────────────
print("\n[2/4] Training GDP Growth Forecasting Model (XGBoost)...")

GDP_FEATURES = [
    "gdp_growth_pct_lag1", "gdp_growth_pct_lag2", "gdp_growth_pct_lag3",
    "gdp_growth_pct_roll3", "gdp_growth_pct_roll3_std",
    "inflation_pct_lag1", "inflation_pct_lag2",
    "interest_rate_pct_lag1", "unemployment_pct_lag1",
    "trade_balance_pct_gdp",
    "oil_brent_yoy", "copper_yoy",
    "exports_usd_lag1", "imports_usd_lag1",
    "population", "gdp_per_capita",
]
GDP_FEATURES = [f for f in GDP_FEATURES if f in econ_df.columns]

train_df = econ_df[econ_df["year"] <= 2020]
test_df  = econ_df[econ_df["year"] > 2020]

X_train = train_df[GDP_FEATURES].fillna(0)
y_train = train_df["gdp_growth_pct"]
X_test  = test_df[GDP_FEATURES].fillna(0)
y_test  = test_df["gdp_growth_pct"]

gdp_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
gdp_model.fit(X_train, y_train)
gdp_preds = gdp_model.predict(X_test)

gdp_mae = mean_absolute_error(y_test, gdp_preds)
gdp_r2  = r2_score(y_test, gdp_preds)
print(f"      MAE: {gdp_mae:.3f}% | R²: {gdp_r2:.3f}")

# Feature importance
gdp_importance = pd.DataFrame({
    "feature": GDP_FEATURES,
    "importance": gdp_model.feature_importances_
}).sort_values("importance", ascending=False)

# ── MODEL 2: INFLATION FORECASTING ──────────────────────────────────────────
print("\n[3/4] Training Inflation Forecasting Model (XGBoost)...")

INF_FEATURES = [
    "inflation_pct_lag1", "inflation_pct_lag2", "inflation_pct_lag3",
    "inflation_pct_roll3", "inflation_pct_roll3_std",
    "gdp_growth_pct_lag1", "interest_rate_pct_lag1",
    "unemployment_pct_lag1", "trade_balance_pct_gdp",
    "oil_brent_yoy", "gold_yoy", "copper_yoy",
    "exports_usd_lag1", "govt_debt_pct_gdp",
]
INF_FEATURES = [f for f in INF_FEATURES if f in econ_df.columns]

X_train_inf = train_df[INF_FEATURES].fillna(0)
y_train_inf = train_df["inflation_pct"]
X_test_inf  = test_df[INF_FEATURES].fillna(0)
y_test_inf  = test_df["inflation_pct"]

inf_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
inf_model.fit(X_train_inf, y_train_inf)
inf_preds = inf_model.predict(X_test_inf)

inf_mae = mean_absolute_error(y_test_inf, inf_preds)
inf_r2  = r2_score(y_test_inf, inf_preds)
print(f"      MAE: {inf_mae:.3f}% | R²: {inf_r2:.3f}")

inf_importance = pd.DataFrame({
    "feature": INF_FEATURES,
    "importance": inf_model.feature_importances_
}).sort_values("importance", ascending=False)

# ── GENERATE 2024-2027 FORECASTS ─────────────────────────────────────────────
print("\n[4/4] Generating 2024–2027 forecasts for all countries...")

FORECAST_YEARS = [2024, 2025, 2026, 2027]
forecast_rows = []

latest = econ_df[econ_df["year"] == econ_df["year"].max()].set_index("iso3")

for iso3 in latest.index:
    row = latest.loc[iso3].copy()
    for year in FORECAST_YEARS:
        # Build feature vector from latest known values
        gdp_feat = {f: row.get(f, 0) for f in GDP_FEATURES}
        inf_feat = {f: row.get(f, 0) for f in INF_FEATURES}

        gdp_pred = float(gdp_model.predict(pd.DataFrame([gdp_feat]).fillna(0))[0])
        inf_pred = float(inf_model.predict(pd.DataFrame([inf_feat]).fillna(0))[0])
        inf_pred = max(0.5, inf_pred)

        # Uncertainty grows with horizon
        horizon = year - 2023
        gdp_uncertainty = 0.8 * horizon
        inf_uncertainty = 0.6 * horizon

        forecast_rows.append({
            "iso3":            iso3,
            "country":         row.get("country", iso3),
            "year":            year,
            "gdp_growth_forecast": round(gdp_pred, 2),
            "gdp_growth_lower":    round(gdp_pred - gdp_uncertainty, 2),
            "gdp_growth_upper":    round(gdp_pred + gdp_uncertainty, 2),
            "inflation_forecast":  round(inf_pred, 2),
            "inflation_lower":     round(inf_pred - inf_uncertainty, 2),
            "inflation_upper":     round(max(0, inf_pred + inf_uncertainty), 2),
            "model_confidence":    round(max(0.3, 1.0 - 0.15 * horizon), 2),
        })

        # Update row for next year (autoregressive)
        row["gdp_growth_pct_lag2"] = row.get("gdp_growth_pct_lag1", gdp_pred)
        row["gdp_growth_pct_lag1"] = gdp_pred
        row["inflation_pct_lag2"]  = row.get("inflation_pct_lag1", inf_pred)
        row["inflation_pct_lag1"]  = inf_pred

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv("data/processed/ml_forecasts.csv", index=False)

# Save model metrics
model_metrics = {
    "gdp_model": {"mae": round(gdp_mae, 3), "r2": round(gdp_r2, 3), "top_features": gdp_importance.head(5)["feature"].tolist()},
    "inflation_model": {"mae": round(inf_mae, 3), "r2": round(inf_r2, 3), "top_features": inf_importance.head(5)["feature"].tolist()},
}
with open("data/processed/model_metrics.json", "w") as f:
    json.dump(model_metrics, f, indent=2)

print(f"      ✅ Forecasts: {len(forecast_df)} rows ({len(latest)} countries × {len(FORECAST_YEARS)} years)")
print(f"\n  GDP Model     → MAE: {gdp_mae:.2f}pp | R²: {gdp_r2:.3f}")
print(f"  Inflation Model → MAE: {inf_mae:.2f}pp | R²: {inf_r2:.3f}")
print(f"\n  Top GDP predictors: {', '.join(gdp_importance.head(3)['feature'].tolist())}")
print(f"  Top Inflation predictors: {', '.join(inf_importance.head(3)['feature'].tolist())}")
print("\n✅ Module 5 complete — ML forecasts saved\n")

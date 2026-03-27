"""
tests/test_forecasting.py
Unit tests for ml/forecasting.py outputs.

Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import json
import os

DATA_DIR = "data/processed"


@pytest.fixture(scope="module")
def forecast_df():
    path = f"{DATA_DIR}/ml_forecasts.csv"
    if not os.path.exists(path):
        pytest.skip("Run python run_all.py first")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def model_metrics():
    path = f"{DATA_DIR}/model_metrics.json"
    if not os.path.exists(path):
        pytest.skip("Run python run_all.py first")
    with open(path) as f:
        return json.load(f)


class TestForecastSchema:

    def test_required_columns(self, forecast_df):
        for col in ["iso3", "country", "year", "gdp_growth_forecast",
                    "gdp_growth_lower", "gdp_growth_upper",
                    "inflation_forecast", "inflation_lower", "inflation_upper",
                    "model_confidence"]:
            assert col in forecast_df.columns, f"Missing column: {col}"

    def test_forecast_years(self, forecast_df):
        years = set(forecast_df["year"].unique())
        assert years == {2024, 2025, 2026, 2027}, f"Unexpected forecast years: {years}"

    def test_row_count(self, forecast_df):
        # 30 countries × 4 years = 120
        assert len(forecast_df) == 120, f"Expected 120 rows, got {len(forecast_df)}"

    def test_all_countries_forecasted(self, forecast_df):
        assert forecast_df["iso3"].nunique() == 30


class TestForecastValues:

    def test_confidence_bounds(self, forecast_df):
        assert (forecast_df["model_confidence"] >= 0).all()
        assert (forecast_df["model_confidence"] <= 1).all()

    def test_confidence_decreases_with_horizon(self, forecast_df):
        """Confidence should decrease as forecast horizon increases."""
        avg_conf = forecast_df.groupby("year")["model_confidence"].mean()
        years_sorted = sorted(avg_conf.index)
        for i in range(len(years_sorted) - 1):
            assert avg_conf[years_sorted[i]] >= avg_conf[years_sorted[i + 1]], \
                "Confidence should not increase with forecast horizon"

    def test_uncertainty_bands_valid(self, forecast_df):
        """Lower bound must be <= point forecast <= upper bound."""
        assert (forecast_df["gdp_growth_lower"] <= forecast_df["gdp_growth_forecast"]).all(), \
            "gdp_growth_lower exceeds point forecast"
        assert (forecast_df["gdp_growth_forecast"] <= forecast_df["gdp_growth_upper"]).all(), \
            "gdp_growth_upper below point forecast"
        assert (forecast_df["inflation_lower"] <= forecast_df["inflation_forecast"]).all()
        assert (forecast_df["inflation_forecast"] <= forecast_df["inflation_upper"]).all()

    def test_inflation_forecast_non_negative(self, forecast_df):
        assert (forecast_df["inflation_lower"] >= 0).all(), \
            "Inflation forecast lower bound cannot be negative"

    def test_gdp_forecast_reasonable_range(self, forecast_df):
        assert forecast_df["gdp_growth_forecast"].min() > -15
        assert forecast_df["gdp_growth_forecast"].max() < 20

    def test_band_widens_with_horizon(self, forecast_df):
        """Uncertainty band width should increase for later forecast years."""
        forecast_df = forecast_df.copy()
        forecast_df["band_width"] = forecast_df["gdp_growth_upper"] - forecast_df["gdp_growth_lower"]
        avg_width = forecast_df.groupby("year")["band_width"].mean()
        years = sorted(avg_width.index)
        for i in range(len(years) - 1):
            assert avg_width[years[i]] <= avg_width[years[i + 1]], \
                f"Band width shrank from {years[i]} to {years[i+1]}"


class TestModelMetrics:

    def test_model_metrics_schema(self, model_metrics):
        for model_key in ["gdp_model", "inflation_model"]:
            assert model_key in model_metrics
            for field in ["mae", "r2", "top_features"]:
                assert field in model_metrics[model_key], \
                    f"Missing field '{field}' in {model_key}"

    def test_r2_positive(self, model_metrics):
        """Both models should achieve positive R² — better than predicting the mean."""
        for model_key in ["gdp_model", "inflation_model"]:
            r2 = model_metrics[model_key]["r2"]
            assert r2 > 0, f"{model_key} has negative R²: {r2:.3f}"

    def test_mae_reasonable(self, model_metrics):
        """GDP MAE should be under 3pp; inflation MAE under 5pp."""
        assert model_metrics["gdp_model"]["mae"] < 3.0, \
            f"GDP MAE too high: {model_metrics['gdp_model']['mae']:.3f}"
        assert model_metrics["inflation_model"]["mae"] < 5.0, \
            f"Inflation MAE too high: {model_metrics['inflation_model']['mae']:.3f}"

    def test_top_features_list_non_empty(self, model_metrics):
        for model_key in ["gdp_model", "inflation_model"]:
            features = model_metrics[model_key]["top_features"]
            assert isinstance(features, list) and len(features) > 0, \
                f"{model_key} top_features is empty"

    def test_lag1_in_top_features(self, model_metrics):
        """Lag-1 of the target should be among top predictors (autoregressive property)."""
        gdp_features = model_metrics["gdp_model"]["top_features"]
        has_lag1 = any("lag1" in f for f in gdp_features)
        assert has_lag1, f"No lag1 feature in GDP top features: {gdp_features}"

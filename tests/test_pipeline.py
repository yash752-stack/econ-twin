"""
tests/test_pipeline.py
Unit tests for pipeline/data_pipeline.py outputs.

Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import json

DATA_DIR = "data/processed"

# ── FIXTURES ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def econ_df():
    path = f"{DATA_DIR}/economic_data.csv"
    if not os.path.exists(path):
        pytest.skip("Run python run_all.py first to generate data/processed/")
    return pd.read_csv(path)

@pytest.fixture(scope="module")
def trade_df():
    path = f"{DATA_DIR}/trade_data.csv"
    if not os.path.exists(path):
        pytest.skip("Run python run_all.py first")
    return pd.read_csv(path)

@pytest.fixture(scope="module")
def comm_df():
    path = f"{DATA_DIR}/commodity_data.csv"
    if not os.path.exists(path):
        pytest.skip("Run python run_all.py first")
    return pd.read_csv(path)


# ── SCHEMA TESTS ─────────────────────────────────────────────────────────────

class TestEconomicDataSchema:
    REQUIRED_COLS = [
        "iso3", "country", "region", "year",
        "gdp_usd", "gdp_growth_pct", "gdp_per_capita",
        "inflation_pct", "unemployment_pct", "interest_rate_pct",
        "exports_usd", "imports_usd", "trade_balance_usd",
        "current_account_pct_gdp", "govt_debt_pct_gdp",
        "population", "oil_dependency_score",
    ]

    def test_required_columns_present(self, econ_df):
        missing = [c for c in self.REQUIRED_COLS if c not in econ_df.columns]
        assert not missing, f"Missing columns: {missing}"

    def test_row_count(self, econ_df):
        # 30 countries × 14 years = 420
        assert len(econ_df) == 420, f"Expected 420 rows, got {len(econ_df)}"

    def test_no_missing_values(self, econ_df):
        null_counts = econ_df[self.REQUIRED_COLS].isnull().sum()
        nulls = null_counts[null_counts > 0]
        assert len(nulls) == 0, f"Unexpected nulls:\n{nulls}"

    def test_country_count(self, econ_df):
        assert econ_df["iso3"].nunique() == 30, f"Expected 30 countries"

    def test_year_range(self, econ_df):
        assert econ_df["year"].min() == 2010
        assert econ_df["year"].max() == 2023

    def test_iso3_format(self, econ_df):
        bad = econ_df[econ_df["iso3"].str.len() != 3]["iso3"].unique()
        assert len(bad) == 0, f"Non-3-char ISO3 codes: {bad}"


class TestEconomicDataValues:

    def test_gdp_positive(self, econ_df):
        assert (econ_df["gdp_usd"] > 0).all(), "All GDP values must be positive"

    def test_gdp_per_capita_positive(self, econ_df):
        assert (econ_df["gdp_per_capita"] > 0).all()

    def test_population_positive(self, econ_df):
        assert (econ_df["population"] > 0).all()

    def test_unemployment_non_negative(self, econ_df):
        assert (econ_df["unemployment_pct"] >= 0).all(), "Unemployment cannot be negative"

    def test_inflation_reasonable_range(self, econ_df):
        # Allow for hyperinflation outliers (Argentina, Turkey) but cap at 200%
        assert econ_df["inflation_pct"].max() < 200, "Inflation > 200% is unrealistic"
        assert econ_df["inflation_pct"].min() > -20, "Deflation < -20% is unrealistic"

    def test_gdp_growth_reasonable_range(self, econ_df):
        # COVID 2020 worst case ~ -10%; boom max ~ +15%
        assert econ_df["gdp_growth_pct"].min() > -15
        assert econ_df["gdp_growth_pct"].max() < 20

    def test_oil_dependency_bounded(self, econ_df):
        assert (econ_df["oil_dependency_score"] >= 0).all()
        assert (econ_df["oil_dependency_score"] <= 1).all()

    def test_trade_balance_formula(self, econ_df):
        computed = econ_df["exports_usd"] - econ_df["imports_usd"]
        delta = (econ_df["trade_balance_usd"] - computed).abs()
        assert (delta < 1e6).all(), "trade_balance_usd ≠ exports - imports"

    def test_covid_2020_gdp_shock(self, econ_df):
        """2020 should show widespread negative growth due to COVID."""
        avg_2020 = econ_df[econ_df["year"] == 2020]["gdp_growth_pct"].mean()
        assert avg_2020 < 0, f"Expected negative avg GDP growth in 2020, got {avg_2020:.2f}"

    def test_major_economies_present(self, econ_df):
        required = {"USA", "CHN", "DEU", "JPN", "IND", "GBR"}
        present = set(econ_df["iso3"].unique())
        missing = required - present
        assert not missing, f"Major economies missing: {missing}"


class TestTradeData:

    def test_trade_schema(self, trade_df):
        for col in ["exporter", "importer", "year", "trade_value_usd"]:
            assert col in trade_df.columns, f"Missing column: {col}"

    def test_trade_values_positive(self, trade_df):
        assert (trade_df["trade_value_usd"] > 0).all()

    def test_no_self_trade(self, trade_df):
        self_trades = trade_df[trade_df["exporter"] == trade_df["importer"]]
        assert len(self_trades) == 0, "Found self-trade entries"

    def test_trade_year_range(self, trade_df):
        assert trade_df["year"].min() >= 2010
        assert trade_df["year"].max() <= 2023

    def test_us_china_trade_exists(self, trade_df):
        link = trade_df[(trade_df["exporter"] == "CHN") & (trade_df["importer"] == "USA")]
        assert len(link) > 0, "CHN→USA trade link missing"


class TestCommodityData:

    def test_commodity_schema(self, comm_df):
        for col in ["year", "oil_brent", "gold", "copper", "wheat", "natural_gas"]:
            assert col in comm_df.columns, f"Missing column: {col}"

    def test_commodity_values_positive(self, comm_df):
        for col in ["oil_brent", "gold", "copper", "wheat"]:
            assert (comm_df[col] > 0).all(), f"{col} has non-positive values"

    def test_commodity_year_count(self, comm_df):
        assert len(comm_df) == 14, f"Expected 14 year rows, got {len(comm_df)}"

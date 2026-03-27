"""
tests/test_shock_engine.py
Unit tests for simulation/shock_engine.py outputs.

Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import json
import os

DATA_DIR = "data/processed"


@pytest.fixture(scope="module")
def shock_df():
    path = f"{DATA_DIR}/shock_results.csv"
    if not os.path.exists(path):
        pytest.skip("Run python run_all.py first")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def scenario_summaries():
    path = f"{DATA_DIR}/scenario_summaries.json"
    if not os.path.exists(path):
        pytest.skip("Run python run_all.py first")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def propagation():
    path = f"{DATA_DIR}/propagation_results.json"
    if not os.path.exists(path):
        pytest.skip("Run python run_all.py first")
    with open(path) as f:
        return json.load(f)


EXPECTED_SCENARIOS = [
    "oil_price_surge_40", "oil_price_crash_50", "us_rate_hike_200bps",
    "china_slowdown_hard", "global_supply_chain",
    "trade_war_us_china", "eurozone_debt_crisis", "pandemic_shock",
]


class TestShockResultsSchema:

    def test_all_scenarios_present(self, shock_df):
        present = shock_df["scenario"].unique()
        for sc in EXPECTED_SCENARIOS:
            assert sc in present, f"Scenario '{sc}' missing from results"

    def test_required_columns(self, shock_df):
        for col in ["iso3", "scenario", "base_gdp_growth", "shocked_gdp_growth",
                    "gdp_growth_delta", "base_inflation", "shocked_inflation",
                    "inflation_delta", "recession_risk"]:
            assert col in shock_df.columns, f"Missing column: {col}"

    def test_row_count(self, shock_df):
        # 8 scenarios × 30 countries = 240
        assert len(shock_df) == 240, f"Expected 240 rows, got {len(shock_df)}"

    def test_recession_risk_values(self, shock_df):
        valid = {"HIGH", "MEDIUM", "LOW"}
        actual = set(shock_df["recession_risk"].unique())
        assert actual <= valid, f"Invalid recession_risk values: {actual - valid}"


class TestShockEconomics:

    def test_pandemic_most_severe(self, shock_df):
        """Pandemic should be the most severe shock — most countries in recession."""
        by_scenario = shock_df.groupby("scenario").apply(
            lambda g: (g["shocked_gdp_growth"] < 0).sum()
        )
        worst = by_scenario.idxmax()
        assert worst == "pandemic_shock", f"Expected pandemic_shock to be worst, got {worst}"

    def test_oil_exporters_benefit_from_surge(self, shock_df):
        """Saudi Arabia (oil exporter) should see positive GDP impact from oil surge."""
        sau = shock_df[(shock_df["iso3"] == "SAU") & (shock_df["scenario"] == "oil_price_surge_40")]
        if len(sau):
            assert sau["gdp_growth_delta"].values[0] >= 0, \
                "SAU should benefit from oil price surge"

    def test_oil_importers_hurt_by_surge(self, shock_df):
        """Japan (heavy oil importer) should see negative GDP impact from oil surge."""
        jpn = shock_df[(shock_df["iso3"] == "JPN") & (shock_df["scenario"] == "oil_price_surge_40")]
        if len(jpn):
            assert jpn["gdp_growth_delta"].values[0] <= 0, \
                "JPN should be hurt by oil price surge"

    def test_china_shock_affects_trading_partners(self, shock_df):
        """China hard landing should spill over to its major trading partners."""
        china_sc = shock_df[shock_df["scenario"] == "china_slowdown_hard"]
        # Korea, Australia, Germany all heavily trade with China
        trading_partners = ["KOR", "AUS", "DEU"]
        for iso3 in trading_partners:
            row = china_sc[china_sc["iso3"] == iso3]
            if len(row):
                assert row["gdp_growth_delta"].values[0] <= 0, \
                    f"{iso3} should be negatively affected by China hard landing"

    def test_shocked_gdp_consistency(self, shock_df):
        """shocked_gdp_growth should equal base + delta."""
        delta_check = (shock_df["shocked_gdp_growth"] - shock_df["base_gdp_growth"] - shock_df["gdp_growth_delta"]).abs()
        assert (delta_check < 0.01).all(), "shocked_gdp_growth ≠ base + delta"

    def test_inflation_non_negative_after_shock(self, shock_df):
        assert (shock_df["shocked_inflation"] >= 0).all(), \
            "Inflation cannot go negative after shock"

    def test_direct_impact_countries_harder_hit(self, shock_df):
        """Direct-impact countries should on average have larger magnitude delta."""
        for sc in ["us_rate_hike_200bps", "china_slowdown_hard", "eurozone_debt_crisis"]:
            sc_data = shock_df[shock_df["scenario"] == sc]
            direct = sc_data[sc_data["is_direct_impact"] == True]["gdp_growth_delta"].abs().mean()
            indirect = sc_data[sc_data["is_direct_impact"] == False]["gdp_growth_delta"].abs().mean()
            if not (sc_data["is_direct_impact"] == True).any():
                continue  # global scenario
            assert direct >= indirect, f"[{sc}] Direct impact ({direct:.3f}) < indirect ({indirect:.3f})"


class TestScenarioSummaries:

    def test_all_scenarios_summarised(self, scenario_summaries):
        for sc in EXPECTED_SCENARIOS:
            assert sc in scenario_summaries, f"'{sc}' missing from summaries"

    def test_summary_keys(self, scenario_summaries):
        for key, val in scenario_summaries.items():
            for field in ["name", "countries_in_recession", "avg_gdp_growth_delta",
                          "avg_inflation_delta", "global_severity"]:
                assert field in val, f"Field '{field}' missing in summary for {key}"

    def test_severity_values(self, scenario_summaries):
        valid = {"SEVERE", "MODERATE", "MILD"}
        for k, v in scenario_summaries.items():
            assert v["global_severity"] in valid, \
                f"Invalid severity '{v['global_severity']}' for {k}"

    def test_recession_count_bounds(self, scenario_summaries):
        for k, v in scenario_summaries.items():
            assert 0 <= v["countries_in_recession"] <= 30, \
                f"{k}: countries_in_recession={v['countries_in_recession']} out of [0, 30]"


class TestPropagation:

    def test_propagation_keys_present(self, propagation):
        expected = ["USA_recession_3pct", "CHN_slowdown_4pct", "SAU_oil_shock_5pct"]
        for k in expected:
            assert k in propagation, f"Propagation key '{k}' missing"

    def test_origin_country_most_affected(self, propagation):
        """The origin country should receive the largest absolute shock."""
        checks = [("USA_recession_3pct", "USA"), ("CHN_slowdown_4pct", "CHN")]
        for key, origin in checks:
            data = propagation[key]
            if origin in data:
                max_shock = max(abs(v) for v in data.values())
                origin_shock = abs(data[origin])
                assert origin_shock == max_shock, \
                    f"[{key}] Origin {origin} not most affected: {origin_shock:.3f} vs max {max_shock:.3f}"

    def test_propagation_values_are_numeric(self, propagation):
        for key, data in propagation.items():
            for country, val in data.items():
                assert isinstance(val, (int, float)), \
                    f"[{key}][{country}] non-numeric value: {val}"

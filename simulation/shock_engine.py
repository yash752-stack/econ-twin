"""
simulation/shock_engine.py
Economic Shock Propagation Engine
Simulates oil price shocks, rate hikes, trade wars, supply chain collapses.
Each shock propagates through the trade network with cascade effects.

Run: python simulation/shock_engine.py
"""

import pandas as pd
import numpy as np
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List

print("=" * 65)
print("  GLOBAL ECONOMIC DIGITAL TWIN — Shock Engine v1.0")
print("  yash752-stack | Module 3: Economic Shock Simulator")
print("=" * 65)

os.makedirs("data/processed", exist_ok=True)

econ_df  = pd.read_csv("data/processed/economic_data.csv")
trade_df = pd.read_csv("data/processed/trade_data.csv")
comm_df  = pd.read_csv("data/processed/commodity_data.csv")
metrics_df = pd.read_csv("data/processed/network_metrics.csv")

BASE_YEAR = 2023
econ_base = econ_df[econ_df["year"] == 2022].set_index("iso3").copy()

@dataclass
class ShockScenario:
    name: str
    description: str
    shocks: Dict  # {parameter: magnitude}
    affected_countries: List[str] = field(default_factory=list)  # [] = global

SHOCK_LIBRARY = {
    "oil_price_surge_40": ShockScenario(
        name="Oil Price Surge +40%",
        description="Oil price increases 40% due to geopolitical conflict in Middle East",
        shocks={"oil_price_pct": 40},
    ),
    "oil_price_crash_50": ShockScenario(
        name="Oil Price Crash -50%",
        description="Oil demand collapse drives prices down 50%",
        shocks={"oil_price_pct": -50},
    ),
    "us_rate_hike_200bps": ShockScenario(
        name="US Rate Hike +200bps",
        description="Federal Reserve raises rates by 200 basis points",
        shocks={"interest_rate_hike_bps": 200},
        affected_countries=["USA"],
    ),
    "china_slowdown_hard": ShockScenario(
        name="China Hard Landing",
        description="China GDP growth drops to 1% due to real estate collapse",
        shocks={"gdp_growth_shock_pct": -4.5},
        affected_countries=["CHN"],
    ),
    "global_supply_chain": ShockScenario(
        name="Global Supply Chain Disruption",
        description="Major ports closed, shipping costs surge 300%",
        shocks={"trade_volume_shock_pct": -25, "inflation_shock_pct": 2.5},
    ),
    "trade_war_us_china": ShockScenario(
        name="US-China Trade War (Full Decoupling)",
        description="USA and China impose 100% tariffs on all bilateral trade",
        shocks={"bilateral_trade_shock_pct": -80},
        affected_countries=["USA", "CHN"],
    ),
    "eurozone_debt_crisis": ShockScenario(
        name="Eurozone Debt Crisis",
        description="Southern European sovereign debt crisis spreads to core Europe",
        shocks={"gdp_growth_shock_pct": -2.5, "interest_rate_hike_bps": 150},
        affected_countries=["ESP", "FRA", "DEU", "BEL", "NLD"],
    ),
    "pandemic_shock": ShockScenario(
        name="Global Pandemic (COVID-scale)",
        description="Global pandemic causes -5% GDP shock universally",
        shocks={"gdp_growth_shock_pct": -5.0, "trade_volume_shock_pct": -20},
    ),
}

# ── TRANSMISSION COEFFICIENTS ────────────────────────────────────────────────
# How each shock transmits to macro variables
TRANSMISSION = {
    "oil_price_pct": {
        "inflation_pct":        0.08,   # 1% oil → 0.08% inflation
        "gdp_growth_pct":      -0.03,   # 1% oil → -0.03% GDP
        "interest_rate_pct":    0.04,   # central banks respond
        "unemployment_pct":     0.015,  # higher costs → job losses
    },
    "interest_rate_hike_bps": {
        "gdp_growth_pct":      -0.025,  # per 10bps
        "inflation_pct":       -0.015,
        "unemployment_pct":     0.010,
        "exports_usd":         -0.008,  # currency appreciation
    },
    "gdp_growth_shock_pct": {
        "unemployment_pct":    -0.40,   # Okun's law
        "inflation_pct":       -0.30,   # demand-pull
        "exports_usd":         -0.15,
        "imports_usd":         -0.20,
    },
    "trade_volume_shock_pct": {
        "gdp_growth_pct":      -0.20,
        "inflation_pct":        0.50,   # supply disruption → prices up
        "unemployment_pct":     0.10,
    },
    "bilateral_trade_shock_pct": {
        "gdp_growth_pct":      -0.15,
        "exports_usd":         -0.25,
        "imports_usd":         -0.25,
        "inflation_pct":        0.30,
    },
}

# ── OIL DEPENDENCY BY COUNTRY ────────────────────────────────────────────────
OIL_EXPORTERS = {"SAU": 1.0, "RUS": 0.85, "NOR": 0.7, "ARE": 0.9, "CAN": 0.4}
OIL_DEPENDENCY = {
    "USA": 0.35, "CHN": 0.45, "DEU": 0.55, "JPN": 0.70, "IND": 0.60,
    "GBR": 0.40, "FRA": 0.50, "BRA": 0.30, "CAN": 0.20, "RUS": -0.8,
    "KOR": 0.75, "AUS": 0.30, "ESP": 0.60, "MEX": 0.25, "IDN": 0.40,
    "NLD": 0.50, "SAU": -1.0, "TUR": 0.65, "CHE": 0.45, "POL": 0.55,
    "SWE": 0.35, "BEL": 0.50, "ARG": 0.30, "NOR": -0.7, "ARE": -0.9,
    "ZAF": 0.45, "SGP": 0.60, "MYS": 0.35, "THA": 0.55, "EGY": 0.40,
}

def run_shock(scenario_key: str, custom_shocks: Dict = None) -> pd.DataFrame:
    """
    Run a shock scenario and return country-level impacts.
    """
    scenario = SHOCK_LIBRARY.get(scenario_key)
    if not scenario and not custom_shocks:
        raise ValueError(f"Unknown scenario: {scenario_key}")

    shocks = custom_shocks if custom_shocks else scenario.shocks
    affected = scenario.affected_countries if scenario else []

    results = []
    for iso3 in econ_base.index:
        base = econ_base.loc[iso3]
        impact = {
            "gdp_growth_pct": 0, "inflation_pct": 0,
            "unemployment_pct": 0, "interest_rate_pct": 0,
            "exports_usd": 0, "imports_usd": 0
        }

        # Is this country directly shocked?
        is_direct = (not affected) or (iso3 in affected)

        # Trade linkage factor (how connected to shocked country)
        trade_linkage = 1.0
        if affected and not is_direct:
            country_metrics = metrics_df[metrics_df["iso3"] == iso3]
            if not country_metrics.empty:
                top_exp = country_metrics.iloc[0]["top_export_partner"]
                top_imp = country_metrics.iloc[0]["top_import_partner"]
                if top_exp in affected or top_imp in affected:
                    trade_linkage = 0.45
                else:
                    trade_linkage = 0.15

        for shock_type, magnitude in shocks.items():
            transmission = TRANSMISSION.get(shock_type, {})
            scale = 1.0 if is_direct else trade_linkage

            # Special case: oil shock scaled by dependency
            if shock_type == "oil_price_pct":
                oil_dep = OIL_DEPENDENCY.get(iso3, 0.4)
                if oil_dep < 0:  # exporter — benefits
                    scale = abs(oil_dep) * 0.5
                    for var, coef in transmission.items():
                        impact[var] = impact.get(var, 0) - (coef * magnitude * scale)
                else:
                    scale = oil_dep
                    for var, coef in transmission.items():
                        impact[var] = impact.get(var, 0) + (coef * magnitude * scale)

            elif shock_type == "interest_rate_hike_bps":
                bps_in_pct = magnitude / 10
                for var, coef in transmission.items():
                    impact[var] = impact.get(var, 0) + (coef * bps_in_pct * scale)

            else:
                for var, coef in transmission.items():
                    impact[var] = impact.get(var, 0) + (coef * magnitude * scale)

        # Apply impacts to base values
        new_gdp_growth = base["gdp_growth_pct"] + impact.get("gdp_growth_pct", 0)
        new_inflation  = max(0, base["inflation_pct"] + impact.get("inflation_pct", 0))
        new_unemp      = max(0, base["unemployment_pct"] + impact.get("unemployment_pct", 0))
        new_int_rate   = max(0, base["interest_rate_pct"] + impact.get("interest_rate_pct", 0))
        new_exports    = base["exports_usd"] * (1 + impact.get("exports_usd", 0)/100)
        new_imports    = base["imports_usd"] * (1 + impact.get("imports_usd", 0)/100)

        results.append({
            "iso3":                iso3,
            "country":             base["country"],
            "region":              base["region"],
            "scenario":            scenario_key,
            "is_direct_impact":    is_direct,
            "base_gdp_growth":     round(base["gdp_growth_pct"], 2),
            "shocked_gdp_growth":  round(new_gdp_growth, 2),
            "gdp_growth_delta":    round(impact.get("gdp_growth_pct", 0), 2),
            "base_inflation":      round(base["inflation_pct"], 2),
            "shocked_inflation":   round(new_inflation, 2),
            "inflation_delta":     round(impact.get("inflation_pct", 0), 2),
            "base_unemployment":   round(base["unemployment_pct"], 2),
            "shocked_unemployment":round(new_unemp, 2),
            "unemployment_delta":  round(impact.get("unemployment_pct", 0), 2),
            "base_interest_rate":  round(base["interest_rate_pct"], 2),
            "shocked_interest_rate":round(new_int_rate, 2),
            "recession_risk":      "HIGH" if new_gdp_growth < 0 else "MEDIUM" if new_gdp_growth < 1 else "LOW",
        })

    return pd.DataFrame(results)

# ── RUN ALL SCENARIOS ────────────────────────────────────────────────────────
print("\n[1/2] Running all shock scenarios...")

all_results = []
scenario_summaries = {}

for key, scenario in SHOCK_LIBRARY.items():
    df = run_shock(key)
    all_results.append(df)
    in_recession = (df["shocked_gdp_growth"] < 0).sum()
    avg_gdp_delta = df["gdp_growth_delta"].mean()
    avg_inf_delta = df["inflation_delta"].mean()
    scenario_summaries[key] = {
        "name": scenario.name,
        "countries_in_recession": int(in_recession),
        "avg_gdp_growth_delta": round(avg_gdp_delta, 3),
        "avg_inflation_delta": round(avg_inf_delta, 3),
        "global_severity": "SEVERE" if in_recession > 15 else "MODERATE" if in_recession > 7 else "MILD",
    }
    print(f"      ✅ {scenario.name[:45]:45s} → {in_recession:2d} countries in recession")

all_results_df = pd.concat(all_results)
all_results_df.to_csv("data/processed/shock_results.csv", index=False)

with open("data/processed/scenario_summaries.json", "w") as f:
    json.dump(scenario_summaries, f, indent=2)

print("\n[2/2] Shock Engine Summary")
print("-" * 55)
for key, summary in scenario_summaries.items():
    sev = summary["global_severity"]
    n   = summary["countries_in_recession"]
    print(f"  {summary['name'][:42]:42s} [{sev:8s}] {n:2d} in recession")
print("-" * 55)
print("\n✅ Module 3 complete — All shock scenarios computed\n")

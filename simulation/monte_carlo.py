"""
simulation/monte_carlo.py
Monte Carlo Macroeconomic Simulation
Runs 10,000 simulated economic futures to generate:
- Recession probability distributions
- Inflation fan charts
- GDP growth confidence intervals

Run: python simulation/monte_carlo.py
"""

import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

print("=" * 65)
print("  GLOBAL ECONOMIC DIGITAL TWIN — Monte Carlo v1.0")
print("  yash752-stack | Module 4: Probabilistic Forecasting")
print("=" * 65)

os.makedirs("data/processed", exist_ok=True)
np.random.seed(42)

econ_df = pd.read_csv("data/processed/economic_data.csv")
BASE_YEAR = 2022
N_SIMS   = 10000
HORIZON  = 5  # years ahead

econ_base = econ_df[econ_df["year"] == BASE_YEAR].set_index("iso3")

# ── SHOCK PROBABILITY CALIBRATION ───────────────────────────────────────────
# Based on historical frequency of major economic shocks
SHOCK_PROBS = {
    "mild_recession":     0.25,  # ~1 in 4 years globally
    "severe_recession":   0.08,  # ~1 in 12 years
    "oil_spike_40pct":    0.15,
    "financial_crisis":   0.05,
    "pandemic":           0.03,
    "trade_war":          0.10,
    "soft_landing":       0.35,  # central bank success
}

COUNTRY_VOLATILITY = {
    "USA": 0.8,  "CHN": 1.0,  "DEU": 0.7,  "JPN": 0.6,  "IND": 1.2,
    "GBR": 0.9,  "FRA": 0.8,  "BRA": 1.8,  "CAN": 0.8,  "RUS": 2.0,
    "KOR": 1.0,  "AUS": 0.8,  "ESP": 1.1,  "MEX": 1.3,  "IDN": 1.2,
    "NLD": 0.7,  "SAU": 1.4,  "TUR": 2.5,  "CHE": 0.5,  "POL": 1.0,
    "SWE": 0.7,  "BEL": 0.7,  "ARG": 3.5,  "NOR": 0.8,  "ARE": 1.1,
    "ZAF": 1.5,  "SGP": 0.9,  "MYS": 1.1,  "THA": 1.2,  "EGY": 1.8,
}

print(f"\n[1/4] Running {N_SIMS:,} Monte Carlo simulations × {HORIZON} years...")

# ── SIMULATION ENGINE ────────────────────────────────────────────────────────
def draw_global_shock(year_idx):
    """Draw global shock regime for a given year"""
    r = np.random.random()
    if r < SHOCK_PROBS["pandemic"]:
        return {"gdp_shock": np.random.normal(-4.5, 1.5), "inf_shock": np.random.normal(-0.5, 0.5), "label": "pandemic"}
    elif r < SHOCK_PROBS["pandemic"] + SHOCK_PROBS["financial_crisis"]:
        return {"gdp_shock": np.random.normal(-3.0, 1.0), "inf_shock": np.random.normal(-1.0, 0.5), "label": "financial_crisis"}
    elif r < SHOCK_PROBS["pandemic"] + SHOCK_PROBS["financial_crisis"] + SHOCK_PROBS["severe_recession"]:
        return {"gdp_shock": np.random.normal(-2.0, 0.8), "inf_shock": np.random.normal(-0.5, 0.3), "label": "severe_recession"}
    elif r < 0.35:
        return {"gdp_shock": np.random.normal(-0.8, 0.4), "inf_shock": np.random.normal(0.2, 0.2), "label": "mild_recession"}
    elif r < 0.60:
        return {"gdp_shock": np.random.normal(0, 0.3),    "inf_shock": np.random.normal(0, 0.2),   "label": "soft_landing"}
    else:
        return {"gdp_shock": np.random.normal(0.5, 0.4),  "inf_shock": np.random.normal(0.3, 0.2), "label": "expansion"}

# Run simulations
sim_results = {}
for iso3 in econ_base.index:
    base_growth    = econ_base.loc[iso3, "gdp_growth_pct"]
    base_inflation = econ_base.loc[iso3, "inflation_pct"]
    base_unemp     = econ_base.loc[iso3, "unemployment_pct"]
    vol            = COUNTRY_VOLATILITY.get(iso3, 1.0)

    gdp_paths   = np.zeros((N_SIMS, HORIZON))
    inf_paths   = np.zeros((N_SIMS, HORIZON))
    unemp_paths = np.zeros((N_SIMS, HORIZON))
    shock_labels= []

    for sim in range(N_SIMS):
        g = base_growth
        inf = base_inflation
        u   = base_unemp
        sim_shocks = []

        for yr in range(HORIZON):
            global_shock = draw_global_shock(yr)
            sim_shocks.append(global_shock["label"])

            # Mean reversion + global shock + idiosyncratic noise
            g   = 0.6 * g + 0.4 * 2.5 + global_shock["gdp_shock"] * vol + np.random.normal(0, 0.6 * vol)
            inf = 0.7 * inf + 0.3 * 2.5 + global_shock["inf_shock"] * vol + np.random.normal(0, 0.5 * vol)
            inf = max(0.1, inf)
            u   = max(1.0, u - 0.35 * (g - 2.0) + np.random.normal(0, 0.3 * vol))

            gdp_paths[sim, yr]   = g
            inf_paths[sim, yr]   = inf
            unemp_paths[sim, yr] = u

        shock_labels.append(sim_shocks)

    # Compute percentiles
    sim_results[iso3] = {
        "gdp_growth": {
            "p10":    gdp_paths.mean(axis=1) if False else [round(x, 2) for x in np.percentile(gdp_paths, 10, axis=0).tolist()],
            "p25":    [round(x, 2) for x in np.percentile(gdp_paths, 25, axis=0).tolist()],
            "p50":    [round(x, 2) for x in np.percentile(gdp_paths, 50, axis=0).tolist()],
            "p75":    [round(x, 2) for x in np.percentile(gdp_paths, 75, axis=0).tolist()],
            "p90":    [round(x, 2) for x in np.percentile(gdp_paths, 90, axis=0).tolist()],
            "mean":   [round(x, 2) for x in gdp_paths.mean(axis=0).tolist()],
        },
        "inflation": {
            "p10":    [round(x, 2) for x in np.percentile(inf_paths, 10, axis=0).tolist()],
            "p25":    [round(x, 2) for x in np.percentile(inf_paths, 25, axis=0).tolist()],
            "p50":    [round(x, 2) for x in np.percentile(inf_paths, 50, axis=0).tolist()],
            "p75":    [round(x, 2) for x in np.percentile(inf_paths, 75, axis=0).tolist()],
            "p90":    [round(x, 2) for x in np.percentile(inf_paths, 90, axis=0).tolist()],
            "mean":   [round(x, 2) for x in inf_paths.mean(axis=0).tolist()],
        },
        "unemployment": {
            "p50":    [round(x, 2) for x in np.percentile(unemp_paths, 50, axis=0).tolist()],
            "mean":   [round(x, 2) for x in unemp_paths.mean(axis=0).tolist()],
        },
        "recession_probability": [
            round(float((gdp_paths[:, yr] < 0).mean()), 3)
            for yr in range(HORIZON)
        ],
        "severe_recession_probability": [
            round(float((gdp_paths[:, yr] < -2).mean()), 3)
            for yr in range(HORIZON)
        ],
        "high_inflation_probability": [
            round(float((inf_paths[:, yr] > 5).mean()), 3)
            for yr in range(HORIZON)
        ],
        "base_values": {
            "gdp_growth": round(base_growth, 2),
            "inflation":  round(base_inflation, 2),
            "unemployment": round(base_unemp, 2),
        }
    }

print(f"      ✅ Simulations complete")

# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
print("\n[2/4] Saving simulation results...")
with open("data/processed/monte_carlo_results.json", "w") as f:
    json.dump(sim_results, f)
print("      ✅ data/processed/monte_carlo_results.json")

# ── GLOBAL RECESSION ANALYSIS ─────────────────────────────────────────────────
print("\n[3/4] Computing global recession metrics...")

recession_rows = []
years_ahead = [BASE_YEAR + i + 1 for i in range(HORIZON)]
for iso3, data in sim_results.items():
    for yr_idx, year in enumerate(years_ahead):
        recession_rows.append({
            "iso3": iso3,
            "year": year,
            "gdp_growth_p50": data["gdp_growth"]["p50"][yr_idx],
            "gdp_growth_p10": data["gdp_growth"]["p10"][yr_idx],
            "gdp_growth_p90": data["gdp_growth"]["p90"][yr_idx],
            "inflation_p50":  data["inflation"]["p50"][yr_idx],
            "inflation_p10":  data["inflation"]["p10"][yr_idx],
            "inflation_p90":  data["inflation"]["p90"][yr_idx],
            "recession_prob": data["recession_probability"][yr_idx],
            "high_inf_prob":  data["high_inflation_probability"][yr_idx],
        })

recession_df = pd.DataFrame(recession_rows)
recession_df.to_csv("data/processed/recession_probabilities.csv", index=False)

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n[4/4] Monte Carlo Summary (Year +1 Recession Probabilities)")
print("-" * 55)
yr1 = recession_df[recession_df["year"] == years_ahead[0]].copy()
yr1_sorted = yr1.nlargest(10, "recession_prob")
for _, row in yr1_sorted.iterrows():
    bar = "█" * int(row["recession_prob"] * 20)
    print(f"  {row['iso3']:4s} {bar:20s} {row['recession_prob']*100:5.1f}% recession risk")
print("-" * 55)
print(f"\n  Simulations run: {N_SIMS:,}")
print(f"  Forecast horizon: {HORIZON} years ({BASE_YEAR+1}–{BASE_YEAR+HORIZON})")
print(f"  Countries covered: {len(sim_results)}")
print("\n✅ Module 4 complete — Monte Carlo results saved\n")

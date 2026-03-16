# 🌍 Global Economic Digital Twin
### AI Macro Simulation Engine

> A macroeconomic simulation platform modelling 30 countries, global trade shock propagation, Monte Carlo probabilistic forecasting, and XGBoost-based GDP/inflation prediction — the type of system used by the IMF, World Bank, and Federal Reserve.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-ML_Forecasting-189AB4?style=flat)
![NetworkX](https://img.shields.io/badge/NetworkX-Trade_Graph-orange?style=flat)

---

## What This Does

Ask questions like:
- **What happens to global GDP if oil prices surge 40%?**
- **Which countries enter recession if China hard-lands?**
- **What's the probability of a US recession in 2025?**
- **How does a trade war propagate through the supply chain network?**

The system answers with country-level forecasts, probability distributions, and network-based shock propagation analysis.

---

## Architecture

```
World Bank / IMF Data
        │
        ▼
Data Warehouse (SQLite)
GDP, Inflation, Trade, Commodities — 30 countries × 14 years
        │
        ▼
Trade Network Graph (NetworkX)
Countries = nodes │ Trade flows = weighted edges
PageRank, Betweenness, Vulnerability scores
        │
        ▼
Economic Shock Engine
Oil shocks │ Rate hikes │ Trade wars │ Pandemics
Cascade propagation through trade network
        │
        ▼
Monte Carlo Simulation (10,000 scenarios)
Recession probabilities │ Inflation fan charts │ GDP distributions
        │
        ▼
XGBoost ML Forecasting
GDP growth │ Inflation │ 2024–2027 forecasts
        │
        ▼
Interactive Streamlit Dashboard
Scenario Explorer │ Network Visualiser │ Fan Charts
```

---

## Quick Start

```bash
git clone https://github.com/yash752-stack/econ-twin.git
cd econ-twin
pip install -r requirements.txt

# Run full pipeline (all 5 modules)
python run_all.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Modules

| Module | File | Description |
|--------|------|-------------|
| 1 | `pipeline/data_pipeline.py` | World Bank data pipeline → SQLite warehouse |
| 2 | `network/trade_network.py` | Trade network graph + centrality metrics |
| 3 | `simulation/shock_engine.py` | 8 shock scenarios with cascade propagation |
| 4 | `simulation/monte_carlo.py` | 10,000 simulated futures per country |
| 5 | `ml/forecasting.py` | XGBoost GDP + inflation forecasting |
| 6 | `dashboard/app.py` | Interactive Streamlit scenario explorer |

---

## Shock Scenarios

| Scenario | Type | Severity |
|----------|------|----------|
| Oil Price Surge +40% | Commodity | Moderate |
| Oil Price Crash -50% | Commodity | Mild |
| US Rate Hike +200bps | Monetary | Moderate |
| China Hard Landing | Growth | Severe |
| Global Supply Chain Disruption | Trade | Severe |
| US-China Trade War (Decoupling) | Trade | Severe |
| Eurozone Debt Crisis | Financial | Moderate |
| Global Pandemic | Systemic | Extreme |

---

## Author

**Yash Chaudhary** — github.com/yash752-stack  
Built as part of a data analytics portfolio targeting institutional-grade modelling roles.

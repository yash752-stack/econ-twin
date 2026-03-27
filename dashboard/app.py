"""
dashboard/app.py
Global Economic Digital Twin — Interactive Streamlit Dashboard

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Global Economic Digital Twin",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.section-header {
    font-size: 18px; font-weight: 600; color: #c0c0d0;
    border-bottom: 1px solid #333; padding-bottom: 6px; margin: 1rem 0 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── DATA LOADERS ─────────────────────────────────────────────────────────────
DATA_DIR = "data/processed"

@st.cache_data
def load_economic_data():
    return pd.read_csv(f"{DATA_DIR}/economic_data.csv")

@st.cache_data
def load_forecasts():
    return pd.read_csv(f"{DATA_DIR}/ml_forecasts.csv")

@st.cache_data
def load_shock_results():
    return pd.read_csv(f"{DATA_DIR}/shock_results.csv")

@st.cache_data
def load_network_metrics():
    return pd.read_csv(f"{DATA_DIR}/network_metrics.csv")

@st.cache_data
def load_recession_probs():
    return pd.read_csv(f"{DATA_DIR}/recession_probabilities.csv")

@st.cache_data
def load_monte_carlo():
    with open(f"{DATA_DIR}/monte_carlo_results.json") as f:
        return json.load(f)

@st.cache_data
def load_scenario_summaries():
    with open(f"{DATA_DIR}/scenario_summaries.json") as f:
        return json.load(f)

@st.cache_data
def load_propagation():
    with open(f"{DATA_DIR}/propagation_results.json") as f:
        return json.load(f)

@st.cache_data
def load_model_metrics():
    with open(f"{DATA_DIR}/model_metrics.json") as f:
        return json.load(f)

def check_data():
    required = [
        "economic_data.csv", "ml_forecasts.csv", "shock_results.csv",
        "network_metrics.csv", "recession_probabilities.csv",
    ]
    return [f for f in required if not os.path.exists(f"{DATA_DIR}/{f}")]

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 Econ Digital Twin")
    st.caption("Global Macro Simulation Engine")
    st.divider()

    missing = check_data()
    if missing:
        st.error(f"⚠️ Missing data. Run `python run_all.py` first.\n\nMissing: {', '.join(missing)}")
        st.stop()

    page = st.radio(
        "Navigation",
        ["📊 Global Overview", "🔴 Shock Scenarios", "📈 ML Forecasts",
         "🕸 Trade Network", "🎲 Monte Carlo", "🔮 Country Deep Dive"],
        label_visibility="collapsed",
    )
    st.divider()

    econ_df = load_economic_data()
    all_countries = sorted(econ_df["country"].unique())
    iso3_to_name = dict(zip(econ_df["iso3"], econ_df["country"]))
    name_to_iso3 = dict(zip(econ_df["country"], econ_df["iso3"]))

    selected_country = st.selectbox("Focus Country", all_countries,
                                     index=all_countries.index("United States"))
    selected_iso3 = name_to_iso3[selected_country]
    st.caption("📦 [github.com/yash752-stack/econ-twin](https://github.com/yash752-stack/econ-twin)")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — GLOBAL OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Global Overview":
    st.title("📊 Global Economic Overview")
    st.caption("Cross-country macroeconomic indicators — 2022 snapshot")

    econ_df = load_economic_data()
    latest = econ_df[econ_df["year"] == 2022].copy()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("World GDP", f"${latest['gdp_usd'].sum()/1e12:.1f}T")
    col2.metric("Avg GDP Growth", f"{latest['gdp_growth_pct'].mean():.1f}%")
    col3.metric("Avg Inflation", f"{latest['inflation_pct'].mean():.1f}%")
    col4.metric("Avg Unemployment", f"{latest['unemployment_pct'].mean():.1f}%")
    col5.metric("In Recession", int((latest["gdp_growth_pct"] < 0).sum()))

    st.divider()
    st.markdown('<div class="section-header">GDP Growth by Country (2022)</div>', unsafe_allow_html=True)
    fig_map = px.choropleth(
        latest, locations="iso3", color="gdp_growth_pct", hover_name="country",
        hover_data={"inflation_pct": ":.1f", "unemployment_pct": ":.1f", "gdp_growth_pct": ":.2f"},
        color_continuous_scale="RdYlGn", color_continuous_midpoint=0, range_color=[-5, 10],
        labels={"gdp_growth_pct": "GDP Growth %"},
    )
    fig_map.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"),
                           margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", height=400)
    st.plotly_chart(fig_map, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">GDP Growth vs Inflation</div>', unsafe_allow_html=True)
        fig_s = px.scatter(latest, x="gdp_growth_pct", y="inflation_pct", size="gdp_usd",
                            color="region", hover_name="country", size_max=40,
                            labels={"gdp_growth_pct": "GDP Growth %", "inflation_pct": "Inflation %"})
        fig_s.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_s.add_hline(y=2, line_dash="dot", line_color="gray", opacity=0.5,
                        annotation_text="2% target", annotation_position="bottom right")
        fig_s.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_s, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Top 15 Economies — GDP Growth</div>', unsafe_allow_html=True)
        top15 = latest.nlargest(15, "gdp_usd").sort_values("gdp_growth_pct")
        fig_b = go.Figure(go.Bar(
            x=top15["gdp_growth_pct"], y=top15["country"], orientation="h",
            marker_color=["#e05c5c" if v < 0 else "#4caf8a" for v in top15["gdp_growth_pct"]],
            text=[f"{v:.1f}%" for v in top15["gdp_growth_pct"]], textposition="outside",
        ))
        fig_b.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown('<div class="section-header">GDP Growth Heatmap — All Countries × All Years</div>',
                unsafe_allow_html=True)
    pivot = econ_df.pivot_table(index="country", columns="year", values="gdp_growth_pct")
    fig_h = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                                  colorscale="RdYlGn", zmid=0, zmin=-10, zmax=12,
                                  colorbar=dict(title="GDP Growth %")))
    fig_h.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_h, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — SHOCK SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔴 Shock Scenarios":
    st.title("🔴 Economic Shock Scenarios")
    st.caption("8 calibrated shock scenarios with cascade propagation through the trade network")

    shock_df = load_shock_results()
    summaries = load_scenario_summaries()
    propagation = load_propagation()

    SCENARIO_LABELS = {
        "oil_price_surge_40": "Oil Surge +40%",
        "oil_price_crash_50": "Oil Crash -50%",
        "us_rate_hike_200bps": "US Rate Hike +200bps",
        "china_slowdown_hard": "China Hard Landing",
        "global_supply_chain": "Supply Chain Collapse",
        "trade_war_us_china": "US-China Trade War",
        "eurozone_debt_crisis": "Eurozone Debt Crisis",
        "pandemic_shock": "Global Pandemic",
    }

    st.markdown('<div class="section-header">Scenario Severity Summary</div>', unsafe_allow_html=True)
    rows = [{"Scenario": SCENARIO_LABELS.get(k, k),
             "Countries in Recession": s["countries_in_recession"],
             "Avg GDP Δ (pp)": s["avg_gdp_growth_delta"],
             "Avg Inflation Δ (pp)": s["avg_inflation_delta"],
             "Severity": s["global_severity"]} for k, s in summaries.items()]
    sum_df = pd.DataFrame(rows).sort_values("Countries in Recession", ascending=False)

    def color_sev(val):
        m = {"SEVERE": "background-color:#5c1a1a;color:#ff8080",
             "MODERATE": "background-color:#4a3800;color:#ffcc55",
             "MILD": "background-color:#1a3a1a;color:#7ccc7c"}
        return m.get(val, "")

    st.dataframe(sum_df.style.applymap(color_sev, subset=["Severity"])
                      .format({"Avg GDP Δ (pp)": "{:.2f}", "Avg Inflation Δ (pp)": "{:.2f}"}),
                 use_container_width=True, hide_index=True)

    st.divider()
    sel_sc = st.selectbox("Analyse Scenario", list(SCENARIO_LABELS.keys()),
                           format_func=lambda k: SCENARIO_LABELS[k])
    sc_data = shock_df[shock_df["scenario"] == sel_sc].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("In Recession", int((sc_data["shocked_gdp_growth"] < 0).sum()))
    c2.metric("Avg GDP Impact", f"{sc_data['gdp_growth_delta'].mean():.2f}pp")
    c3.metric("Avg Inflation Impact", f"{sc_data['inflation_delta'].mean():.2f}pp")

    st.markdown('<div class="section-header">GDP Growth: Baseline vs Post-Shock</div>', unsafe_allow_html=True)
    top20 = sc_data.nlargest(20, "base_gdp_growth").sort_values("shocked_gdp_growth")
    fig_wf = go.Figure()
    fig_wf.add_trace(go.Bar(name="Baseline", x=top20["iso3"], y=top20["base_gdp_growth"],
                             marker_color="#4a7fb5", opacity=0.7))
    fig_wf.add_trace(go.Bar(name="Post-Shock", x=top20["iso3"], y=top20["shocked_gdp_growth"],
                             marker_color=["#e05c5c" if v < 0 else "#4caf8a" for v in top20["shocked_gdp_growth"]]))
    fig_wf.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
    fig_wf.update_layout(barmode="group", height=380, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", yaxis_title="GDP Growth %")
    st.plotly_chart(fig_wf, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">GDP Shock Impact Map</div>', unsafe_allow_html=True)
        fig_m = px.choropleth(sc_data, locations="iso3", color="gdp_growth_delta", hover_name="iso3",
                               color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                               labels={"gdp_growth_delta": "GDP Δ (pp)"})
        fig_m.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)",
                             geo=dict(showframe=False))
        st.plotly_chart(fig_m, use_container_width=True)
    with col_r:
        st.markdown('<div class="section-header">Shock Propagation</div>', unsafe_allow_html=True)
        prop_key = st.selectbox("Origin", list(propagation.keys()),
                                 format_func=lambda k: k.replace("_", " "))
        prop_df = pd.DataFrame([{"iso3": k, "shock": v} for k, v in propagation[prop_key].items()])
        prop_df = prop_df.sort_values("shock")
        fig_p = go.Figure(go.Bar(
            x=prop_df["shock"], y=prop_df["iso3"], orientation="h",
            marker_color=["#e05c5c" if v < 0 else "#4caf8a" for v in prop_df["shock"]],
        ))
        fig_p.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_p, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML FORECASTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 ML Forecasts":
    st.title("📈 XGBoost Macroeconomic Forecasts")
    st.caption("2024–2027 GDP & inflation forecasts using lag features, rolling stats, commodity prices")

    forecast_df = load_forecasts()
    econ_df = load_economic_data()
    metrics = load_model_metrics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP Model MAE", f"{metrics['gdp_model']['mae']:.2f}pp")
    c2.metric("GDP Model R²", f"{metrics['gdp_model']['r2']:.3f}")
    c3.metric("Inflation MAE", f"{metrics['inflation_model']['mae']:.2f}pp")
    c4.metric("Inflation R²", f"{metrics['inflation_model']['r2']:.3f}")
    st.caption(f"Top GDP predictors: {', '.join(metrics['gdp_model']['top_features'][:3])}")
    st.divider()

    st.markdown('<div class="section-header">Country Forecast with Uncertainty Bands</div>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    fc_country = col_a.selectbox("Country", sorted(forecast_df["country"].unique()),
                                  index=list(sorted(forecast_df["country"].unique())).index("United States"))
    fc_metric = col_b.selectbox("Metric", ["GDP Growth", "Inflation"])

    hist = econ_df[econ_df["country"] == fc_country][["year", "gdp_growth_pct", "inflation_pct"]]
    fc = forecast_df[forecast_df["country"] == fc_country]

    fig_fc = go.Figure()
    if fc_metric == "GDP Growth":
        y_hist, y_fc, y_lo, y_hi = "gdp_growth_pct", "gdp_growth_forecast", "gdp_growth_lower", "gdp_growth_upper"
        band_color, line_color, ylabel = "74,207,138", "#4caf8a", "GDP Growth %"
        fig_fc.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, annotation_text="Recession")
    else:
        y_hist, y_fc, y_lo, y_hi = "inflation_pct", "inflation_forecast", "inflation_lower", "inflation_upper"
        band_color, line_color, ylabel = "224,92,92", "#e05c5c", "Inflation %"
        fig_fc.add_hline(y=2, line_dash="dot", line_color="gray", opacity=0.4, annotation_text="2% target")

    fig_fc.add_trace(go.Scatter(x=hist["year"], y=hist[y_hist], name="Historical",
                                 mode="lines+markers", line=dict(color="#4a7fb5", width=2)))
    fig_fc.add_trace(go.Scatter(
        x=list(fc["year"]) + list(fc["year"])[::-1],
        y=list(fc[y_hi]) + list(fc[y_lo])[::-1],
        fill="toself", fillcolor=f"rgba({band_color},0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="Confidence Band",
    ))
    fig_fc.add_trace(go.Scatter(x=fc["year"], y=fc[y_fc], name="Forecast",
                                 mode="lines+markers", line=dict(color=line_color, width=2, dash="dash"),
                                 marker=dict(symbol="diamond", size=8)))
    fig_fc.add_vrect(x0=2023.5, x1=2027.5, fillcolor="rgba(255,255,255,0.03)",
                      annotation_text="Forecast zone", annotation_position="top left")
    fig_fc.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          yaxis_title=ylabel, xaxis_title="Year")
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown('<div class="section-header">2027 Global GDP Growth Forecast</div>', unsafe_allow_html=True)
    fc_2027 = forecast_df[forecast_df["year"] == 2027]
    fig_fc_map = px.choropleth(fc_2027, locations="iso3", color="gdp_growth_forecast",
                                hover_name="country",
                                hover_data={"inflation_forecast": ":.1f",
                                            "gdp_growth_forecast": ":.2f", "model_confidence": ":.0%"},
                                color_continuous_scale="RdYlGn", color_continuous_midpoint=2,
                                labels={"gdp_growth_forecast": "Forecast GDP Growth %"})
    fig_fc_map.update_layout(height=380, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)",
                              geo=dict(showframe=False, showcoastlines=True))
    st.plotly_chart(fig_fc_map, use_container_width=True)

    fc_table = fc_2027[["country", "gdp_growth_forecast", "gdp_growth_lower", "gdp_growth_upper",
                         "inflation_forecast", "model_confidence"]].copy()
    fc_table.columns = ["Country", "GDP Forecast", "GDP Lower", "GDP Upper", "Inflation Forecast", "Confidence"]
    st.dataframe(
        fc_table.sort_values("GDP Forecast", ascending=False)
               .style.format({"GDP Forecast": "{:.2f}%", "GDP Lower": "{:.2f}%", "GDP Upper": "{:.2f}%",
                               "Inflation Forecast": "{:.2f}%", "Confidence": "{:.0%}"})
               .background_gradient(subset=["GDP Forecast"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — TRADE NETWORK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🕸 Trade Network":
    st.title("🕸 Global Trade Network")
    st.caption("NetworkX graph — PageRank, betweenness centrality, vulnerability scores")

    net_df = load_network_metrics()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries", len(net_df))
    c2.metric("Trade Links", "86 bilateral")
    c3.metric("Total Trade", f"${net_df['trade_volume_bn'].sum()/2:.0f}B")
    c4.metric("Most Critical", net_df.iloc[0]["country"])
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">PageRank — Most Systemically Critical</div>',
                    unsafe_allow_html=True)
        fig_pr = px.bar(net_df.nlargest(10, "pagerank"), x="pagerank", y="country", orientation="h",
                         color="pagerank", color_continuous_scale="Blues",
                         labels={"pagerank": "PageRank", "country": ""})
        fig_pr.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              coloraxis_showscale=False)
        st.plotly_chart(fig_pr, use_container_width=True)
    with col_r:
        st.markdown('<div class="section-header">Trade Vulnerability Score</div>', unsafe_allow_html=True)
        fig_v = px.bar(net_df.nlargest(10, "vulnerability_score"), x="vulnerability_score", y="country",
                        orientation="h", color="vulnerability_score", color_continuous_scale="Reds",
                        labels={"vulnerability_score": "Vulnerability", "country": ""})
        fig_v.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             coloraxis_showscale=False)
        st.plotly_chart(fig_v, use_container_width=True)

    st.markdown('<div class="section-header">Betweenness vs PageRank (bubble = trade volume)</div>',
                unsafe_allow_html=True)
    fig_ns = px.scatter(net_df, x="pagerank", y="betweenness", size="trade_volume_bn",
                         color="region", hover_name="country", size_max=40,
                         labels={"pagerank": "PageRank", "betweenness": "Betweenness"})
    fig_ns.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ns, use_container_width=True)

    show_cols = ["country", "region", "pagerank", "betweenness", "total_exports_bn",
                 "total_imports_bn", "vulnerability_score", "top_export_partner", "top_import_partner"]
    st.dataframe(
        net_df[show_cols].sort_values("pagerank", ascending=False)
            .style.format({"pagerank": "{:.4f}", "betweenness": "{:.4f}", "vulnerability_score": "{:.3f}",
                           "total_exports_bn": "{:.0f}", "total_imports_bn": "{:.0f}"}),
        use_container_width=True, hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🎲 Monte Carlo":
    st.title("🎲 Monte Carlo Probabilistic Forecasts")
    st.caption("10,000 simulated futures per country — fan charts, recession probability heatmap")

    mc_data = load_monte_carlo()
    rec_df = load_recession_probs()

    c1, c2, c3 = st.columns(3)
    c1.metric("Simulations/Country", "10,000")
    c2.metric("Forecast Horizon", "5 years")
    c3.metric("Countries", len(mc_data))
    st.divider()

    mc_sel = st.selectbox("Country", sorted(mc_data.keys()), index=sorted(mc_data.keys()).index("USA"),
                           format_func=lambda k: iso3_to_name.get(k, k))

    mc = mc_data[mc_sel]
    years = [2023, 2024, 2025, 2026, 2027]

    def fan(years, pdata, title, rgb):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years + years[::-1], y=pdata["p90"] + pdata["p10"][::-1],
            fill="toself", fillcolor=f"rgba({rgb},0.10)", line=dict(color="rgba(0,0,0,0)"), name="P10–P90"))
        fig.add_trace(go.Scatter(
            x=years + years[::-1], y=pdata["p75"] + pdata["p25"][::-1],
            fill="toself", fillcolor=f"rgba({rgb},0.25)", line=dict(color="rgba(0,0,0,0)"), name="P25–P75"))
        fig.add_trace(go.Scatter(x=years, y=pdata["p50"], mode="lines+markers",
                                  line=dict(color=f"rgb({rgb})", width=2), name="Median"))
        fig.add_trace(go.Scatter(x=years, y=pdata["mean"], mode="lines",
                                  line=dict(color=f"rgb({rgb})", width=1, dash="dot"), name="Mean"))
        fig.update_layout(title=title, height=320, paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", yaxis_title="%")
        return fig

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(fan(years, mc["gdp_growth"], "GDP Growth Fan Chart", "74,207,138"),
                        use_container_width=True)
    with col_r:
        st.plotly_chart(fan(years, mc["inflation"], "Inflation Fan Chart", "224,92,92"),
                        use_container_width=True)

    st.markdown('<div class="section-header">Recession Probability Heatmap — All Countries × Years</div>',
                unsafe_allow_html=True)
    rec_pivot = rec_df.pivot_table(index="iso3", columns="year", values="recession_prob")
    fig_rh = go.Figure(go.Heatmap(
        z=rec_pivot.values * 100, x=[str(y) for y in rec_pivot.columns], y=rec_pivot.index.tolist(),
        colorscale="Reds", zmin=0, zmax=60, colorbar=dict(title="Recession Prob %"),
        text=[[f"{v:.0f}%" for v in row] for row in rec_pivot.values * 100],
        texttemplate="%{text}",
    ))
    fig_rh.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rh, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — COUNTRY DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔮 Country Deep Dive":
    st.title(f"🔮 {selected_country}")

    econ_df = load_economic_data()
    forecast_df = load_forecasts()
    net_df = load_network_metrics()
    shock_df = load_shock_results()

    hist = econ_df[econ_df["iso3"] == selected_iso3].sort_values("year")
    fc = forecast_df[forecast_df["iso3"] == selected_iso3]
    net_row = net_df[net_df["iso3"] == selected_iso3]
    lat = hist[hist["year"] == 2022].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GDP (2022)", f"${lat['gdp_usd']/1e12:.2f}T")
    c2.metric("GDP Growth", f"{lat['gdp_growth_pct']:.1f}%")
    c3.metric("Inflation", f"{lat['inflation_pct']:.1f}%")
    c4.metric("Unemployment", f"{lat['unemployment_pct']:.1f}%")
    c5.metric("Govt Debt/GDP", f"{lat['govt_debt_pct_gdp']:.0f}%")

    if len(net_row):
        nr = net_row.iloc[0]
        c6, c7, c8 = st.columns(3)
        c6.metric("PageRank", f"{nr['pagerank']:.4f}")
        c7.metric("Top Export To", nr["top_export_partner"])
        c8.metric("Vulnerability", f"{nr['vulnerability_score']:.3f}")

    st.divider()
    st.markdown('<div class="section-header">Historical Macro Indicators</div>', unsafe_allow_html=True)
    fig_h = make_subplots(rows=2, cols=2,
                           subplot_titles=["GDP Growth %", "Inflation %", "Unemployment %", "Trade ($T)"])
    fig_h.add_trace(go.Scatter(x=hist["year"], y=hist["gdp_growth_pct"], mode="lines+markers",
                                line=dict(color="#4caf8a"), name="GDP Growth"), row=1, col=1)
    fig_h.add_trace(go.Scatter(x=hist["year"], y=hist["inflation_pct"], mode="lines+markers",
                                line=dict(color="#e05c5c"), name="Inflation"), row=1, col=2)
    fig_h.add_trace(go.Scatter(x=hist["year"], y=hist["unemployment_pct"], mode="lines+markers",
                                line=dict(color="#f0a050"), name="Unemployment"), row=2, col=1)
    fig_h.add_trace(go.Scatter(x=hist["year"], y=hist["exports_usd"]/1e12, mode="lines",
                                line=dict(color="#4a7fb5"), name="Exports"), row=2, col=2)
    fig_h.add_trace(go.Scatter(x=hist["year"], y=hist["imports_usd"]/1e12, mode="lines",
                                line=dict(color="#9b59b6", dash="dash"), name="Imports"), row=2, col=2)
    fig_h.add_hline(y=0, row=1, col=1, line_dash="dot", line_color="gray", opacity=0.4)
    fig_h.add_hline(y=2, row=1, col=2, line_dash="dot", line_color="gray", opacity=0.4)
    fig_h.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         showlegend=False)
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown('<div class="section-header">Shock Vulnerability — All 8 Scenarios</div>',
                unsafe_allow_html=True)
    SCENARIO_LABELS = {
        "oil_price_surge_40": "Oil +40%", "oil_price_crash_50": "Oil -50%",
        "us_rate_hike_200bps": "US Rate +200bps", "china_slowdown_hard": "China Landing",
        "global_supply_chain": "Supply Chain", "trade_war_us_china": "Trade War",
        "eurozone_debt_crisis": "Eurozone Crisis", "pandemic_shock": "Pandemic",
    }
    cs = shock_df[shock_df["iso3"] == selected_iso3].copy()
    cs["label"] = cs["scenario"].map(SCENARIO_LABELS)
    cs = cs.sort_values("gdp_growth_delta")
    fig_cs = go.Figure(go.Bar(
        y=cs["label"], x=cs["gdp_growth_delta"], orientation="h",
        marker_color=["#e05c5c" if v < 0 else "#4caf8a" for v in cs["gdp_growth_delta"]],
        text=[f"{v:+.2f}pp" for v in cs["gdp_growth_delta"]], textposition="outside",
    ))
    fig_cs.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.4)
    fig_cs.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="GDP Growth Impact (pp)")
    st.plotly_chart(fig_cs, use_container_width=True)

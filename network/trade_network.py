"""
network/trade_network.py
Global Trade Network Analysis
Builds the economic graph, computes centrality, dependency metrics,
and shock propagation pathways.

Run: python network/trade_network.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
import os

print("=" * 65)
print("  GLOBAL ECONOMIC DIGITAL TWIN — Trade Network v1.0")
print("  yash752-stack | Module 2: Global Trade Network")
print("=" * 65)

os.makedirs("data/processed", exist_ok=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
print("\n[1/5] Loading trade data...")
trade_df = pd.read_csv("data/processed/trade_data.csv")
econ_df  = pd.read_csv("data/processed/economic_data.csv")

# Use latest year
YEAR = 2022
trade_y = trade_df[trade_df["year"] == YEAR]
econ_y  = econ_df[econ_df["year"] == YEAR].set_index("iso3")

# ── BUILD GRAPH ──────────────────────────────────────────────────────────────
print("\n[2/5] Building directed trade network...")

G = nx.DiGraph()

# Add country nodes
for iso3, row in econ_y.iterrows():
    G.add_node(iso3,
        name=row["country"],
        region=row["region"],
        gdp=row["gdp_usd"],
        gdp_growth=row["gdp_growth_pct"],
        inflation=row["inflation_pct"],
        unemployment=row["unemployment_pct"],
        exports=row["exports_usd"],
        imports=row["imports_usd"],
        oil_dependency=row["oil_dependency_score"],
    )

# Add trade edges
for _, row in trade_y.iterrows():
    if row["exporter"] in G.nodes and row["importer"] in G.nodes:
        G.add_edge(
            row["exporter"], row["importer"],
            weight=row["trade_value_usd"],
            weight_bn=row["trade_value_usd"] / 1e9,
        )

print(f"      ✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── CENTRALITY METRICS ───────────────────────────────────────────────────────
print("\n[3/5] Computing centrality and dependency metrics...")

# Degree centrality
in_degree  = nx.in_degree_centrality(G)
out_degree = nx.out_degree_centrality(G)

# Weighted degree (trade volume)
total_exports = {n: sum(d["weight"] for _, _, d in G.out_edges(n, data=True)) for n in G.nodes}
total_imports = {n: sum(d["weight"] for _, _, d in G.in_edges(n, data=True)) for n in G.nodes}

# PageRank — which countries are most "depended on" by important partners
pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

# Betweenness centrality — which countries are key intermediaries
betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

# Clustering coefficient (undirected view)
G_undirected = G.to_undirected()
clustering = nx.clustering(G_undirected)

# ── DEPENDENCY METRICS ───────────────────────────────────────────────────────
# For each country, who are their top 3 export/import partners?
dependency = {}
for node in G.nodes:
    out_edges = [(v, d["weight"]) for _, v, d in G.out_edges(node, data=True)]
    in_edges  = [(u, d["weight"]) for u, _, d in G.in_edges(node, data=True)]

    top_export_partners = sorted(out_edges, key=lambda x: -x[1])[:3]
    top_import_partners = sorted(in_edges,  key=lambda x: -x[1])[:3]

    total_exp = total_exports.get(node, 1)
    total_imp = total_imports.get(node, 1)

    # Concentration: how dependent on single partner?
    exp_concentration = (top_export_partners[0][1] / total_exp) if top_export_partners and total_exp > 0 else 0
    imp_concentration = (top_import_partners[0][1] / total_imp) if top_import_partners and total_imp > 0 else 0

    dependency[node] = {
        "top_export_partners": [(c, round(v/1e9, 1)) for c, v in top_export_partners],
        "top_import_partners": [(c, round(v/1e9, 1)) for c, v in top_import_partners],
        "export_concentration": round(exp_concentration, 3),
        "import_concentration": round(imp_concentration, 3),
        "vulnerability_score":  round((exp_concentration + imp_concentration) / 2, 3),
    }

# ── BUILD METRICS DATAFRAME ──────────────────────────────────────────────────
metrics_rows = []
for node in G.nodes:
    data = G.nodes[node]
    metrics_rows.append({
        "iso3":                node,
        "country":             data.get("name", node),
        "region":              data.get("region", ""),
        "gdp_usd":             data.get("gdp", 0),
        "pagerank":            round(pagerank.get(node, 0), 5),
        "betweenness":         round(betweenness.get(node, 0), 5),
        "in_degree_centrality":round(in_degree.get(node, 0), 4),
        "out_degree_centrality":round(out_degree.get(node, 0), 4),
        "clustering_coef":     round(clustering.get(node, 0), 4),
        "total_exports_bn":    round(total_exports.get(node, 0) / 1e9, 1),
        "total_imports_bn":    round(total_imports.get(node, 0) / 1e9, 1),
        "trade_volume_bn":     round((total_exports.get(node, 0) + total_imports.get(node, 0)) / 1e9, 1),
        "export_concentration":dependency[node]["export_concentration"],
        "import_concentration":dependency[node]["import_concentration"],
        "vulnerability_score": dependency[node]["vulnerability_score"],
        "top_export_partner":  dependency[node]["top_export_partners"][0][0] if dependency[node]["top_export_partners"] else "",
        "top_import_partner":  dependency[node]["top_import_partners"][0][0] if dependency[node]["top_import_partners"] else "",
    })

metrics_df = pd.DataFrame(metrics_rows).sort_values("pagerank", ascending=False)
metrics_df.to_csv("data/processed/network_metrics.csv", index=False)

# ── SHOCK PROPAGATION ────────────────────────────────────────────────────────
print("\n[4/5] Computing shock propagation pathways...")

def simulate_shock_propagation(G, origin_country, shock_magnitude, max_hops=3):
    """
    Simulate how a GDP shock in one country propagates through trade links.
    Uses trade-weighted cascade model.
    """
    affected = {origin_country: shock_magnitude}
    propagation_log = [{
        "hop": 0, "country": origin_country,
        "shock": shock_magnitude, "channel": "origin"
    }]

    current_wave = {origin_country: shock_magnitude}

    for hop in range(1, max_hops + 1):
        next_wave = {}
        for src_country, src_shock in current_wave.items():
            # Shock spreads to trading partners
            for _, dest, edge_data in G.out_edges(src_country, data=True):
                if dest in affected:
                    continue
                # Transmission: trade share × shock magnitude × decay
                src_exports = total_exports.get(src_country, 1)
                if src_exports == 0:
                    continue
                trade_share = edge_data["weight"] / src_exports
                transmitted  = src_shock * trade_share * (0.6 ** hop)  # decay per hop

                if abs(transmitted) > 0.01:  # threshold
                    next_wave[dest] = transmitted
                    affected[dest]  = transmitted
                    propagation_log.append({
                        "hop": hop,
                        "country": dest,
                        "shock": round(transmitted, 3),
                        "channel": f"{src_country}→{dest} (trade share: {trade_share:.2%})"
                    })

        current_wave = next_wave
        if not current_wave:
            break

    return affected, propagation_log

# Run example propagation: US recession
us_recession_affected, us_log = simulate_shock_propagation(G, "USA", -3.0)
china_shock_affected,  cn_log = simulate_shock_propagation(G, "CHN", -4.0)
oil_shock_affected,    oil_log = simulate_shock_propagation(G, "SAU", -5.0)

propagation_results = {
    "USA_recession_3pct": {k: round(v, 3) for k, v in us_recession_affected.items()},
    "CHN_slowdown_4pct":  {k: round(v, 3) for k, v in china_shock_affected.items()},
    "SAU_oil_shock_5pct": {k: round(v, 3) for k, v in oil_shock_affected.items()},
}

with open("data/processed/propagation_results.json", "w") as f:
    json.dump(propagation_results, f, indent=2)

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n[5/5] Network Analysis Summary")
print("-" * 45)
print(f"  Nodes (countries): {G.number_of_nodes()}")
print(f"  Edges (trade links): {G.number_of_edges()}")
total_trade = sum(d["weight"] for _, _, d in G.edges(data=True))
print(f"  Total trade volume: ${total_trade/1e12:.1f}T")
print(f"\n  Top 5 by PageRank (most critical nodes):")
for _, row in metrics_df.head(5).iterrows():
    print(f"    {row['iso3']:4s} {row['country']:20s} PR={row['pagerank']:.4f}  Trade=${row['trade_volume_bn']:.0f}B")
print(f"\n  Most vulnerable (high trade concentration):")
vuln = metrics_df.nlargest(3, "vulnerability_score")
for _, row in vuln.iterrows():
    print(f"    {row['iso3']:4s} {row['country']:20s} Vuln={row['vulnerability_score']:.3f}")
print(f"\n  US Recession spillover: {len(us_recession_affected)} countries affected")
print(f"  China Slowdown spillover: {len(china_shock_affected)} countries affected")
print("-" * 45)
print("\n✅ Module 2 complete — Network metrics + propagation saved\n")

# Save graph for dashboard
nx.write_gexf(G, "data/processed/trade_network.gexf")
print("  ✅ Graph exported: data/processed/trade_network.gexf")

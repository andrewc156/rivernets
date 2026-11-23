"""Figures for reports/pollution_sources_report.tex.

usage: python make_pollution_figures.py      (after analyze_pollution_vs_error.py)

  reports/figures/pollution_map.*       river network, traced sources by group, stations
  reports/figures/pollution_error.*     where each target's error is (graph model), with
                                        what lies upstream of the worst stations
  reports/figures/pollution_heatmap.*   Spearman correlation of each pressure metric with
                                        the stations' error shares
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

FIG = Path("reports/figures")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"
RIVER = "#b9c7d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlecolor": INK})
TARGETS = ["Water temperature", "DO concentration", "Specific conductance", "pH", "Turbidity"]
TSHORT = {"Water temperature": "Water temp.", "DO concentration": "DO", "Specific conductance": "Spec. cond.",
          "pH": "pH", "Turbidity": "Turbidity"}
GROUP = {"Municipal wastewater": "Municipal wastewater",
         **{c: "Mining and energy" for c in ["Power plant", "Coal mine", "Metal or uranium mine", "Other mining",
                                             "Oil and gas"]},
         **{c: "Factories and other permits" for c in ["Manufacturing", "Industrial stormwater", "Fish hatchery",
                                                       "Drinking-water plant", "Other individual permit"]},
         "Data centre": "Data centre"}
STYLE = {"Municipal wastewater": ("o", BLUE), "Mining and energy": ("s", ORANGE),
         "Factories and other permits": ("D", AQUA), "Data centre": ("^", INK)}
# what is upstream of / special about the stations that carry most of the error (from
# data/pollution/source_links.csv, the graph and the station records; see the report)
NOTE = {
    "09510000": "below Bartlett Dam",
    "09380000": "27 km below Glen Canyon Dam",
    "09234500": "1 km below Flaming Gorge Dam",
    "09315000": "23 coal mines, 25 oil-gas permits upstream",
    "09365000": "Farmington WWTP 3 km up; Navajo Dam 81 km",
    "09379500": "San Juan, below Four Corners energy area",
    "09106485": "summer record, 351 cfs below irrigation diversions",
    "09180000": "Paradox Valley salt brine; uranium-belt mines",
    "09368000": "6-day record (Sep 2023); coal mines, power plant 36-40 km up",
    "09371010": "6-day record (Sep 2023)",
    "09085150": "Glenwood WWTF 6 km and hot springs upstream",
    "09092570": "below Glenwood springs and Roaring Fork",
    "09163500": "Grand Junction (Persigo WWTF 60 km up)",
    "09429490": "Las Vegas wastewater via Lake Mead; 19 data centres",
}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_map():
    G = pickle.load(open("graphs/colorado/master_compressed_river_graph.pkl", "rb"))
    a = next(iter(G.nodes(data=True)))[1]
    lat0, lon0 = a["latitude"] + a["row"] / 1200.0, a["longitude"] - a["col"] / 1200.0
    src = pd.read_csv("data/pollution/sources_traced.csv")
    src = src[src["category"].isin(GROUP)].assign(group=lambda d: d["category"].map(GROUP))
    E = pd.read_csv("results/pollution/station_errors.csv")
    P = pd.read_csv("results/pollution/station_pressures.csv")

    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.10)
    for u, v, d in G.edges(data=True):
        rc = np.array([divmod(int(n), 15001) for n in d["original_path"]], float)
        ax.plot(lon0 + rc[:, 1] / 1200.0, lat0 - rc[:, 0] / 1200.0, color=RIVER, lw=1.0, zorder=1)
    for grp, (m, c) in STYLE.items():
        g = src[src["group"] == grp]
        size = 9 if grp != "Data centre" else 26
        if grp == "Municipal wastewater":
            size = 6 + 9 * np.sqrt(g["design_mgd"].fillna(0.1).clip(0.1, 160))
        down = g["first_node"].notna().to_numpy()
        sz = size if np.isscalar(size) else size.to_numpy()
        ax.scatter(g["lon"][~down], g["lat"][~down], s=sz if np.isscalar(sz) else sz[~down], marker=m, color=c,
                   alpha=0.16, lw=0, zorder=2)
        ax.scatter(g["lon"][down], g["lat"][down], s=sz if np.isscalar(sz) else sz[down], marker=m, color=c,
                   alpha=0.85, edgecolor=SURFACE, lw=0.3, zorder=3)
    st = P.rename(columns={"Unnamed: 0": "node"}) if "Unnamed: 0" in P else P
    lon = {n: d["longitude"] for n, d in G.nodes(data=True)}
    lat = {n: d["latitude"] for n, d in G.nodes(data=True)}
    ax.scatter([lon[n] for n in st["node"]], [lat[n] for n in st["node"]], s=70, color=SURFACE, edgecolor=INK,
               lw=1.4, zorder=5)
    lab = {"09180000": ("Dolores near Cisco", (6, -3), "left"), "09380000": ("Lees Ferry", (6, 2), "left"),
           "09510000": ("Verde below Bartlett Dam", (6, -2), "left"),
           "09368000": ("San Juan at Shiprock", (-6, -9), "right"),
           "09371010": ("Four Corners", (-6, 4), "right"),
           "09106485": ("Colorado above Gunnison", (-7, 6), "right"),
           "09365000": ("Farmington", (6, 3), "left"), "09234500": ("Greendale", (6, 2), "left"),
           "09429490": ("above Imperial Dam", (-6, 0), "right"),
           "09085150": ("Glenwood Springs", (-6, 6), "right")}
    for n, row in st.iterrows():
        site = row["stations"].split(",")[0].replace("USGS-", "")
        if site in lab:
            t, off, ha = lab[site]
            ax.annotate(t, (lon[row["node"]], lat[row["node"]]), xytext=off, textcoords="offset points",
                        fontsize=7, color=INK, ha=ha, va="center", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.8))
    for name, (la, lo, txt) in {"Las Vegas": (36.17, -115.14, "Las Vegas"), "Phoenix": (33.45, -112.07, "Phoenix"),
                                "Tucson": (32.22, -110.97, "Tucson"),
                                "Grand Junction": (39.06, -108.55, "Grand Junction")}.items():
        ax.text(lo, la - 0.22, txt, fontsize=6.5, color=MUTED, ha="center", va="top", style="italic", zorder=4)
    ax.set_xlim(-115.9, -104.6)
    ax.set_ylim(31.4, 43.2)
    ax.set_aspect(1 / np.cos(np.radians(37.5)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    h = [Line2D([], [], ls="", marker=STYLE[g][0], color=STYLE[g][1], ms=6 if g != "Data centre" else 7, label=g)
         for g in STYLE]
    h += [Line2D([], [], ls="", marker="o", mfc=SURFACE, mec=INK, mew=1.4, ms=7, label="Water-quality station"),
          Line2D([], [], color=RIVER, lw=1.6, label="River network (graph)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.012, "Solid: drains past at least one station. Faded: drains elsewhere. "
             "Wastewater marker area grows with design flow.", ha="center", fontsize=7, color=INK2)
    save(fig, "pollution_map")


def fig_error():
    E = pd.read_csv("results/pollution/station_errors.csv", dtype={"name": str})
    P = pd.read_csv("results/pollution/station_pressures.csv")
    site = {r["node"]: r["stations"].split(",")[0].replace("USGS-", "") for _, r in P.iterrows()}
    SHORTNAME = {"09510000": "Verde below Bartlett Dam", "09380000": "Colorado at Lees Ferry",
                 "09315000": "Green at Green River, UT", "09234500": "Green near Greendale",
                 "09365000": "San Juan at Farmington", "09106485": "Colorado above Gunnison",
                 "09163500": "Colorado at CO-UT line", "09379500": "San Juan at Mexican Hat",
                 "09180000": "Dolores near Cisco", "09092570": "Colorado at Rulison",
                 "09085150": "Colorado below Glenwood", "09429490": "Colorado above Imperial Dam",
                 "09368000": "San Juan at Shiprock", "09371010": "San Juan at Four Corners"}
    fig, axes = plt.subplots(5, 1, figsize=(7.2, 8.6), gridspec_kw={"hspace": 0.62})
    fig.subplots_adjust(left=0.25, right=0.955, top=0.97, bottom=0.06)
    for ax, t in zip(axes, TARGETS):
        e = E[(E["model"] == "Graph-Laplacian") & (E["target"] == t)].sort_values("share", ascending=False)
        top = e.head(5)
        rest = e.iloc[5:]
        labels = [SHORTNAME.get(site.get(n, ""), nm.title()) for n, nm in zip(top["node"], top["name"])] \
            + [f"other {len(rest)} stations"]
        vals = list(top["share"]) + [rest["share"].sum()]
        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals, color=[BLUE] * len(top) + [GRID], height=0.68, edgecolor=SURFACE, lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.6)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for yy, v, n in zip(y, vals, list(top["node"]) + [None]):
            txt = f"{v:.0%}"
            note = NOTE.get(site.get(n, ""), "") if n is not None else ""
            ax.text(v + 0.01, yy, txt + (f"   {note}" if note else ""), va="center", fontsize=6.3,
                    color=INK2)
        ax.set_title(f"{t}: graph model R² {e['r2_model'].iloc[0]:.3f}, {len(e)} stations", loc="left")
    axes[-1].set_xlabel("Station's share of the graph model's squared error (equal station weights)", fontsize=7)
    save(fig, "pollution_error")


def fig_heatmap():
    R = pd.read_csv("results/pollution/pressure_error_spearman.csv")
    metrics = list(dict.fromkeys(R["metric"]))
    lab = R.drop_duplicates("metric").set_index("metric")
    M = R.pivot(index="metric", columns="target", values="rho").reindex(index=metrics, columns=TARGETS)
    Pv = R.pivot(index="metric", columns="target", values="p").reindex(index=metrics, columns=TARGETS)
    Q = R.pivot(index="metric", columns="target", values="q").reindex(index=metrics, columns=TARGETS)
    fig, ax = plt.subplots(figsize=(7.2, 8.4))
    fig.subplots_adjust(left=0.47, right=0.97, top=0.93, bottom=0.07)
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    im = ax.imshow(M.to_numpy(float), cmap=cmap, norm=TwoSlopeNorm(vmin=-0.8, vcenter=0, vmax=0.8), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.5, color=MUTED)
                continue
            strong = Pv.iat[i, j] < 0.05
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.2,
                    color=INK if strong else INK2, fontweight="bold" if strong else "normal")
            if Q.iat[i, j] < 0.05:
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec=INK, lw=1.2))
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels([TSHORT[t] for t in TARGETS], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([lab.loc[m, "label"] for m in metrics], fontsize=6.6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    themes = lab.loc[metrics, "theme"].to_numpy()
    for i in range(1, len(metrics)):
        if themes[i] != themes[i - 1]:
            ax.axhline(i - 0.5, color=SURFACE, lw=3)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.02, aspect=50)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=6.5)
    cb.set_label("Spearman rho with the station's error share\n(red: more pressure, larger error; "
                 "bold: p < 0.05; boxed: FDR < 0.05)", fontsize=6.8, color=INK2)
    save(fig, "pollution_heatmap")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_map()
    fig_error()
    fig_heatmap()
    print("ok")


if __name__ == "__main__":
    main()

"""Poster figures (300 dpi PNG + vector PDF) from the saved benchmark results.

usage: python make_poster_figures.py      (seconds; no model fitting)

  poster_1_models.*   LOO R2 of every model for each target, graph model highlighted
  poster_2_splits.*   R2 as more of the network is held out (LOO, 3 stations, whole subbasin)
  poster_3_map.*      river network map: where the graph model beats Ridge for specific conductance
  poster_4_graph.*    the river (MERIT cells) next to the graph built from it
  poster_5_zoom.*     close-up: nodes on the river and edges following it

reads results/split_comparison.csv, results/split_subbasin_folds.csv and the LOO
predictions (results/benchmark_results_inner_fs_graph, results/benchmark_results_inner_fs)
writes reports/figures/
"""
import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUT = Path("reports/figures")

# validated reference palette (light mode): three categorical slots, all-pairs safe
GRAPH, RIDGE, GB = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, CONTEXT = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#b9b8b1"
DIV_NEG, DIV_MID, DIV_POS = "#e34948", "#dcdbd5", "#2a78d6"   # red <- gray -> blue

TARGETS = ["Water temperature", "DO concentration", "Specific conductance", "pH"]
TLABEL = {"Water temperature": "Water temperature", "DO concentration": "Dissolved oxygen",
          "Specific conductance": "Specific conductance", "pH": "pH", "Turbidity": "Turbidity"}
MLABEL = {"Graph-Laplacian": "Graph-Laplacian", "Graph-Laplacian + GB": "Graph + GB",
          "Gradient Boosting": "Gradient boosting", "Random Forest": "Random forest",
          "Elastic Net": "Elastic net", "Spline GAM": "Spline GAM", "OLS": "OLS",
          "Ridge": "Ridge", "Dummy": "Training mean"}

mpl.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png", "and .pdf")


def title(fig, main, sub):
    fig.text(0.0, 1.04, main, fontsize=19, fontweight="bold", color=INK, transform=fig.transFigure)
    fig.text(0.0, 1.0, sub, fontsize=13, color=INK2, transform=fig.transFigure)


# ---------------------------------------------------------------------------
def fig_models(sc):
    loo = sc[sc["split"] == "loo"].pivot_table(index="target", columns="model", values="R2")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y = np.arange(len(TARGETS))[::-1]
    ax.axvline(0, color=AXIS, lw=1, zorder=1)
    for yi, t in zip(y, TARGETS):
        row = loo.loc[t]
        # the six non-graph models (the graph + GB hybrid was not adopted; Dummy is the 0 line)
        others = row.drop(["Graph-Laplacian", "Graph-Laplacian + GB", "Dummy"])
        g = row["Graph-Laplacian"]
        lo, hi = min(others.min(), g), max(others.max(), g)
        ax.hlines(yi, lo, hi, color=GRID, lw=2, zorder=1)
        ax.scatter(others, [yi] * len(others), s=70, color=CONTEXT, edgecolor=SURFACE, lw=2, zorder=2)
        ax.scatter([g], [yi], s=150, color=GRAPH, edgecolor=SURFACE, lw=2, zorder=4)
        best = others.idxmax()
        ax.annotate(f"{g:.2f}", (g, yi), xytext=(0, 12), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold", color=INK)
        ax.annotate(f"best other: {MLABEL[best]} {others.max():.2f}", (others.max(), yi),
                    xytext=(0, -13), textcoords="offset points", ha="center", va="top",
                    fontsize=10.5, color=INK2)
        lead = g - others.max()
        ax.annotate("0.00" if abs(lead) < 0.005 else f"{lead:+.2f}", (1.0, yi), xycoords=("axes fraction", "data"), xytext=(14, 0),
                    textcoords="offset points", ha="left", va="center", annotation_clip=False,
                    fontsize=12.5, fontweight="bold" if lead > 0.05 else "normal",
                    color=INK if lead > 0.05 else MUTED)
    ax.annotate("graph lead", (1.0, 1.0), xycoords="axes fraction", xytext=(14, 2),
                textcoords="offset points", ha="left", va="bottom", fontsize=10.5, color=MUTED,
                annotation_clip=False)
    ax.set_yticks(y, [TLABEL[t] for t in TARGETS])
    ax.tick_params(axis="y", length=0, labelsize=13, labelcolor=INK)
    ax.set_xlim(-0.35, 1.0)
    ax.set_ylim(-0.6, len(TARGETS) - 0.35)
    ax.set_xlabel("R² at a held-out station (leave one station out, 17–24 stations per target)")
    ax.xaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.annotate("← worse than predicting the mean", (0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-6, 2), textcoords="offset points", ha="right", va="bottom",
                fontsize=10, color=MUTED, annotation_clip=False)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=11, color=GRAPH, mec=SURFACE, label="Graph-Laplacian model"),
               plt.Line2D([], [], ls="", marker="o", ms=8, color=CONTEXT, mec=SURFACE,
                          label="Six other models: OLS, Ridge, elastic net, spline GAM, random forest, gradient boosting")]
    fig.subplots_adjust(top=0.83)
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.905), ncol=2, frameon=False,
               fontsize=10.5, handletextpad=0.3, columnspacing=1.5)
    title(fig, "The river-network term pays off for salinity and pH",
          "R² of each model for each water-quality target: the graph model leads by 0.36 on "
          "specific conductance and 0.09 on pH")
    save(fig, "poster_1_models")


# ---------------------------------------------------------------------------
def fig_splits(sc):
    splits = ["loo", "station_repeated", "subbasin"]
    xl = ["1 station", "3 stations", "Whole\nsubbasin"]
    panels = ["Water temperature", "Specific conductance", "pH"]
    # drawn in this order: graph last, on top; Ridge wider underneath so a tie still shows both
    series = [("Ridge", RIDGE, "Ridge", 5.0), ("Gradient Boosting", GB, "Gradient boosting", 2.5),
              ("Graph-Laplacian", GRAPH, "Graph-Laplacian", 2.5)]
    w = sc.pivot_table(index=["target", "model"], columns="split", values="R2")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharey=True)
    for ax, t in zip(axes, panels):
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        ax.yaxis.grid(True, color=GRID, lw=1)
        ax.set_axisbelow(True)
        ends = {}
        for m, c, lab, lw in series:
            v = w.loc[(t, m), splits].to_numpy(float)
            ax.plot(range(3), v, color=c, lw=lw, zorder=3)
            ax.scatter(range(3), v, s=70 if lw < 3 else 120, color=c, edgecolor=SURFACE, lw=2, zorder=4)
            ends[lab] = (round(v[-1], 2), c)
        # one label per distinct end value (ties share a label), spaced with leaders
        groups = {}
        for lab, (val, c) in ends.items():
            groups.setdefault(val, []).append((lab, c))
        placed = []
        for val in sorted(groups):
            labs = sorted(groups[val], key=lambda x: x[0] != "Graph-Laplacian")
            ypos = val
            for pv in placed:
                if abs(ypos - pv) < 0.12:
                    ypos = pv + 0.12
            placed.append(ypos)
            ax.annotate(" = ".join(l.replace("Graph-Laplacian", "Graph") if len(labs) > 1 else l for l, _ in labs) + f"  {val:.2f}", (2, val), xytext=(2.15, ypos),
                        textcoords="data", va="center", ha="left", fontsize=11, color=INK,
                        arrowprops=dict(arrowstyle="-", color=labs[0][1], lw=1.2, shrinkA=0, shrinkB=5))
        g0 = w.loc[(t, "Graph-Laplacian"), "loo"]
        ax.annotate(f"{g0:.2f}", (0, g0), xytext=(-10, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=11, fontweight="bold", color=INK)
        ax.set_title(TLABEL[t], loc="left", fontweight="bold", color=INK, pad=10)
        ax.set_xticks(range(3), xl)
        ax.tick_params(axis="x", length=0, labelsize=11.5)
        ax.set_xlabel("Held out together", fontsize=11, color=MUTED)
        ax.set_xlim(-0.4, 4.1)
        ax.spines["left"].set_visible(ax is axes[0])
    axes[0].set_ylabel("R² at held-out stations")
    axes[0].set_ylim(-1.3, 1.0)
    axes[0].text(2.15, 0.03, "predicting the mean", fontsize=9.5, color=MUTED, va="bottom", ha="left")
    handles = [plt.Line2D([], [], color=c, lw=2.5, marker="o", ms=7, mec=SURFACE, mew=1.5, label=lab)
               for _, c, lab, _ in series[::-1]]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.93), ncol=3, frameon=False,
               fontsize=11.5)
    fig.subplots_adjust(wspace=0.25, top=0.84)
    title(fig, "Network skill is local: it fades as neighbours are removed",
          "Specific conductance: the graph model's lead holds with 3 stations out, "
          "but vanishes when its whole subbasin is held out")
    save(fig, "poster_2_splits")


# ---------------------------------------------------------------------------
def fig_map():
    with open("graphs/colorado/master_compressed_river_graph.pkl", "rb") as f:
        G = pickle.load(f)
    cols = 15001
    a = next(iter(G.nodes(data=True)))[1]
    lat0, lon0 = a["latitude"] + a["row"] / 1200.0, a["longitude"] - a["col"] / 1200.0

    def ll(n):
        r, c = divmod(int(n), cols)
        return lon0 + c / 1200.0, lat0 - r / 1200.0

    # per-station error of the graph model against Ridge, LOO, specific conductance
    pg = pd.read_pickle("results/benchmark_results_inner_fs_graph/benchmark_predictions.pkl")
    pr = pd.read_pickle("results/benchmark_results_inner_fs/benchmark_predictions.pkl")
    t = "Specific conductance"
    mse = lambda p: (p.assign(e=(p.y_true - p.y_pred) ** 2).groupby("node").e.mean())
    g = mse(pg[(pg.target == t) & (pg.model == "Graph-Laplacian")])
    r = mse(pr[(pr.target == t) & (pr.model == "Ridge")])
    ratio = np.log2(r / g).rename("log2")          # > 0: graph error lower
    subb = json.load(open("results/benchmark_results_split_subbasin/benchmark_config.json"))[
        "graph_laplacian"]["subbasins"]
    folds = pd.read_csv("results/split_subbasin_folds.csv")
    gloo = folds[(folds.target == t) & (folds.model == "Graph-Laplacian")].set_index("fold")["R2_loo"]

    fig, ax = plt.subplots(figsize=(9.5, 9.6))
    ax.set_anchor("N")
    fig.subplots_adjust(top=0.975, bottom=0.03)
    for u, v, d in G.edges(data=True):
        xy = np.array([ll(n) for n in d["original_path"]])
        ax.plot(xy[:, 0], xy[:, 1], color=AXIS, lw=1.1, zorder=1)
    dams = {"Glen Canyon": (36.9375, -111.4844), "Hoover": (36.0161, -114.7377),
            "Davis": (35.1972, -114.5697), "Parker": (34.2964, -114.1394),
            "Flaming Gorge": (40.9147, -109.4217), "Navajo": (36.8006, -107.6125),
            "Blue Mesa": (38.4528, -107.3358)}
    for name, (la, lo) in dams.items():
        ax.scatter(lo, la, marker="^", s=60, color=INK2, edgecolor=SURFACE, lw=1.5, zorder=3)
    for name in ["Glen Canyon", "Hoover", "Flaming Gorge"]:
        la, lo = dams[name]
        ax.annotate(f"{name} Dam", (lo, la), xytext=(7, -2), textcoords="offset points",
                    fontsize=9.5, color=INK2, ha="left", va="top")

    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    pts = pd.DataFrame({"node": ratio.index, "v": ratio.clip(-2, 2).to_numpy()})
    pts["lon"] = [G.nodes[n]["longitude"] for n in pts.node]
    pts["lat"] = [G.nodes[n]["latitude"] for n in pts.node]
    sc_ = ax.scatter(pts.lon, pts.lat, c=pts.v, cmap=cmap, norm=norm, s=150, edgecolor=SURFACE,
                     lw=2, zorder=5)
    # subbasin labels with the graph model's LOO R2 within the subbasin
    pts["sub"] = pts.node.astype(str).map(subb)
    where = {"Green": (-110.45, 40.9, "right"), "Colorado headwaters": (-106.85, 39.75, "left"),
             "Gunnison + Colorado-Dolores": (-108.15, 37.95, "left"), "San Juan": (-108.6, 36.25, "left"),
             "Lower Colorado + Verde": (-111.25, 35.2, "left")}
    for s_, (x, yv, ha) in where.items():
        extra = "\nall 3 stations below dams" if s_.startswith("Lower") else ""
        ax.text(x, yv, f"{s_}\nR² {gloo[s_]:.2f}{extra}", fontsize=11.5, color=INK, ha=ha, va="center",
                fontweight="bold", linespacing=1.3, zorder=6)
    ax.set_xlim(-115.5, -104.4)
    ax.set_ylim(31.6, 42.3)
    ax.set_aspect(1 / np.cos(np.radians(38)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(sc_, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02, aspect=40, panchor=(0.5, 1.0),
                      ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_xticklabels(["≥4× higher", "2× higher", "same", "2× lower", "≥4× lower"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=11, labelcolor=INK2)
    cb.set_label("Graph model's specific-conductance error vs Ridge, per station (leave one station out)",
                 color=INK2, fontsize=11.5)
    ax.scatter([], [], marker="^", s=60, color=INK2, label="Major dam")
    ax.legend(loc="lower left", frameon=False, fontsize=11)
    title(fig, "Where the network helps: salinity in the upper basin",
          "Blue stations: the graph model's error is lower than Ridge's. Labels give its R² "
          "within each subbasin.")
    save(fig, "poster_3_map")


# ---------------------------------------------------------------------------
GAUGE, CONFL = RIDGE, INK2   # flow-gauge nodes (orange), confluences (dark grey)
ZOOM = (-109.5, -106.95, 38.55, 39.75)   # lon0, lon1, lat0, lat1: Glenwood Springs to Cisco


def load_graph():
    with open("graphs/colorado/master_compressed_river_graph.pkl", "rb") as f:
        G = pickle.load(f)
    a = next(iter(G.nodes(data=True)))[1]
    lat0, lon0 = a["latitude"] + a["row"] / 1200.0, a["longitude"] - a["col"] / 1200.0

    def cells(path):
        rc = np.array([divmod(int(n), 15001) for n in path], float)
        return np.column_stack([lon0 + rc[:, 1] / 1200.0, lat0 - rc[:, 0] / 1200.0])

    kind = {}
    for n, d in G.nodes(data=True):
        if d.get("usgs_clusters"):
            kind[n] = "station"
        elif d.get("support_gauges"):
            kind[n] = "gauge"
        elif G.degree(n) >= 3:
            kind[n] = "confluence"
        else:
            kind[n] = "end"
    return G, cells, kind


NODE_STYLE = {   # marker, size, face, edge, zorder, legend label
    "station": ("o", 120, GRAPH, SURFACE, 6, "Water-quality station (26)"),
    "gauge": ("o", 90, GAUGE, SURFACE, 6, "Flow gauge only (3)"),
    "confluence": ("D", 55, CONFL, SURFACE, 5, "Confluence (18)"),
    "end": ("o", 40, SURFACE, MUTED, 4, "Source or outlet (20)"),
}


def draw_nodes(ax, G, kind, scale=1.0, lw=1.5):
    for k, (m, sz, fc, ec, z, _) in NODE_STYLE.items():
        ns = [n for n in G if kind[n] == k]
        ax.scatter([G.nodes[n]["longitude"] for n in ns], [G.nodes[n]["latitude"] for n in ns],
                   marker=m, s=sz * scale, color=fc, edgecolor=ec, lw=lw, zorder=z)


def node_legend(fig_or_ax, **kw):
    h = [plt.Line2D([], [], ls="", marker=m, ms=np.sqrt(sz) * 0.85, mfc=fc, mec=ec if ec != SURFACE else fc,
                    mew=1.3, label=lab) for m, sz, fc, ec, _, lab in NODE_STYLE.values()]
    return fig_or_ax.legend(handles=h, frameon=False, **kw)


def geo_axes(ax):
    ax.set_aspect(1 / np.cos(np.radians(38)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def scalebar(ax, lon, lat, km, lat_ref=38.0):
    deg = km / (111.32 * np.cos(np.radians(lat_ref)))
    ax.plot([lon, lon + deg], [lat, lat], color=INK2, lw=2, solid_capstyle="butt")
    ax.text(lon + deg / 2, lat, f"{km} km", ha="center", va="bottom", fontsize=10.5, color=INK2,
            transform=ax.transData, bbox=dict(fc="none", ec="none", pad=4))


def fig_graph():
    G, cells, kind = load_graph()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 8.6))
    for ax in (a1, a2):
        geo_axes(ax)
        ax.set_xlim(-115.5, -105.6); ax.set_ylim(31.6, 42.4)
    # left: the river, as the model's 67,401 MERIT grid cells
    for u, v, d in G.edges(data=True):
        xy = cells(d["original_path"])
        a1.plot(xy[:, 0], xy[:, 1], color=GRAPH, lw=1.6, alpha=0.55, zorder=2)
    draw_nodes(a1, G, kind, scale=0.8)
    x0, x1, y0, y1 = ZOOM
    a1.add_patch(mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK, lw=1.2, zorder=7))
    a1.annotate("zoomed in on\nthe next figure", (x1, y1), xytext=(8, 4), textcoords="offset points",
                fontsize=10.5, color=INK2, ha="left", va="bottom")
    scalebar(a1, -115.2, 32.0, 200)
    a1.set_title("The river: 67,401 grid cells (90 m)", loc="left", fontweight="bold", color=INK, pad=8)
    # right: the graph; each edge is a whole river reach, drawn straight, pointing downstream
    for u, v, d in G.edges(data=True):
        a2.annotate("", xy=(G.nodes[v]["longitude"], G.nodes[v]["latitude"]),
                    xytext=(G.nodes[u]["longitude"], G.nodes[u]["latitude"]),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color=MUTED, lw=1.3,
                                    shrinkA=5, shrinkB=6), zorder=3)
    draw_nodes(a2, G, kind, scale=0.8)
    a2.set_title("The graph: 67 nodes, 66 edges", loc="left", fontweight="bold", color=INK, pad=8)
    a2.text(-115.3, 41.9, "Arrows point downstream.\nEach edge stands for one river reach\n"
            "and is weighted by its length.", fontsize=10.5, color=INK2, va="top", linespacing=1.35)
    node_legend(fig, loc="lower left", bbox_to_anchor=(0.0, 0.905), ncol=4, fontsize=11,
                handletextpad=0.2, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.04, top=0.88)
    title(fig, "From river to graph: the Colorado basin as a network",
          "Only stations, confluences and river ends are kept as nodes; "
          "everything between them collapses into a single weighted edge")
    save(fig, "poster_4_graph")


def fig_zoom():
    G, cells, kind = load_graph()
    x0, x1, y0, y1 = ZOOM
    fig, ax = plt.subplots(figsize=(14, 8.0))
    fig.subplots_adjust(top=0.97, bottom=0.02, left=0.01, right=0.99)
    geo_axes(ax)
    ax.set_anchor("N")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    inside = lambda xy: (xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1)
    paths = []
    for u, v, d in G.edges(data=True):
        xy = cells(d["original_path"])
        if not inside(xy).any():
            continue
        paths.append(xy)
        a, b = G.nodes[u], G.nodes[v]
        ax.plot([a["longitude"], b["longitude"]], [a["latitude"], b["latitude"]], color=MUTED, lw=1.1,
                ls=(0, (3, 3)), zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=1.6, color=GRAPH, alpha=0.6, lw=0, zorder=2)
        m = xy[len(xy) // 2]
        if inside(m[None])[0] and d["length_m"] > 30000:
            ax.text(m[0], m[1], f"{d['length_m'] / 1000:.0f} km", fontsize=11, color=INK, ha="center",
                    va="center", zorder=8, bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=GRID, lw=0.8))
    draw_nodes(ax, G, kind, scale=1.6, lw=2)
    for text, xy in [("Colorado River", (-107.62, 39.62)), ("Gunnison River", (-108.05, 38.62)),
                     ("Dolores River", (-109.02, 38.62))]:
        ax.text(*xy, text, fontsize=12, color=INK2, style="italic", ha="center", va="center", zorder=8)
    gj = G.nodes[92101856]
    ax.annotate("Gunnison River joins\nat Grand Junction", (gj["longitude"], gj["latitude"]),
                xytext=(-108.95, 38.9), textcoords="data", fontsize=11, color=INK, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=7), zorder=8)
    scalebar(ax, x1 - 0.45, y0 + 0.05, 25, lat_ref=39.2)

    # inset: the cells themselves, around the Gunnison confluence
    ix0, ix1, iy0, iy1 = -108.63, -108.50, 39.025, 39.095
    ax.add_patch(mpl.patches.Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, fill=False, ec=INK, lw=1, zorder=9))
    ins = ax.inset_axes([0.0, 0.50, 0.30, 0.46])
    for xy in paths:
        k = (xy[:, 0] > ix0) & (xy[:, 0] < ix1) & (xy[:, 1] > iy0) & (xy[:, 1] < iy1)
        ins.scatter(xy[k, 0], xy[k, 1], marker="s", s=14, color=GRAPH, alpha=0.75, lw=0, zorder=2)
    draw_nodes(ins, G, kind, scale=2.2, lw=2.2)
    geo_axes(ins)
    ins.set_xlim(ix0, ix1); ins.set_ylim(iy0, iy1)
    for sp in ins.spines.values():
        sp.set_visible(True); sp.set_color(INK); sp.set_linewidth(1)
    ins.set_facecolor(SURFACE)
    ins.set_title("Close-up: every square is one 90 m river cell", fontsize=11, color=INK, loc="left", pad=4)
    ax.indicate_inset(bounds=(ix0, iy0, ix1 - ix0, iy1 - iy0), inset_ax=ins, edgecolor=MUTED, lw=0.8)

    h = [plt.Line2D([], [], ls="", marker="s", ms=5, mfc=GRAPH, mec=GRAPH, alpha=0.75,
                    label="River grid cell (90 m)"),
         plt.Line2D([], [], color=MUTED, lw=1.1, ls=(0, (3, 3)), label="Graph edge, drawn straight"),
         *[plt.Line2D([], [], ls="", marker=m, ms=np.sqrt(sz) * 0.85, mfc=fc, mec=ec if ec != SURFACE else fc,
                      mew=1.3, label=lab.split(" (")[0]) for m, sz, fc, ec, _, lab in NODE_STYLE.values()
           if lab.split(" (")[0] != "Source or outlet"]]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.12), frameon=False, fontsize=11.5)
    title(fig, "Nodes sit on the river; edges follow it",
          "Colorado River from Glenwood Springs to Cisco, Utah. Each graph edge (dashed) stands for the "
          "chain of river cells between two nodes, labelled with its length.")
    save(fig, "poster_5_zoom")


def main():
    sc = pd.read_csv("results/split_comparison.csv")
    fig_models(sc)
    fig_splits(sc)
    fig_map()
    fig_graph()
    fig_zoom()


if __name__ == "__main__":
    main()

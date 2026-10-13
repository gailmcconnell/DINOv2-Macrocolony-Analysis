"""
Created on Tue Oct  6 15:10:17 2026

@author: Gail McConnell, SIPBS, University of Strathclyde
"""

from pathlib import Path
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import linregress, f_oneway, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

# Easiest option: put this script in the same folder as DINO_tile_features.csv.
ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "DINO_results" / "DINO_tile_features.csv"


OUTPUT_DIR = ROOT / "Analysis_output"

# Tiles below this fraction of colony signal are excluded from
# signal-restricted spatial/radial analyses.
SIGNAL_THRESHOLD = 0.50

# Sensitivity thresholds used to show whether conclusions depend on cutoff.
SENSITIVITY_THRESHOLDS = [0.25, 0.50, 0.75, 0.90]

# Reproducibility
RANDOM_STATE = 42

# Plot export
DPI = 300

# Set font
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def safe_mkdir(p):
    p.mkdir(parents=True, exist_ok=True)


def savefig(fig, pathbase):
    """Save both PNG and PDF."""
    fig.tight_layout()
    fig.savefig(str(pathbase) + ".png", dpi=DPI, bbox_inches="tight")
    fig.savefig(str(pathbase) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def find_column(df, candidates, required=True):
    """Case-insensitive candidate-column finder."""
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )
    return None


def infer_feature_columns(df):
    """
    Identify numeric DINO feature columns while excluding known metadata.
    Prefers common names such as feature_0, feat_0, dino_0, embedding_0.
    """
    patterns = [
        r"^feature[_\-]?\d+$",
        r"^feat[_\-]?\d+$",
        r"^dino[_\-]?\d+$",
        r"^embedding[_\-]?\d+$",
        r"^f\d+$",
    ]
    matches = []
    for c in df.columns:
        lc = c.lower()
        if any(re.match(p, lc) for p in patterns):
            matches.append(c)

    if len(matches) >= 32:
        def numkey(x):
            m = re.search(r"(\d+)$", x)
            return int(m.group(1)) if m else 10**9
        return sorted(matches, key=numkey)

    # Conservative fallback: numeric columns excluding obvious metadata.
    metadata_words = {
        "row", "col", "tile_row", "tile_col", "x", "y",
        "signal_fraction", "replicate", "rep", "strain", "medium",
        "colony", "image", "filename", "file", "id", "index"
    }
    numeric = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if c.lower() not in metadata_words:
            numeric.append(c)
    if len(numeric) < 32:
        raise ValueError("Could not reliably identify DINO feature columns.")
    return numeric


def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    return multipletests(pvals, method="fdr_bh")[1]


def eta_squared_from_anova(table):
    """Add eta squared for ordinary ANOVA table."""
    ss_total = table["sum_sq"].sum()
    out = table.copy()
    out["eta_sq"] = out["sum_sq"] / ss_total if ss_total > 0 else np.nan
    return out


def cohen_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a)-1)*s1 + (len(b)-1)*s2) /
                     (len(a)+len(b)-2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def add_covariance_ellipse(ax, x, y, color, n_std=2.0,
                           linewidth=1.5, alpha=0.18):
    """
    Add a covariance-based data ellipse centred on the group mean.
    The ellipse semi-axes are n_std times the standard deviations
    along the covariance eigenvectors.
    """
    xy = np.column_stack([np.asarray(x, dtype=float),
                          np.asarray(y, dtype=float)])
    xy = xy[np.isfinite(xy).all(axis=1)]

    if len(xy) < 3:
        return

    centre = xy.mean(axis=0)
    cov = np.cov(xy, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0)
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_std * np.sqrt(eigvals[0])
    height = 2 * n_std * np.sqrt(eigvals[1])

    ellipse = Ellipse(
        xy=centre,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=1
    )
    ax.add_patch(ellipse)


# ---------------------------------------------------------------------
# OUTPUT FOLDERS
# ---------------------------------------------------------------------

folders = {
    "qc": OUTPUT_DIR / "00_QC",
    "global": OUTPUT_DIR / "01_GLOBAL_PHENOTYPE",
    "maps": OUTPUT_DIR / "02_SPATIAL_MAPS",
    "radial": OUTPUT_DIR / "03_RADIAL_PHENOTYPE",
    "stats": OUTPUT_DIR / "04_STATISTICS",
    "figs": OUTPUT_DIR / "05_FIGURES",
}
for p in [OUTPUT_DIR, *folders.values()]:
    safe_mkdir(p)


# ---------------------------------------------------------------------
# LOAD AND IDENTIFY COLUMNS
# ---------------------------------------------------------------------

if not INPUT_CSV.exists():
    # Friendly fallback for the filename supplied in this chat.
    alternatives = list(Path(".").glob("*DINO*tile*features*.csv"))
    if len(alternatives) == 1:
        INPUT_CSV = alternatives[0]
    else:
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV.resolve()}.\n"
            "Put this script beside the DINO tile-feature CSV or edit INPUT_CSV."
        )

df = pd.read_csv(INPUT_CSV)

strain_col = find_column(df, ["strain", "genotype"])
medium_col = find_column(df, ["medium", "media", "condition"])
rep_col = find_column(df, ["replicate", "rep", "biological_replicate"])
row_col = find_column(df, ["tile_row", "row"])
col_col = find_column(df, ["tile_col", "col", "column"])
x_col = find_column(df, ["x", "tile_x", "x_coord"])
y_col = find_column(df, ["y", "tile_y", "y_coord"])
signal_col = find_column(df, ["signal_fraction", "signal", "foreground_fraction"])
file_col = find_column(
    df, ["filename", "file", "image", "image_name", "source_image"],
    required=False
)
colony_col = find_column(
    df, ["colony", "colony_id", "sample", "sample_id"],
    required=False
)

feature_cols = infer_feature_columns(df)

# Standardise metadata names internally.
work = df.copy()
work["_strain"] = work[strain_col].astype(str)
work["_medium"] = work[medium_col].astype(str)
work["_rep"] = work[rep_col].astype(str)
work["_row"] = pd.to_numeric(work[row_col])
work["_col"] = pd.to_numeric(work[col_col])
work["_x"] = pd.to_numeric(work[x_col])
work["_y"] = pd.to_numeric(work[y_col])
work["_signal"] = pd.to_numeric(work[signal_col])

if colony_col:
    work["_colony"] = work[colony_col].astype(str)
elif file_col:
    # If tile filenames vary within a colony, this may not be sufficient,
    # so use strain-medium-replicate as the robust biological identifier.
    work["_colony"] = (
        work["_strain"] + "__" + work["_medium"] + "__R" + work["_rep"]
    )
else:
    work["_colony"] = (
        work["_strain"] + "__" + work["_medium"] + "__R" + work["_rep"]
    )

# Guarantee uniqueness at the biological-replicate level.
work["_bio_id"] = (
    work["_strain"] + "__" + work["_medium"] + "__R" + work["_rep"]
)

X = work[feature_cols].apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    raise ValueError("Missing/non-numeric values detected in DINO feature columns.")


# ---------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------

qc_lines = []
qc_lines.append("DINOv2 MACROCOLONY MASTER ANALYSIS — QC")
qc_lines.append("=" * 60)
qc_lines.append(f"Input file: {INPUT_CSV.resolve()}")
qc_lines.append(f"Rows: {len(work)}")
qc_lines.append(f"DINO features: {len(feature_cols)}")
qc_lines.append(f"Strains: {sorted(work['_strain'].unique())}")
qc_lines.append(f"Media: {sorted(work['_medium'].unique())}")
qc_lines.append(f"Biological colonies: {work['_bio_id'].nunique()}")
qc_lines.append("")

counts = (
    work.groupby(["_strain", "_medium", "_rep"])
        .size()
        .reset_index(name="n_tiles")
)
counts.to_csv(folders["qc"] / "tiles_per_colony.csv", index=False)

qc_lines.append("Tiles per colony:")
qc_lines.append(str(counts["n_tiles"].describe()))
qc_lines.append("")

design = (
    work[["_strain", "_medium", "_rep", "_bio_id"]]
    .drop_duplicates()
    .groupby(["_strain", "_medium"])
    .size()
    .unstack(fill_value=0)
)
design.to_csv(folders["qc"] / "experimental_design.csv")
qc_lines.append("Biological replicates per strain x medium:")
qc_lines.append(design.to_string())
qc_lines.append("")

# Signal fraction distribution
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(work["_signal"], bins=30)
ax.axvline(SIGNAL_THRESHOLD, linestyle="--")
ax.set_xlabel("Signal fraction")
ax.set_ylabel("Tile count")
ax.set_title("Tile signal-fraction QC")
savefig(fig, folders["qc"] / "signal_fraction_distribution")

signal_summary = (
    work.groupby(["_bio_id", "_strain", "_medium"])
        .agg(
            n_tiles=("_signal", "size"),
            mean_signal=("_signal", "mean"),
            median_signal=("_signal", "median"),
            min_signal=("_signal", "min"),
            n_signal_tiles=("_signal", lambda z: int((z >= SIGNAL_THRESHOLD).sum()))
        )
        .reset_index()
)
signal_summary.to_csv(folders["qc"] / "signal_summary_by_colony.csv", index=False)

with open(folders["qc"] / "QC_REPORT.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(qc_lines))


# ---------------------------------------------------------------------
# GLOBAL COLONY PHENOTYPE
# Mean-pool DINO tile embeddings to one vector per biological colony.
# Also calculate a signal-weighted version as sensitivity.
# ---------------------------------------------------------------------

feature_matrix = work[feature_cols].to_numpy(dtype=float)

# Unweighted colony means
global_unweighted = (
    work.groupby(["_bio_id", "_strain", "_medium", "_rep"])[feature_cols]
        .mean()
        .reset_index()
)

# Signal-weighted colony means
weighted_rows = []
for keys, g in work.groupby(["_bio_id", "_strain", "_medium", "_rep"]):
    weights = np.clip(g["_signal"].to_numpy(dtype=float), 0, 1)
    feats = g[feature_cols].to_numpy(dtype=float)
    if weights.sum() == 0:
        pooled = feats.mean(axis=0)
    else:
        pooled = np.average(feats, axis=0, weights=weights)
    rec = {
        "_bio_id": keys[0], "_strain": keys[1],
        "_medium": keys[2], "_rep": keys[3]
    }
    rec.update(dict(zip(feature_cols, pooled)))
    weighted_rows.append(rec)

global_weighted = pd.DataFrame(weighted_rows)
global_unweighted.to_csv(
    folders["global"] / "colony_embeddings_mean_pooled.csv", index=False
)
global_weighted.to_csv(
    folders["global"] / "colony_embeddings_signal_weighted.csv", index=False
)

# PCA on colony-level embeddings.
# PCA does not require feature standardisation here because DINO embedding
# dimensions are on a common representation scale; centring is intrinsic.
G = global_weighted[feature_cols].to_numpy(dtype=float)
pca_global = PCA(n_components=min(10, len(global_weighted)-1), random_state=RANDOM_STATE)
Gpcs = pca_global.fit_transform(G)

global_scores = global_weighted[
    ["_bio_id", "_strain", "_medium", "_rep"]
].copy()
for i in range(Gpcs.shape[1]):
    global_scores[f"PC{i+1}"] = Gpcs[:, i]

global_scores.to_csv(
    folders["global"] / "colony_level_PCA_scores.csv", index=False
)

loadings = pd.DataFrame(
    pca_global.components_.T,
    index=feature_cols,
    columns=[f"PC{i+1}" for i in range(Gpcs.shape[1])]
)
loadings.to_csv(folders["global"] / "colony_level_PCA_loadings.csv")

var_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(Gpcs.shape[1])],
    "variance_explained": pca_global.explained_variance_ratio_,
    "cumulative_variance": np.cumsum(pca_global.explained_variance_ratio_)
})
var_df.to_csv(folders["global"] / "PCA_variance_explained.csv", index=False)

# Global PCA plot
# ABC3D-matched styling:
#   BW25113 = orange
#   ΔamiA   = blue
#   ΔompR   = green
#   ΔydgD   = purple
# Medium is encoded by fill: LB filled, M9 hollow.
# Ellipses are covariance-based data ellipses at two standard deviations.
fig, ax = plt.subplots(figsize=(8, 6))

strain_colours = {
    "BW25113": "#ff9800",
    "amiA": "#0000ff",
    "ompR": "#008000",
    "ydgD": "#800080",
}

strain_markers = {
    "BW25113": "o",
    "amiA": "s",
    "ompR": "^",
    "ydgD": "D",
}

strain_labels = {
    "BW25113": "BW25113",
    "amiA": r"$\Delta amiA$",
    "ompR": r"$\Delta ompR$",
    "ydgD": r"$\Delta ydgD$",
}

strain_order = ["BW25113", "amiA", "ompR", "ydgD"]
medium_order = ["LB", "M9"]

for strain in strain_order:
    for medium in medium_order:
        g = global_scores[
            (global_scores["_strain"] == strain) &
            (global_scores["_medium"] == medium)
        ]

        if g.empty:
            continue

        colour = strain_colours[strain]
        marker = strain_markers[strain]

        # Two-standard-deviation covariance ellipse for each strain x medium group.
        # Match the ABC3D convention:
        #   LB = translucent filled ellipse with solid outline
        #   M9 = unfilled ellipse with dashed outline
        xy = np.column_stack([
            g["PC1"].to_numpy(dtype=float),
            g["PC2"].to_numpy(dtype=float)
        ])
        xy = xy[np.isfinite(xy).all(axis=1)]

        if len(xy) >= 3:
            centre = xy.mean(axis=0)
            cov = np.cov(xy, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[order], 0)
            eigvecs = eigvecs[:, order]

            angle = np.degrees(
                np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
            )
            width = 4.0 * np.sqrt(eigvals[0])
            height = 4.0 * np.sqrt(eigvals[1])

            ellipse = Ellipse(
                xy=centre,
                width=width,
                height=height,
                angle=angle,
                facecolor=colour if medium == "LB" else "none",
                edgecolor=colour,
                linewidth=1.5,
                linestyle="-" if medium == "LB" else "--",
                alpha=0.18 if medium == "LB" else 1.0,
                zorder=1
            )
            ax.add_patch(ellipse)

        # LB = filled; M9 = hollow.
        face = colour if medium == "LB" else "none"

        ax.scatter(
            g["PC1"],
            g["PC2"],
            label=f"{strain_labels[strain]} – {medium}",
            marker=marker,
            s=65,
            facecolors=face,
            edgecolors=colour,
            linewidths=1.5,
            alpha=1.0,
            zorder=3
        )

ax.set_xlabel(
    f"PC1 ({100*pca_global.explained_variance_ratio_[0]:.1f}%)",
    fontsize=11
)
ax.set_ylabel(
    f"PC2 ({100*pca_global.explained_variance_ratio_[1]:.1f}%)",
    fontsize=11
)
ax.tick_params(axis="both", labelsize=10)

# No plot title for the publication figure.
ax.legend(
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

savefig(fig, folders["global"] / "FIG2_colony_level_DINO_PCA")


# ---------------------------------------------------------------------
# GLOBAL PHENOTYPE STATISTICS
# Analyze leading PCs that together explain >=80% variance.
# Each colony is one observation.
# ---------------------------------------------------------------------

cum = np.cumsum(pca_global.explained_variance_ratio_)
n80 = int(np.searchsorted(cum, 0.80) + 1)
n80 = max(2, min(n80, Gpcs.shape[1]))

anova_rows = []
for pc_i in range(1, n80 + 1):
    pc = f"PC{pc_i}"
    tmp = global_scores.rename(
        columns={"_strain": "strain", "_medium": "medium"}
    )
    model = ols(f"{pc} ~ C(strain) * C(medium)", data=tmp).fit()
    tab = sm.stats.anova_lm(model, typ=2)
    tab = eta_squared_from_anova(tab)
    for effect, row in tab.iterrows():
        if effect == "Residual":
            continue
        anova_rows.append({
            "PC": pc,
            "effect": effect,
            "sum_sq": row["sum_sq"],
            "df": row["df"],
            "F": row["F"],
            "p": row["PR(>F)"],
            "eta_sq": row["eta_sq"],
            "PC_variance_explained": pca_global.explained_variance_ratio_[pc_i-1]
        })

global_anova = pd.DataFrame(anova_rows)
if len(global_anova):
    global_anova["p_FDR"] = bh_adjust(global_anova["p"])
global_anova.to_csv(
    folders["stats"] / "global_DINO_PC_two_way_ANOVA.csv", index=False
)


# ---------------------------------------------------------------------
# TILE-LEVEL PCA FOR SPATIAL MAPPING
#
# Fit PCA on signal-containing tiles only to prevent empty background from
# defining the main spatial axes. Transform ALL tiles afterward for QC/maps.
# ---------------------------------------------------------------------

signal_mask = work["_signal"] >= SIGNAL_THRESHOLD
tile_pca = PCA(n_components=5, random_state=RANDOM_STATE)
tile_pca.fit(work.loc[signal_mask, feature_cols].to_numpy(dtype=float))
tile_scores_all = tile_pca.transform(work[feature_cols].to_numpy(dtype=float))

for i in range(tile_scores_all.shape[1]):
    work[f"_tilePC{i+1}"] = tile_scores_all[:, i]

tile_var = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(tile_scores_all.shape[1])],
    "variance_explained": tile_pca.explained_variance_ratio_
})
tile_var.to_csv(folders["maps"] / "tile_PCA_variance_explained.csv", index=False)

# Retain compact tile-level derived table
derived_cols = [
    "_bio_id", "_strain", "_medium", "_rep",
    "_row", "_col", "_x", "_y", "_signal"
] + [f"_tilePC{i+1}" for i in range(tile_scores_all.shape[1])]
work[derived_cols].to_csv(
    folders["maps"] / "tile_spatial_PCA_scores.csv", index=False
)


# ---------------------------------------------------------------------
# COLONY CENTRE + NORMALISED RADIAL POSITION
#
# Centre is signal-weighted centroid of tile centres.
# Radial position is distance from centroid divided by the 95th percentile
# signal-tile distance within that colony; values clipped at 1.
# This is more defensible than using square-image boundaries.
# ---------------------------------------------------------------------

radial_parts = []
for bio_id, g0 in work.groupby("_bio_id"):
    g = g0.copy()
    sig = np.clip(g["_signal"].to_numpy(dtype=float), 0, 1)

    # Use signal as weights; fallback to grid mean.
    if sig.sum() > 0:
        cx = np.average(g["_x"], weights=sig)
        cy = np.average(g["_y"], weights=sig)
    else:
        cx = g["_x"].mean()
        cy = g["_y"].mean()

    g["_centre_x"] = cx
    g["_centre_y"] = cy
    g["_radius_px"] = np.sqrt((g["_x"] - cx)**2 + (g["_y"] - cy)**2)

    valid = g["_signal"] >= SIGNAL_THRESHOLD
    if valid.sum() >= 3:
        edge_r = np.percentile(g.loc[valid, "_radius_px"], 95)
    else:
        edge_r = g["_radius_px"].max()

    if edge_r <= 0:
        edge_r = 1.0

    g["_radial_position"] = np.clip(g["_radius_px"] / edge_r, 0, 1)
    radial_parts.append(g)

work = pd.concat(radial_parts, ignore_index=True)

work[[
    "_bio_id", "_strain", "_medium", "_rep",
    "_row", "_col", "_x", "_y", "_signal",
    "_centre_x", "_centre_y", "_radius_px", "_radial_position",
    "_tilePC1", "_tilePC2", "_tilePC3"
]].to_csv(
    folders["radial"] / "tile_radial_positions_and_DINO_scores.csv",
    index=False
)


# ---------------------------------------------------------------------
# SPATIAL MAPS FOR ALL 40 COLONIES
# Map tile PC1 because it is the dominant learned morphological axis.
# Also save signal-fraction maps as QC.
# ---------------------------------------------------------------------

map_dir_all = folders["maps"] / "all_colonies"
safe_mkdir(map_dir_all)

def grid_from_group(g, value_col):
    rows = sorted(g["_row"].unique())
    cols = sorted(g["_col"].unique())
    mat = np.full((len(rows), len(cols)), np.nan)
    rmap = {r: i for i, r in enumerate(rows)}
    cmap = {c: i for i, c in enumerate(cols)}
    for _, z in g.iterrows():
        mat[rmap[z["_row"]], cmap[z["_col"]]] = z[value_col]
    return mat

# Use a common PC1 scale across signal tiles so maps are comparable.
pc1_valid = work.loc[work["_signal"] >= SIGNAL_THRESHOLD, "_tilePC1"]
pc1_lim = np.nanpercentile(np.abs(pc1_valid), 98)

for bio_id, g in work.groupby("_bio_id"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    sigmat = grid_from_group(g, "_signal")
    pc1mat = grid_from_group(g, "_tilePC1")
    # Mask low-signal tiles in phenotype map.
    pc1masked = np.where(sigmat >= SIGNAL_THRESHOLD, pc1mat, np.nan)

    im0 = axes[0].imshow(sigmat, origin="upper", vmin=0, vmax=1)
    axes[0].set_title("Signal fraction")
    axes[0].set_xlabel("Tile column")
    axes[0].set_ylabel("Tile row")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(
        pc1masked, origin="upper",
        vmin=-pc1_lim, vmax=pc1_lim,
        cmap="coolwarm"
    )
    axes[1].set_title("DINO spatial phenotype (tile PC1)")
    axes[1].set_xlabel("Tile column")
    axes[1].set_ylabel("Tile row")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    strain = g["_strain"].iloc[0]
    medium = g["_medium"].iloc[0]
    rep = g["_rep"].iloc[0]
    fig.suptitle(f"{strain} | {medium} | replicate {rep}")

    savefig(fig, map_dir_all / re.sub(r"[^A-Za-z0-9_.-]", "_", bio_id))


# ---------------------------------------------------------------------
# AUTOMATIC REPRESENTATIVE COLONY PER STRAIN x MEDIUM
#
# Representative = colony closest to its group's centroid in global DINO
# embedding space (first n80 PCs).
# ---------------------------------------------------------------------

rep_records = []
for (strain, medium), g in global_scores.groupby(["_strain", "_medium"]):
    pcs = [f"PC{i}" for i in range(1, n80+1)]
    arr = g[pcs].to_numpy()
    centroid = arr.mean(axis=0)
    dist = np.linalg.norm(arr - centroid, axis=1)
    idx = np.argmin(dist)
    chosen = g.iloc[idx]
    rep_records.append({
        "strain": strain,
        "medium": medium,
        "bio_id": chosen["_bio_id"],
        "replicate": chosen["_rep"],
        "distance_to_group_centroid": dist[idx]
    })

representatives = pd.DataFrame(rep_records)
representatives.to_csv(
    folders["figs"] / "representative_colonies.csv", index=False
)

# 8-panel representative DINO spatial map figure
# Publication styling: same strain ordering as Figure 2, no plot heading.
plot_strains = [s for s in strain_order if s in set(work["_strain"])]
plot_media = [m for m in medium_order if m in set(work["_medium"])]

fig, axes = plt.subplots(
    len(plot_strains), len(plot_media),
    figsize=(4*len(plot_media), 3.5*len(plot_strains)),
    squeeze=False
)

for i, strain in enumerate(plot_strains):
    for j, medium in enumerate(plot_media):
        rec = representatives[
            (representatives["strain"] == strain) &
            (representatives["medium"] == medium)
        ].iloc[0]
        g = work[work["_bio_id"] == rec["bio_id"]]
        sigmat = grid_from_group(g, "_signal")
        pc1mat = grid_from_group(g, "_tilePC1")
        masked = np.where(sigmat >= SIGNAL_THRESHOLD, pc1mat, np.nan)
        im = axes[i, j].imshow(
            masked, origin="upper",
            vmin=-pc1_lim, vmax=pc1_lim,
            cmap="coolwarm"
        )

        # Column labels identify medium; row labels identify strain.
        if i == 0:
            axes[i, j].set_title(medium, fontsize=11)
        if j == 0:
            axes[i, j].set_ylabel(
                strain_labels.get(strain, strain),
                fontsize=11,
                rotation=90,
                labelpad=8
            )

        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

# Shared colourbar; no overall figure heading.
cbar = fig.colorbar(
    im, ax=axes.ravel().tolist(),
    fraction=0.02, pad=0.02
)
cbar.set_label("DINO tile PC1", fontsize=11)
cbar.ax.tick_params(labelsize=10)

fig.savefig(
    folders["figs"] / "FIG3_representative_spatial_maps.png",
    dpi=DPI, bbox_inches="tight"
)
fig.savefig(
    folders["figs"] / "FIG3_representative_spatial_maps.pdf",
    bbox_inches="tight"
)
plt.close(fig)


# ---------------------------------------------------------------------
# RADIAL PHENOTYPE
#
# Primary per-colony metric:
# slope of tile PC1 vs normalised radial position, using signal tiles only.
#
# Also calculate centre-edge delta using inner <=0.33 and outer >=0.67.
# These are colony-level summary phenotypes, preserving n=5/group.
# ---------------------------------------------------------------------

def radial_metrics_for_threshold(data, threshold):
    rows = []
    for bio_id, g0 in data.groupby("_bio_id"):
        g = g0[g0["_signal"] >= threshold].copy()

        rec = {
            "_bio_id": bio_id,
            "_strain": g0["_strain"].iloc[0],
            "_medium": g0["_medium"].iloc[0],
            "_rep": g0["_rep"].iloc[0],
            "signal_threshold": threshold,
            "n_tiles_used": len(g)
        }

        if len(g) >= 5 and g["_radial_position"].nunique() >= 3:
            lr = linregress(g["_radial_position"], g["_tilePC1"])
            rec.update({
                "radial_PC1_slope": lr.slope,
                "radial_PC1_intercept": lr.intercept,
                "radial_PC1_r": lr.rvalue,
                "radial_PC1_p_descriptive": lr.pvalue
            })
        else:
            rec.update({
                "radial_PC1_slope": np.nan,
                "radial_PC1_intercept": np.nan,
                "radial_PC1_r": np.nan,
                "radial_PC1_p_descriptive": np.nan
            })

        inner = g[g["_radial_position"] <= 0.33]["_tilePC1"]
        outer = g[g["_radial_position"] >= 0.67]["_tilePC1"]
        rec["inner_PC1_mean"] = inner.mean() if len(inner) else np.nan
        rec["outer_PC1_mean"] = outer.mean() if len(outer) else np.nan
        rec["outer_minus_inner_PC1"] = (
            rec["outer_PC1_mean"] - rec["inner_PC1_mean"]
            if np.isfinite(rec["inner_PC1_mean"]) and
               np.isfinite(rec["outer_PC1_mean"])
            else np.nan
        )
        rows.append(rec)
    return pd.DataFrame(rows)

sensitivity = pd.concat(
    [radial_metrics_for_threshold(work, t)
     for t in SENSITIVITY_THRESHOLDS],
    ignore_index=True
)
sensitivity.to_csv(
    folders["qc"] / "radial_metric_signal_threshold_sensitivity.csv",
    index=False
)

radial_metrics = sensitivity[
    np.isclose(sensitivity["signal_threshold"], SIGNAL_THRESHOLD)
].copy()
radial_metrics.to_csv(
    folders["radial"] / "colony_radial_DINO_metrics.csv", index=False
)


# ---------------------------------------------------------------------
# RADIAL TRAJECTORIES
# Bin tiles within each colony first, then average COLONIES within group.
# This prevents tile pseudoreplication.
# ---------------------------------------------------------------------

bins = np.linspace(0, 1, 6)
labels = (bins[:-1] + bins[1:]) / 2

valid = work[work["_signal"] >= SIGNAL_THRESHOLD].copy()
valid["_radial_bin"] = pd.cut(
    valid["_radial_position"],
    bins=bins,
    labels=labels,
    include_lowest=True
)
valid["_radial_bin"] = valid["_radial_bin"].astype(float)

colony_bin = (
    valid.groupby(
        ["_bio_id", "_strain", "_medium", "_rep", "_radial_bin"],
        observed=True
    )["_tilePC1"]
    .mean()
    .reset_index(name="mean_tile_PC1")
)
colony_bin.to_csv(
    folders["radial"] / "colony_radial_binned_PC1.csv", index=False
)

group_bin = (
    colony_bin.groupby(
        ["_strain", "_medium", "_radial_bin"], observed=True
    )["mean_tile_PC1"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
group_bin["sem"] = group_bin["std"] / np.sqrt(group_bin["count"])
group_bin.to_csv(
    folders["radial"] / "group_radial_trajectories.csv", index=False
)

# Figure 4: radial trajectories, one publication-ready panel per medium.
# Colours and strain markers match Figure 2. No plot headings.
for medium in medium_order:
    gm = group_bin[group_bin["_medium"] == medium]
    if gm.empty:
        continue

    fig, ax = plt.subplots(figsize=(7, 5))

    for strain in strain_order:
        gs = gm[gm["_strain"] == strain].sort_values("_radial_bin")
        if gs.empty:
            continue

        colour = strain_colours[strain]
        marker = strain_markers[strain]

        ax.errorbar(
            gs["_radial_bin"],
            gs["mean"],
            yerr=gs["sem"],
            color=colour,
            marker=marker,
            markerfacecolor=colour,
            markeredgecolor=colour,
            markersize=6,
            linewidth=1.5,
            capsize=3,
            label=strain_labels[strain]
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(
        "Normalised radial position (0 = centre, 1 = edge)",
        fontsize=11
    )
    ax.set_ylabel("DINO spatial phenotype (tile PC1)", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9, frameon=False)

    # No plot title: medium is encoded in the filename/panel designation.
    savefig(
        fig,
        folders["radial"] /
        f"FIG4_radial_trajectory_{re.sub(r'[^A-Za-z0-9_.-]', '_', medium)}"
    )


# ---------------------------------------------------------------------
# TWO-WAY ANOVA ON COLONY-LEVEL SPATIAL METRICS
# ---------------------------------------------------------------------

spatial_anova_tables = []
for metric in ["radial_PC1_slope", "outer_minus_inner_PC1"]:
    tmp = radial_metrics.dropna(subset=[metric]).rename(
        columns={"_strain": "strain", "_medium": "medium"}
    )
    if len(tmp) < 8:
        continue
    model = ols(f"{metric} ~ C(strain) * C(medium)", data=tmp).fit()
    tab = sm.stats.anova_lm(model, typ=2)
    tab = eta_squared_from_anova(tab).reset_index().rename(
        columns={"index": "effect", "PR(>F)": "p"}
    )
    tab["metric"] = metric
    spatial_anova_tables.append(tab)

if spatial_anova_tables:
    spatial_anova = pd.concat(spatial_anova_tables, ignore_index=True)
    spatial_anova.to_csv(
        folders["stats"] / "spatial_metrics_two_way_ANOVA.csv", index=False
    )
else:
    spatial_anova = pd.DataFrame()


# ---------------------------------------------------------------------
# PERMUTATION VALIDATION OF PRIMARY SPATIAL MEDIUM EFFECT
# ---------------------------------------------------------------------
# Colony is the unit of permutation.
#
# Medium labels are shuffled WITHIN STRAIN, so every permutation preserves
# the 4-strain structure and the balanced 5-LB / 5-M9 allocation per strain.
# Each colony's complete radial phenotype remains intact.
#
# The observed statistic is the C(medium) F value from the same two-way
# ANOVA used above. 10,000 restricted permutations provide an empirical
# null distribution. A +1 correction is used for the permutation p-value.

N_PERMUTATIONS = 10000
PERMUTATION_SEED = 20261007
rng = np.random.default_rng(PERMUTATION_SEED)

perm_metric = "radial_PC1_slope"
perm_df = radial_metrics[
    ["_bio_id", "_strain", "_medium", perm_metric]
].dropna().rename(
    columns={"_strain": "strain", "_medium": "medium"}
).copy()

perm_formula = f"{perm_metric} ~ C(strain) * C(medium)"

observed_model = ols(perm_formula, data=perm_df).fit()
observed_table = sm.stats.anova_lm(observed_model, typ=2)
observed_medium_F = float(observed_table.loc["C(medium)", "F"])

permuted_F = np.empty(N_PERMUTATIONS, dtype=float)

for p in range(N_PERMUTATIONS):
    shuffled = perm_df.copy()

    # Restricted shuffle: LB/M9 labels only within each strain.
    shuffled["medium"] = (
        shuffled.groupby("strain", group_keys=False)["medium"]
        .transform(lambda x: rng.permutation(x.to_numpy()))
    )

    perm_model = ols(perm_formula, data=shuffled).fit()
    perm_table = sm.stats.anova_lm(perm_model, typ=2)
    permuted_F[p] = float(perm_table.loc["C(medium)", "F"])

n_equal_or_more_extreme = int(np.sum(permuted_F >= observed_medium_F))
permutation_p = (
    n_equal_or_more_extreme + 1
) / (
    N_PERMUTATIONS + 1
)

permutation_summary = pd.DataFrame([{
    "metric": perm_metric,
    "effect_tested": "medium (restricted within-strain permutation)",
    "observed_F": observed_medium_F,
    "n_permutations": N_PERMUTATIONS,
    "n_permuted_F_ge_observed": n_equal_or_more_extreme,
    "permutation_p": permutation_p,
    "seed": PERMUTATION_SEED
}])

permutation_summary.to_csv(
    folders["stats"] / "radial_PC1_slope_medium_permutation_test.csv",
    index=False
)

pd.DataFrame({
    "permutation": np.arange(1, N_PERMUTATIONS + 1),
    "permuted_medium_F": permuted_F
}).to_csv(
    folders["stats"] / "radial_PC1_slope_medium_permutation_null.csv",
    index=False
)

# QC plot of the empirical null distribution.
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(permuted_F, bins=50, edgecolor="black", linewidth=0.5)
ax.axvline(
    observed_medium_F,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"Observed F = {observed_medium_F:.3f}"
)
ax.set_xlabel("Permuted medium-effect F statistic", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.tick_params(axis="both", labelsize=10)
ax.legend(frameon=False, fontsize=9)
savefig(
    fig,
    folders["stats"] / "radial_PC1_slope_medium_permutation_null"
)

print("\nPERMUTATION VALIDATION — PRIMARY SPATIAL MEDIUM EFFECT")
print("  Unit of permutation: biological colony")
print("  Shuffle: LB/M9 labels within each strain")
print(f"  Observed medium F: {observed_medium_F:.6f}")
print(f"  Permutations: {N_PERMUTATIONS}")
print(f"  Permuted F >= observed: {n_equal_or_more_extreme}")
print(f"  Permutation p: {permutation_p:.6g}")


# ---------------------------------------------------------------------
# PAIRWISE STRAIN CONTRASTS WITHIN EACH MEDIUM
# Welch t-test + Cohen's d + BH-FDR.
# These are colony-level contrasts (n~5/group).
# ---------------------------------------------------------------------

pair_rows = []
strain_levels = sorted(radial_metrics["_strain"].unique())

for metric in ["radial_PC1_slope", "outer_minus_inner_PC1"]:
    for medium in sorted(radial_metrics["_medium"].unique()):
        gm = radial_metrics[radial_metrics["_medium"] == medium]
        for i in range(len(strain_levels)):
            for j in range(i+1, len(strain_levels)):
                s1, s2 = strain_levels[i], strain_levels[j]
                a = gm.loc[gm["_strain"] == s1, metric].dropna().to_numpy()
                b = gm.loc[gm["_strain"] == s2, metric].dropna().to_numpy()
                if len(a) >= 2 and len(b) >= 2:
                    test = ttest_ind(a, b, equal_var=False)
                    pair_rows.append({
                        "metric": metric,
                        "medium": medium,
                        "strain_1": s1,
                        "strain_2": s2,
                        "mean_1": np.mean(a),
                        "mean_2": np.mean(b),
                        "difference_1_minus_2": np.mean(a) - np.mean(b),
                        "cohen_d": cohen_d(a, b),
                        "t": test.statistic,
                        "p": test.pvalue
                    })

pairwise = pd.DataFrame(pair_rows)
if len(pairwise):
    pairwise["p_FDR"] = np.nan
    for (metric, medium), idx in pairwise.groupby(
        ["metric", "medium"]
    ).groups.items():
        pairwise.loc[idx, "p_FDR"] = bh_adjust(pairwise.loc[idx, "p"])
pairwise.to_csv(
    folders["stats"] / "strain_pairwise_spatial_contrasts.csv", index=False
)


# ---------------------------------------------------------------------
# MEDIUM EFFECT WITHIN EACH STRAIN
# ---------------------------------------------------------------------

medium_rows = []
medium_levels = sorted(radial_metrics["_medium"].unique())

if len(medium_levels) == 2:
    m1, m2 = medium_levels
    for metric in ["radial_PC1_slope", "outer_minus_inner_PC1"]:
        for strain in strain_levels:
            gs = radial_metrics[radial_metrics["_strain"] == strain]
            a = gs.loc[gs["_medium"] == m1, metric].dropna().to_numpy()
            b = gs.loc[gs["_medium"] == m2, metric].dropna().to_numpy()
            if len(a) >= 2 and len(b) >= 2:
                test = ttest_ind(a, b, equal_var=False)
                medium_rows.append({
                    "metric": metric,
                    "strain": strain,
                    "medium_1": m1,
                    "medium_2": m2,
                    "mean_1": np.mean(a),
                    "mean_2": np.mean(b),
                    "difference_1_minus_2": np.mean(a)-np.mean(b),
                    "cohen_d": cohen_d(a, b),
                    "t": test.statistic,
                    "p": test.pvalue
                })

medium_effects = pd.DataFrame(medium_rows)
if len(medium_effects):
    medium_effects["p_FDR"] = np.nan
    for metric, idx in medium_effects.groupby("metric").groups.items():
        medium_effects.loc[idx, "p_FDR"] = bh_adjust(
            medium_effects.loc[idx, "p"]
        )
medium_effects.to_csv(
    folders["stats"] / "medium_effect_within_strain.csv", index=False
)


# ---------------------------------------------------------------------
# FIGURE 5: COLONY-LEVEL SPATIAL METRIC
# Individual biological replicates shown.
# Figure 2 styling: strain colours, strain markers, LB filled, M9 hollow.
# ---------------------------------------------------------------------

metric = "radial_PC1_slope"
plot_df = radial_metrics.dropna(subset=[metric]).copy()

fig, ax = plt.subplots(figsize=(9, 5.5))

# Arrange strains along x, with LB and M9 slightly offset within each strain.
base_positions = np.arange(len(strain_order), dtype=float)
medium_offsets = {"LB": -0.14, "M9": 0.14}

for i, strain in enumerate(strain_order):
    colour = strain_colours[strain]
    marker = strain_markers[strain]

    for medium in medium_order:
        vals = plot_df.loc[
            (plot_df["_strain"] == strain) &
            (plot_df["_medium"] == medium),
            metric
        ].to_numpy(dtype=float)

        if not len(vals):
            continue

        pos = base_positions[i] + medium_offsets[medium]
        jitter = np.linspace(-0.035, 0.035, len(vals))
        face = colour if medium == "LB" else "none"

        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=60,
            marker=marker,
            facecolors=face,
            edgecolors=colour,
            linewidths=1.4,
            zorder=3,
            label=f"{strain_labels[strain]} – {medium}"
        )

        # Mean line in the same strain colour.
        ax.plot(
            [pos - 0.08, pos + 0.08],
            [np.mean(vals), np.mean(vals)],
            color=colour,
            linewidth=2,
            zorder=4
        )

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(base_positions)
ax.set_xticklabels(
    [strain_labels[s] for s in strain_order],
    fontsize=10
)
ax.set_ylabel("Within-colony radial DINO PC1 slope", fontsize=11)
ax.tick_params(axis="y", labelsize=10)

# Remove duplicate legend entries while preserving Figure 2 ordering.
handles, legend_labels = ax.get_legend_handles_labels()
unique = {}
for h, lab in zip(handles, legend_labels):
    if lab not in unique:
        unique[lab] = h

ordered_labels = [
    f"{strain_labels[s]} – {m}"
    for s in strain_order
    for m in medium_order
    if f"{strain_labels[s]} – {m}" in unique
]
ax.legend(
    [unique[lab] for lab in ordered_labels],
    ordered_labels,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

# No plot title.
savefig(fig, folders["figs"] / "FIG5_radial_spatial_phenotype")


# ---------------------------------------------------------------------
# SENSITIVITY FIGURE
# ---------------------------------------------------------------------

sens_group = (
    sensitivity.groupby(
        ["signal_threshold", "_strain", "_medium"]
    )["radial_PC1_slope"]
    .mean()
    .reset_index()
)

# Match Figure 2: one colour and symbol per strain; LB filled, M9 hollow.
fig, ax = plt.subplots(figsize=(8, 5))
for strain in strain_order:
    for medium in medium_order:
        g = sens_group.loc[
            (sens_group["_strain"] == strain) &
            (sens_group["_medium"] == medium)
        ].sort_values("signal_threshold")
        if g.empty:
            continue

        colour = strain_colours[strain]
        marker = strain_markers[strain]
        ax.plot(
            g["signal_threshold"], g["radial_PC1_slope"],
            color=colour,
            linestyle="-" if medium == "LB" else "--",
            marker=marker,
            markerfacecolor=colour if medium == "LB" else "white",
            markeredgecolor=colour,
            markeredgewidth=1.5,
            markersize=7,
            linewidth=1.6,
            label=f"{strain_labels[strain]} – {medium}"
        )
ax.set_xlabel("Signal-fraction threshold", fontsize=11)
ax.set_ylabel("Mean colony radial PC1 slope", fontsize=11)
ax.tick_params(axis="both", labelsize=10)
# No plot title, consistent with the publication figures.
ax.legend(fontsize=9, frameon=False, bbox_to_anchor=(1.02, 1),
          loc="upper left")
savefig(fig, folders["qc"] / "signal_threshold_sensitivity")


# ---------------------------------------------------------------------
# MACHINE-READABLE GROUP SUMMARY
# ---------------------------------------------------------------------

summary = (
    radial_metrics.groupby(["_strain", "_medium"])
    .agg(
        n_colonies=("_bio_id", "nunique"),
        mean_radial_slope=("radial_PC1_slope", "mean"),
        sd_radial_slope=("radial_PC1_slope", "std"),
        mean_outer_minus_inner=("outer_minus_inner_PC1", "mean"),
        sd_outer_minus_inner=("outer_minus_inner_PC1", "std"),
        mean_tiles_used=("n_tiles_used", "mean")
    )
    .reset_index()
)
summary.to_csv(folders["stats"] / "group_summary.csv", index=False)


# ---------------------------------------------------------------------
# HUMAN-READABLE RESULTS SUMMARY
# ---------------------------------------------------------------------

lines = []
lines.append("DINOv2 E. COLI MACROCOLONY MASTER ANALYSIS")
lines.append("=" * 72)
lines.append("")
lines.append("DATASET")
lines.append(f"Rows (tiles): {len(work)}")
lines.append(f"Biological colonies: {work['_bio_id'].nunique()}")
lines.append(f"DINO feature dimensions: {len(feature_cols)}")
lines.append(f"Strains: {', '.join(sorted(work['_strain'].unique()))}")
lines.append(f"Media: {', '.join(sorted(work['_medium'].unique()))}")
lines.append(
    f"Primary spatial signal threshold: {SIGNAL_THRESHOLD:.2f}"
)
lines.append("")
lines.append("GLOBAL COLONY PHENOTYPE")
lines.append(
    f"PC1 variance explained: "
    f"{100*pca_global.explained_variance_ratio_[0]:.2f}%"
)
lines.append(
    f"PC2 variance explained: "
    f"{100*pca_global.explained_variance_ratio_[1]:.2f}%"
)
lines.append(
    f"PCs required for >=80% colony-level variance: {n80}"
)
lines.append("")

if len(global_anova):
    lines.append("Two-way ANOVA on leading colony-level DINO PCs:")
    for _, r in global_anova.iterrows():
        lines.append(
            f"  {r['PC']} | {r['effect']}: "
            f"F={r['F']:.3f}, p={r['p']:.4g}, "
            f"FDR={r['p_FDR']:.4g}, eta²={r['eta_sq']:.3f}"
        )
    lines.append("")

lines.append("SPATIAL PHENOTYPE")
lines.append(
    "Primary metric: within-colony slope of tile DINO PC1 versus "
    "normalised radial position."
)
lines.append(
    "Secondary metric: outer minus inner mean tile DINO PC1."
)
lines.append(
    "Tiles are never treated as biological replicates; all inferential "
    "comparisons use one spatial summary per colony."
)
lines.append("")

if len(spatial_anova):
    lines.append("Two-way ANOVA on colony-level spatial metrics:")
    for _, r in spatial_anova.iterrows():
        if r["effect"] == "Residual":
            continue
        lines.append(
            f"  {r['metric']} | {r['effect']}: "
            f"F={r['F']:.3f}, p={r['p']:.4g}, eta²={r['eta_sq']:.3f}"
        )
    lines.append("")

lines.append("GROUP SUMMARY")
lines.append(summary.to_string(index=False))
lines.append("")
lines.append("OUTPUT GUIDE")
lines.append(
    "01_GLOBAL_PHENOTYPE/FIG2_colony_level_DINO_PCA.* "
    "-> candidate Figure 2"
)
lines.append(
    "02_SPATIAL_MAPS/all_colonies/ -> QC spatial maps for every colony"
)
lines.append(
    "05_FIGURES/FIG3_representative_spatial_maps.* "
    "-> candidate Figure 3"
)
lines.append(
    "03_RADIAL_PHENOTYPE/FIG4_radial_trajectory_* "
    "-> centre-to-edge phenotype plots"
)
lines.append(
    "05_FIGURES/FIG5_radial_spatial_phenotype.* "
    "-> candidate genotype/medium spatial figure"
)
lines.append("")
lines.append("INTERPRETATION CAUTION")
lines.append(
    "Tile PC1 is an unsupervised learned morphological axis. A significant "
    "radial PC1 effect demonstrates spatial structure in DINO representation, "
    "but does not by itself identify the underlying biological structure."
)
lines.append(
    "Inspect the corresponding source images and representative tiles before "
    "assigning biological meaning to the learned axis."
)

with open(OUTPUT_DIR / "RESULTS_SUMMARY.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


# ---------------------------------------------------------------------
# FINISH
# ---------------------------------------------------------------------

print("=" * 72)
print("MASTER DINOV2 MACROCOLONY ANALYSIS COMPLETE")
print("=" * 72)
print(f"Input:  {INPUT_CSV.resolve()}")
print(f"Output: {OUTPUT_DIR.resolve()}")
print("")
print("Start with:")
print("  1) RESULTS_SUMMARY.txt")
print("  2) 00_QC/QC_REPORT.txt")
print("  3) 01_GLOBAL_PHENOTYPE/FIG2_colony_level_DINO_PCA.pdf")
print("  4) 05_FIGURES/FIG3_representative_spatial_maps.pdf")
print("  5) 03_RADIAL_PHENOTYPE/FIG4_radial_trajectory_*.pdf")
print("  6) 05_FIGURES/FIG5_radial_spatial_phenotype.pdf")
print("")
print("Do not interpret statistical significance before checking the QC plots.")

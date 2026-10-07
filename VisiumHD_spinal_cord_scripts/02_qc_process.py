# %% [markdown]
# # 02. Quality control of cell-level spatial data
#
# This step adds slice labels, calculates mitochondrial and library-size
# metrics, and applies the original numeric filters. Plotting is intentionally
# omitted; the metrics remain in `adata.obs` so readers can inspect or plot them
# interactively when needed.

# %%
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
from roifile import roiread
from shapely.geometry import Point, Polygon

sample_name = "control"
input_path = Path("results/processed") / f"{sample_name}_expend10_raw.h5ad"
slice_roi_path = Path("upload/polygons") / f"{sample_name}_slices.zip"
output_path = Path("results/processed") / f"{sample_name}_expend10_after_qc.h5ad"

# %% [markdown]
# ## 1. Load data and calculate QC metrics

# %%
adata = ad.read_h5ad(input_path)
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=["mt"],
    percent_top=None,
    log1p=False,
    inplace=True,
)
valid = (adata.obs["n_genes_by_counts"] > 0) & (adata.obs["total_counts"] > 0)
adata.obs["log10GenesPerUMI"] = np.nan
adata.obs.loc[valid, "log10GenesPerUMI"] = (
    np.log10(adata.obs.loc[valid, "n_genes_by_counts"])
    / np.log10(adata.obs.loc[valid, "total_counts"])
)
adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt", "log10GenesPerUMI"]].describe()

# One compact QC figure is kept for interactive inspection.
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt", "log10GenesPerUMI"],
    jitter=False,
    multi_panel=True,
)

# %% [markdown]
# ## 2. Assign each cell to a tissue slice

# %%
roi_objects = roiread(str(slice_roi_path))
slice_polygons = gpd.GeoDataFrame(
    {
        "slice": [f"slice_{i + 1}" for i in range(len(roi_objects))],
        "geometry": [Polygon(roi.coordinates()) for roi in roi_objects],
    },
    geometry="geometry",
)

cell_points = gpd.GeoDataFrame(
    {"cell_id": adata.obs_names.astype(str)},
    geometry=[Point(x, y) for x, y in np.asarray(adata.obsm["spatial"])[:, :2]],
).set_index("cell_id", drop=False)
slice_assignment = gpd.sjoin(
    cell_points[["cell_id", "geometry"]],
    slice_polygons[["slice", "geometry"]],
    how="left",
    predicate="within",
).reset_index(drop=True)
adata.obs["slice"] = (
    slice_assignment.drop_duplicates("cell_id").set_index("cell_id")["slice"]
    .reindex(adata.obs_names.astype(str))
    .to_numpy()
)
adata.obs["slice"].value_counts(dropna=False)

# %% [markdown]
# ## 3. Apply the original QC thresholds
#
# The bounds are deliberately visible here so that another reader can inspect
# or change them without opening a helper module.

# %%
qc_keep = (
    adata.obs["slice"].notna()
    & (adata.obs["log10GenesPerUMI"] > 0.9)
    & (adata.obs["log10GenesPerUMI"] < 1.0)
    & (adata.obs["pct_counts_mt"] < 15)
    & (adata.obs["total_counts"] < 10000)
    & (adata.obs["n_genes_by_counts"] < 5000)
    & (adata.obs["n_genes_by_counts"] > 10)
)
print(f"retaining {qc_keep.sum():,}/{adata.n_obs:,} cells")
adata = adata[qc_keep].copy()
adata

# %%
output_path.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(output_path)
print(f"saved {output_path}")

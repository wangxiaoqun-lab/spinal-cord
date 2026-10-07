# %% [markdown]
# # 01. Prepare the Expend10 cell-level spatial data
#
# The original cell-segmentation experiments are intentionally not included.
# This workflow starts from the already prepared **Expend10** cell/nucleus
# polygons and aggregates Visium HD spot counts into one row per segmented cell.
#
# Run this file from the repository root and change `sample_name` for the second
# sample. The polygon file should be the Expend10 ROI set exported from ImageJ,
# or an equivalent GeoJSON file.

# %% [markdown]
# ## 1. Choose the sample and paths

# %%
from pathlib import Path

sample_name = "control"  # change to "activated" for the second sample
upload_dir = Path("upload")
results_dir = Path("results") / "processed"
results_dir.mkdir(parents=True, exist_ok=True)

visium_dir = upload_dir / "visium" / sample_name / "square_002um"
expend10_polygon_path = upload_dir / "polygons" / f"{sample_name}_expend10_cells.zip"
output_path = results_dir / f"{sample_name}_expend10_raw.h5ad"

# %% [markdown]
# ## 2. Load the Visium HD matrix and Expend10 polygons

# %%
import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scanpy as sc
from roifile import roiread
from scipy import sparse
from shapely.geometry import Point, Polygon

adata = sc.read_visium(visium_dir)
adata.var_names_make_unique()
print(f"spots={adata.n_obs:,}, genes={adata.n_vars:,}")

if expend10_polygon_path.suffix.lower() == ".zip":
    roi_objects = roiread(str(expend10_polygon_path))
    polygon_table = gpd.GeoDataFrame(
        {
            "polygon_id": [f"cell_{i + 1:06d}" for i in range(len(roi_objects))],
            "geometry": [Polygon(roi.coordinates()) for roi in roi_objects],
        },
        geometry="geometry",
    )
else:
    polygon_table = gpd.read_file(expend10_polygon_path)
    if "polygon_id" not in polygon_table.columns:
        polygon_table["polygon_id"] = [f"cell_{i + 1:06d}" for i in range(len(polygon_table))]

polygon_table["polygon_id"] = polygon_table["polygon_id"].astype(str)
polygon_table = polygon_table.loc[~polygon_table.geometry.is_empty].copy()
polygon_table.head()

# %% [markdown]
# ## 3. Assign each Visium spot to an Expend10 polygon
#
# A barcode that falls into more than one polygon is discarded. This is the
# overlap rule used in the original Expend10 aggregation step.

# %%
coordinates = np.asarray(adata.obsm["spatial"], dtype=float)
spot_points = gpd.GeoDataFrame(
    {"barcode": adata.obs_names.astype(str)},
    geometry=[Point(x, y) for x, y in coordinates[:, :2]],
).set_index("barcode", drop=False)

joined = gpd.sjoin(
    spot_points[["barcode", "geometry"]],
    polygon_table[["polygon_id", "geometry"]],
    how="left",
    predicate="within",
).reset_index(drop=True)
joined["is_within_polygon"] = joined["polygon_id"].notna()
overlapping_barcodes = set(
    joined.loc[joined.duplicated("barcode", keep=False), "barcode"].astype(str)
)
joined["is_not_overlapping"] = ~joined["barcode"].isin(overlapping_barcodes)

assigned = joined.loc[
    joined["is_within_polygon"] & joined["is_not_overlapping"],
    ["barcode", "polygon_id"],
].drop_duplicates("barcode")
print(f"unique assignments={len(assigned):,}; overlapping barcodes={len(overlapping_barcodes):,}")

# %% [markdown]
# ## 4. Sum counts within each Expend10 cell

# %%
assigned_barcodes = assigned["barcode"].tolist()
spot_counts = sparse.csr_matrix(adata[assigned_barcodes, :].X)
row_by_barcode = {barcode: i for i, barcode in enumerate(assigned_barcodes)}

summed_rows = []
cell_ids = []
spot_counts_per_cell = []
for polygon_id, barcode_series in assigned.groupby("polygon_id", sort=False)["barcode"]:
    row_indices = [row_by_barcode[barcode] for barcode in barcode_series]
    summed_rows.append(spot_counts[row_indices].sum(axis=0))
    cell_ids.append(polygon_id)
    spot_counts_per_cell.append(len(row_indices))

cell_counts = sparse.vstack([sparse.csr_matrix(row) for row in summed_rows]).tocsr()
cell_adata = ad.AnnData(X=cell_counts, var=adata.var.copy())
cell_adata.obs = pd.DataFrame(
    {"polygon_id": cell_ids, "n_spot_barcodes": spot_counts_per_cell},
    index=cell_ids,
)

polygon_metadata = polygon_table.set_index("polygon_id").loc[cell_ids]
cell_adata.obs["x"] = polygon_metadata.geometry.centroid.x.to_numpy()
cell_adata.obs["y"] = polygon_metadata.geometry.centroid.y.to_numpy()
cell_adata.obs["area"] = polygon_metadata.geometry.area.to_numpy()
cell_adata.obsm["spatial"] = cell_adata.obs[["x", "y"]].to_numpy()
if "spatial" in adata.uns:
    cell_adata.uns["spatial"] = adata.uns["spatial"].copy()

cell_adata.var_names_make_unique()
cell_adata

# A single spatial preview is retained as a sanity check for the aggregation.
sc.pl.embedding(cell_adata, basis="spatial", color="n_spot_barcodes", size=20)

# %% [markdown]
# ## 5. Save the Expend10 cell-level object

# %%
cell_adata.write_h5ad(output_path)
print(f"saved {output_path}")

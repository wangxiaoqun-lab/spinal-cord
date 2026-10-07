# %% [markdown]
# # 03. Integrate, cluster and annotate all cells
#
# This step creates the main integrated object, performs the broad cell
# clustering, assigns the first-pass cell-type labels, and calculates marker
# tables. Neuron re-clustering is kept separately in step 04 because it is a
# distinct analysis described separately in the methods.

# %%
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

control_path = Path("results/processed/control_expend10_after_qc.h5ad")
activated_path = Path("results/processed/activated_expend10_after_qc.h5ad")
results_dir = Path("results")
markers_dir = results_dir / "markers"
markers_dir.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Integrate the two Expend10 samples

# %%
control = ad.read_h5ad(control_path)
activated = ad.read_h5ad(activated_path)
adata = ad.concat(
    {"Control": control, "Activated": activated},
    label="batch",
    index_unique="-",
    join="outer",
)
adata.layers["counts"] = adata.X.copy()
print(adata.obs["batch"].value_counts())
print(f"cells={adata.n_obs:,}, genes={adata.n_vars:,}")

# %% [markdown]
# ## 2. Normalize and cluster all cells

# %%
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
adata.layers["normalized"] = adata.X.copy()

sc.pp.scale(adata)
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_pcs=50, n_neighbors=100)
sc.tl.umap(adata, min_dist=0.5)
sc.tl.leiden(adata, resolution=2, key_added="leiden")
adata.obs["leiden"].value_counts().sort_index()

# Representative whole-object view.
sc.pl.umap(adata, color="leiden", legend_loc="on data")

# %% [markdown]
# ## 3. Assign broad cell-type labels

# %%
cluster_to_celltype = {
    "0": "Astrocyte_1", "1": "Neuron_1", "2": "Oligodendrocyte",
    "3": "Neuron_2", "4": "Astrocyte_3", "5": "Astrocyte_2",
    "6": "Neuron_3", "7": "Microglia", "8": "Astrocyte_1",
    "9": "Oligodendrocyte", "10": "Neuron_4", "11": "Neuron_5",
    "12": "Endothelial Cell", "13": "Oligodendrocyte", "14": "Astrocyte_1",
    "15": "OPC", "16": "Neuron_6", "17": "Neuron_7",
    "18": "Pericyte", "19": "VSMC", "20": "Ependymal cell",
    "21": "Plasma cell", "22": "Blood cell",
}
cluster_ids = adata.obs["leiden"].astype(str)
adata.obs["celltype"] = cluster_ids.map(cluster_to_celltype)

adata.obs["celltype"] = pd.Categorical(adata.obs["celltype"])
adata.obs["celltype"].value_counts()
sc.pl.umap(adata, color="celltype")

# A coarser class label is retained for downstream subgroup analyses.
cluster_to_class = {
    "0": "Astrocyte", "1": "Neuron", "2": "Oligodendrocyte", "3": "Neuron",
    "4": "Astrocyte", "5": "Astrocyte", "6": "Neuron", "7": "Microglia",
    "8": "Astrocyte", "9": "Oligodendrocyte", "10": "Neuron", "11": "Neuron",
    "12": "Endothelial Cell", "13": "Oligodendrocyte", "14": "Astrocyte",
    "15": "OPC", "16": "Neuron", "17": "Neuron", "18": "Pericyte",
    "19": "VSMC", "20": "Ependymal cell", "21": "Plasma cell", "22": "Blood cell",
}
adata.obs["cellclass"] = cluster_ids.map(cluster_to_class)

# %% [markdown]
# ## 4. Calculate marker tables

# %%
sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", layer="normalized")
sc.get.rank_genes_groups_df(adata, group=None).to_csv(
    markers_dir / "markers_by_leiden.csv", index=False
)
sc.tl.rank_genes_groups(adata, groupby="celltype", method="wilcoxon", layer="normalized")
celltype_markers = sc.get.rank_genes_groups_df(adata, group=None)
celltype_markers.to_csv(markers_dir / "markers_by_celltype.csv", index=False)

sc.pl.dotplot(adata, var_names=["Snap25", "Syt1", "Aqp4", "Gfap", "Mbp", "Cldn5", "Csf1r"], groupby="celltype", standard_scale="var")

# %% [markdown]
# ## 5. Save the integrated object

# %%
adata.write_h5ad(results_dir / "all.h5ad")
print(f"saved all-cell object: {adata.n_obs:,} cells")

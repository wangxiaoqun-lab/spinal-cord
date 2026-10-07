# %% [markdown]
# # 04. Split the neuronal groups to the used annotation level
#
# This step starts from the integrated object produced in step 03, extracts the
# broad neuronal population, and refines clusters `10` and `3` to the annotation
# level used downstream.
# Later exploratory neuron-internal subtype mapping is not included.

# %%
from pathlib import Path

import scanpy as sc

input_path = Path("results/all.h5ad")
results_dir = Path("results")
output_path = results_dir / "neuron_level2.h5ad"

all_adata = sc.read_h5ad(input_path)

neuron_mask = all_adata.obs["cellclass"].astype(str) == "Neuron"
adata = all_adata[neuron_mask].copy()
print(f"extracted neuron subset: {adata.n_obs:,} cells")

# %% [markdown]
# ## 1. annotation of neuronal clusters

# %%

neuron_annotation = {
    "1": "Neuron_Snap25_Lamp5",
    "3": "Neuron_mix", 
    "6": "Neuron_Snap25_Plekhb1", 
    "10": "Neuron_mix", 
    "11": "Neuron_Snap25_Plxnb3",
    "16": "Neuron_Chat_Cartpt",
    "17": "Neuron_Chat_Chodl",
}

cluster_ids = adata.obs["leiden"].astype(str)
adata.obs["celltype_reannotation"] = cluster_ids.map(neuron_annotation)

# %% [markdown]
# ## 2. Split the two mix neuronal clusters

# %%
sc.tl.leiden(
    adata,
    restrict_to=("celltype_reannotation", ["Neuron_mix"]),
    key_added="celltype_temp",
    resolution=0.8,
)

split_to_celltype = {
    "Neuron_mix,0": "Neuron_Pdyn_Gal",
    "Neuron_mix,1": "Neuron_Maf_Cck",
    "Neuron_mix,2": "Neuron_Ntsr2_Slc39a12",
    "Neuron_mix,3": "Neuron_Prkcg_Nts",
    "Neuron_mix,4": "Neuron_Ntsr2_Slc39a12",
}

subcluster_ids = adata.obs["celltype_temp"].astype(str)
split_labels = subcluster_ids.map(split_to_celltype)
split_mask = split_labels.notna()
adata.obs.loc[split_mask, "celltype_reannotation"] = split_labels[split_mask]

adata.obs["celltype_reannotation"].value_counts()
sc.pl.umap(adata, color="celltype_reannotation")

# %% [markdown]
# ## 3. Save the refined neuronal object

# %%
adata.write_h5ad(output_path)
print(f"saved {output_path}")

# %% [markdown]
# # 05. Calculate IEG AUCell scores
#
# This step runs after the neuronal group split and uses `neuron_level2.h5ad`.

# %%
library(sceasy)
library(Seurat)
library(AUCell)

input_h5ad <- "results/neuron_level2.h5ad"
output_dir <- "results/aucell"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# %% [markdown]
# ## 1. Convert the AnnData object to Seurat

# %%
neuron_rds <- file.path(output_dir, "neuron_input.rds")
sceasy::convertFormat(
  obj = input_h5ad,
  from = "anndata",
  to = "seurat",
  outFile = neuron_rds
)
neuron <- readRDS(neuron_rds)
dim(neuron)

# %% [markdown]
# ## 2. Define the IEG set and calculate AUCell

# %%
IEG_GENES <- c(
  "Jun", "Jund", "Nr4a1", "Myc", "Homer1", "Bdnf", "Arc", "Fos",
  "Fosl2", "Fosb", "Egr1", "Egr2", "Egr4", "Junb", "Per1"
)
gene_sets <- list(IEG_AUCell = IEG_GENES)
IEG_GENES[IEG_GENES %in% rownames(neuron)]

expression_matrix <- GetAssayData(neuron, slot = "data")
cell_rankings <- AUCell_buildRankings(
  expression_matrix,
  nCores = 4,
  plotStats = FALSE
)
cells_AUC <- AUCell_calcAUC(gene_sets, cell_rankings)
auc_matrix <- getAUC(cells_AUC)
dim(auc_matrix)

# %% [markdown]
# ## 3. Add scores and define the high-IEG group

# %%
auc_metadata <- as.data.frame(t(auc_matrix))
neuron <- AddMetaData(neuron, metadata = auc_metadata)
auc_threshold <- as.numeric(quantile(neuron$IEG_AUCell, probs = 0.90, na.rm = TRUE))
neuron$IEG_high <- neuron$IEG_AUCell > auc_threshold
print(auc_threshold)
table(neuron$IEG_high, useNA = "ifany")
hist(neuron$IEG_AUCell, breaks = 40, main = "IEG AUCell score", xlab = "AUCell score")

# %%
write.csv(neuron@meta.data, file.path(output_dir, "meta_neuron.csv"), row.names = TRUE)
writeLines(as.character(auc_threshold), file.path(output_dir, "IEG_AUCell_90th_percentile.txt"))
saveRDS(neuron, file.path(output_dir, "neuron_aucell.rds"))

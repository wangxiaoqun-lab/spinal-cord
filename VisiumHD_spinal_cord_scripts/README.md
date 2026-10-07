# Spinal cord Visium HD data analysis

This folder contains five interactive scripts for processing spinal cord Visium HD data. The scripts are organized by analysis stage and use `# %%` cell markers so that the processing logic can be read and inspected interactively in Jupyter, VS Code, or Spyder.

## Scripts

| File | Main purpose |
|---|---|
| `01_prepare_expend10_spatial_data.py` | Aggregate Visium HD spots within the provided cell or nucleus polygons to generate cell-level expression data. |
| `02_qc_process.py` | Calculate quality-control metrics, assign cells to tissue slices, and filter low-quality cells. |
| `03_integrate_and_annotate.py` | Integrate the control and activated samples, cluster all cells, assign broad cell labels, and calculate marker tables. |
| `04_neuron_recluster.py` | Extract the neuronal population from the integrated object and refine the neuronal annotations. |
| `05_aucell_process.R` | Calculate IEG AUCell scores for the refined neuronal population. |

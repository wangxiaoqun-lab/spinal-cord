# Axon quantification and brain-wide visualization

This folder contains six Python scripts for quantifying reconstructed spinal axons and visualizing brain-wide anatomical data. The scripts are numbered by function. The private imaging data, gray-matter label volume, regional Z-score table, atlas files, transformed cell-coordinate files, and binary spinal-cord masks are not included.

## Scripts

| File | Main purpose |
|---|---|
| `01_run_statistics.py` | Configure the input paths, run the spinal-cord and gray-matter analyses, and export the two final Excel workbooks. |
| `02_axon_metrics.py` | Read the four SWC reconstructions, infer axonal structures, and calculate whole-spinal-cord and four-segment axon metrics. |
| `03_gray_metrics.py` | Assign axon segments, terminals, and branch points to the dorsal, middle, and ventral gray-matter regions and their intersections with the four spinal segments. |
| `04_brain_zscore_heatmap.py` | Read an external regional Z-score table and atlas annotation volume and generate a multipage anatomical heatmap PDF. |
| `05_whole_brain_3d_cells.py` | Reconstruct an atlas-derived three-dimensional brain surface and render registered cell coordinates from multiple viewpoints. |
| `06_spinal_shell_swc_html.py` | Read private outer/inner binary TIFF stacks and four SWCs, infer axons and other neurites, and generate a standalone interactive HTML without gray-matter subregion data. |

## Interactive spinal-cord HTML

`06_spinal_shell_swc_html.py` requires two same-sized binary TIFF stacks and exactly four SWC files. It keeps the latest project viewer controls for orientation, camera angles, shell and neurite styling, lighting, coordinate and oblique clipping, transparency sorting, scale bars, and PNG/PDF/SVG export. It does not read or embed dorsal/middle/ventral gray-matter labels.

Example:

```powershell
python 06_spinal_shell_swc_html.py `
  --outer-mask "D:\private_data\outer_mask.tif" `
  --inner-mask "D:\private_data\inner_mask.tif" `
  --swc "D:\private_data\strong_central_3.swc" "D:\private_data\stong-4_central_done.swc" "D:\private_data\tree-10_dorsal_done.swc" "D:\private_data\tree-11_dorsal_done.swc" `
  --output "D:\results\spinal_shell_four_neurons.html"
```

The default reconstruction settings reproduce the project geometry: ZYX crop `(0:6418, 20:1310, 0:939)`, 4×4×4 mesh downsampling, ZYX display scaling `(7.1, 6.0, 7.1)`, 15 smoothing iterations, orthographic projection, XZY display order, black background, and no initial clipping plane. Use `--help` to view the optional overrides.

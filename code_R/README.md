# VIVID analysis (R `vivid` package)

`run_vivid.R` runs the `vivi()` function of the `vivid` R package on the JULES-crop emulator
(the network forward pass is reconstructed in pure R from the exported weights, so the script
depends only on `vivid` and `ggplot2`). It computes the importance and interaction matrices at
the level of the original predictors and generates the heatmaps and networks with common color
scales. `vivi_R_{sdm,ldm,gdm}.csv` are the resulting matrices used for the figures in the paper.

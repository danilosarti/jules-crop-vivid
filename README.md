# Integration of Interpretable Machine Learning and JULES-crop

Companion repository for the paper by Prudente Junior, Fray da Silva, Marin and Sarti.
A multi-output neural-network emulator of JULES-crop combined with interpretability
through the VIVID framework.

## Structure

- `manuscript/` - manuscript (.docx) with the figures embedded at 300 dpi
- `code/` - reproducible Python pipeline (see `code/PIPELINE.md`)
- `code_R/` - VIVID analysis with the R `vivid` package (`run_vivid.R`) and the output matrices
- `data/` - input data (`simulated.rds`, `met_data_calibracao.csv`)
- `figures/` - figures (PNG at 300 dpi + vector PDF)

## Reproduce (Python)

    cd code
    pip install -r requirements.txt
    python 01_prep_data.py && python 02_train.py && python 03_vivid.py && python 04_figures.py

Test-set performance: R2 = 0.96 (SDM), 0.93 (GDM), 0.89 (LDM); RMSE = 0.45 / 0.62 / 0.41 t/ha.
Importance ranking (all outputs): thermal time > cultivar > site > solar radiation.

## VIVID figures

The importance and interaction figures in the paper were produced with the `vivid` R package
(see `code_R/`), with the Vimp (importance) and Vint (interaction) color scales made common
across the three outputs so that shading is directly comparable.

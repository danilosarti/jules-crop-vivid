# JULES-crop + VIVID - Python pipeline

Emulates JULES-crop with a multi-output neural network and applies VIVID
(variable importance + pairwise interactions).

## Input data (../data)

- `simulated.rds` - daily JULES-crop biometric output
- `met_data_calibracao.csv` - sub-daily meteorological forcing

## How to run

    pip install -r requirements.txt
    python 01_prep_data.py     # -> csv/model_data_v2.csv
    python 02_train.py         # -> nn_v2.keras, eval_v2.npz, meta_v2.json
    python 03_vivid.py         # -> csv/vivi_v2_{sdm,ldm,gdm}.csv
    python 04_figures.py       # -> figures_300dpi/*.png and *.pdf

With the fixed seeds (123 for training, 7 for VIVID) the result is deterministic.

## Expected result (test set)

| Output | R2 | RMSE (t/ha) |
|--------|-----|-------------|
| SDM    | 0.96 | 0.45 |
| GDM    | 0.93 | 0.62 |
| LDM    | 0.89 | 0.41 |

Importance ranking (all outputs): thermal time > cultivar > site > solar radiation.

## Predictors

Continuous: accumulated thermal time, Tmean, Tmax, solar radiation, diffuse radiation,
specific humidity. Categorical: cultivar, site, water regime.

The importance and interaction figures reported in the paper are produced with the `vivid`
R package (see `../code_R`).

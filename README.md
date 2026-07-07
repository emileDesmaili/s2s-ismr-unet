# s2s-ismr-unet

Code for Unet postprocessing of ISMR at S2S range (Esmaili, Robertson et al. 2025).

## Setup

Create the `s2s-cnn` conda environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate s2s-cnn
```

## Pipeline

The pipeline has two stages: (1) hindcast tuning (produces `outputs/*.nc`), then
(2) notebooks that read those outputs and render the paper figures.

### 1. Hindcast tuning

Run each `tune_XXX.py` script for the respective model configuration; `com`
targets the common period and `full` targets the full period. Select lead
times inside the script.

- `tune_GEFS_full.py`, `tune_GEFS_com.py`
- `tune_IITM_full.py`, `tune_IITM_com.py`
- `tune_ECMWF_full.py`, `tune_ECMWF_com.py`
- `tune_MME.py`, `tune_2MME.py`

Outputs land under `outputs/{Full Period,Common Period,MME,2MME}/{model}_IMD/`.

### 2. Figure notebooks and scripts

Each entry lists the paper figures it produces. Paths below are the notebook's
save location (the paper `.tex` may reference a differently-named folder — the
files are identical).

| Notebook / script | Paper figures produced |
| --- | --- |
| `ACCs.ipynb` | `figures/{GEFS,IITM,ECMWF}_IMD/climo.pdf`, `figures/IITM_IMD/climo_obs.pdf`, `figures/{GEFS,IITM,ECMWF}_IMD/ACC_wk{1,2,3,4,3-4}.pdf` |
| `Bar_plot.ipynb` | `figures/barplots/{GEFS,IITM,ECMWF}_RAW_ELR_UNET_boxplot.pdf`, `MME_{ELR,unet}_plots.pdf`, `Unet_vs_ELR_fractionMME.pdf`, `facet_region_bar_week_{1,2,3-4}.pdf`, `MME_region_facet_week_{1,2,3-4}.pdf` |
| `RPSS_plots.ipynb` | `figures/rpss/{UNet,ELR}_{Full,Common}.pdf`, `IITM_Full_Train_Val_Test.pdf` |
| `RPSS_drivers.ipynb` | `figures/RPSS drivers/RPSS_MJO_plain.pdf`, `RPSS_ENSO4_plain.pdf` |
| `plot_rpss_drivers_combined.py` | `figures/RPSS_drivers_combined_by_model.pdf` |
| `DeepNet_prototype_GEFS_IMD.ipynb` | `figures/{Full Period,Common Period,MME}/{model}_IMD/wk3_4_{Below_Normal,Normal,Above_Normal}.pdf` (via `utils.plots.reliability_diagram_compare`) — re-run with each `(model, obs, dir)` combination in the paper |
| `Realtime_fcast_MME.ipynb` (2023 season) | `figures/rt_forecast_MME/MME_IMD/{accuracy,2cat_error_rate,rpss}_comparison.pdf`, per-date forecast pdfs, `figures/drivers/{sst_anomalies_2023,IMD_anomalies,nino34_correlation,mjo_2023}.pdf` |
| `Realtime_fcast_MME_24.ipynb` (2024 season) | `figures/rt_forecast_MME_24/MME_IMD/{accuracy,2cat_error_rate,rpss}_comparison.pdf` |



## Acknowledgements

Code for the U-Net architecture was adapted from Horat and Lerch (2023):
<https://github.com/HoratN/pp-s2s>.

## Correspondence

Please contact *ede2110@columbia.edu* for any questions or inquiries regarding
the code and its usage.

# s2s-ismr-unet
Code for Unet postprocessing of ISMR at S2S range (Esmaili, Robertson et al. 2025)

- First create the s2s-cnn conda environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

- Next, activate the environment:

```bash
conda activate s2s-cnn
```

# 1. Notebooks

a. ```ACCs.ipynb``` for figure 1 and 2 (climatologies, and ACC scores)

b. ```Bar_plot.ipynb``` for tuned hindcasts skill plots

c. ```Realtime_forecast_MME.ipynb```  for real-time forecast evaluation, 2023 season MJO/ENSO diagnostic and GradCAM maps

# 2. Hindcast tuning


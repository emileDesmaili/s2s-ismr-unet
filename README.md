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

c. ```RPSS_plots.ipynb``` for spatial RPSS plots at weeks3--4

d. ```Realtime_forecast_MME.ipynb```  for real-time forecast evaluation, 2023 season MJO/ENSO diagnostic.

e. ```DeepNet_prototype_GEFS_IMD.ipynb``` main testing notebook: loads GEFS and IMD data (but feel free to change the lead time, domain, and model/obs combinations), trains and evaluates the U-Net model, and produces hindcast RPSS scores.

f. ```RPSS_drivers.ipynb``` RPSS stratification by MJO phase and ENSO mode (Niño4), including spatial maps, composite significance stippling, bar charts, and combined MJO×ENSO heatmaps.



# 2. Hindcast tuning

To tune the hindcast, please run each ```tune_XXX.py``` script for the respective model configuration. *com* refers to the common period and *full* refers to the full period. Select the lead times accordingly.

# Acknowledgements 

codes for the U-Net architecture were directly adapted from Horat and Lerch (2023), accessible at the following repository:  https://github.com/HoratN/pp-s2s

# Correspondence

Please contact *ede2110@columbia.edu* for any questions or inquiries regarding the code and its usage.

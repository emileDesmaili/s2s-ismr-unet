"""Combined Raw / ELR / U-Net RPSS by MJO phase and ENSO mode.

Each model is evaluated on its own valid dates:
- Raw: GEFS wk3-4 anomalies on full 1989-2018 period, RPSS = 1 - sqrt(1 - ACC^2)
- ELR: probabilistic RPSS on ELR's CV test folds (Full Period file)
- U-Net: probabilistic RPSS on U-Net's CV test folds (Full Period file)

Outputs: figures/RPSS_drivers_combined_by_model.pdf (and .png)
"""
import sys
sys.path.insert(0, 'utils')

import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import xskillscore
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgb

import dataloader

# ----- Nature-style rcParams -----
mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         8,
    'axes.labelsize':    9,
    'axes.titlesize':    9.5,
    'axes.titleweight':  'bold',
    'axes.linewidth':    0.6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'legend.fontsize':   8,
    'legend.frameon':    False,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

GEFS_DIR = 'outputs/Full Period/GEFS_IMD'
FIG_DIR  = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

# =====================================================================
# Drivers
# =====================================================================
def load_mjo():
    mjo = pd.read_csv(
        "download/mjo.txt", skiprows=2, sep=r'\s+', header=None,
        names=["year","month","day","RMM1","RMM2","phase","amplitude","method","extra"],
        na_values=[1e35, 999]
    ).drop(columns="extra", errors="ignore")
    mjo["date"] = pd.to_datetime(dict(year=mjo.year, month=mjo.month, day=mjo.day), errors="coerce")
    mjo = mjo[(mjo['date'] >= '1980-01-01') & (mjo['date'] <= '2025-12-31')].set_index('date')
    return mjo[['RMM1','RMM2']].rolling(window=15, center=True, min_periods=1).mean().to_xarray()

def load_nino4():
    records = []
    pattern = re.compile(r"[+-]?\d+\.\d+")
    with open("download/sst_weekly.txt") as f:
        for line in f:
            if re.match(r"\s*\d{2}[A-Z]{3}\d{4}", line):
                date = line.split()[0]
                nums = list(map(float, pattern.findall(line)))
                if len(nums) == 8:
                    records.append([date] + nums)
    cols = ["Week","Nino12_SST","Nino12_SSTA","Nino3_SST","Nino3_SSTA",
            "Nino34_SST","Nino34_SSTA","Nino4_SST","Nino4_SSTA"]
    nino = pd.DataFrame(records, columns=cols)
    nino["Week"] = pd.to_datetime(nino["Week"], format="%d%b%Y")
    nino = nino.set_index("Week").rolling(window=2, center=True, min_periods=1).mean()
    return nino.to_xarray()

MJO_XR   = load_mjo()
NINO_XR  = load_nino4()

def mjo_phase(dates):
    dates = pd.to_datetime(dates)
    rmm1 = MJO_XR['RMM1'].sel(date=dates, method='nearest').values
    rmm2 = MJO_XR['RMM2'].sel(date=dates, method='nearest').values
    return (((np.arctan2(rmm2, rmm1) + 2*np.pi) % (2*np.pi)) // (np.pi/4) + 1).astype(int)

def enso_mode(dates):
    v = NINO_XR["Nino4_SSTA"].sel(Week=pd.to_datetime(dates), method='nearest').values
    return np.where(v >= 0.5, 1, np.where(v <= -0.5, -1, 0))

# =====================================================================
# Raw GEFS: ACC-derived RPSS
# =====================================================================
def compute_raw():
    print("=== Raw GEFS (1989-2018) ===")
    x, y = dataloader.get_data(
        years=(1989, 2018), download=False, week="wk3-4", obs="IMD",
        domain=[67, 98, 7, 38], season="May-Sep", model="GEFS", regrid=1
    )
    gefs_climo = xr.open_dataset('download/gefs_climo.nc')
    gefs_climo['S'] = pd.to_datetime(gefs_climo['S'])
    gefs_climo['L'] = gefs_climo['L'] / np.timedelta64(1, 'D')

    x = x.mean('M')
    x_anom_list, y_anom_list = [], []
    for week_i in set(x['T'].dt.isocalendar().week.values):
        x_w = x.where(x['T'].dt.isocalendar().week == week_i, drop=True)
        y_w = y.where(y['T'].dt.isocalendar().week == week_i, drop=True)
        first_date = pd.Timestamp(x_w['S'].values[0])
        x_climo = gefs_climo.sel(
            S=(gefs_climo['S'].dt.month == first_date.month) & (gefs_climo['S'].dt.day == first_date.day)
        ).sel(L=slice(16.5, 28.5)).mean('L').mean('S').rename({'pr':'prcp'})
        x_anom_list.append(x_w - x_climo)
        y_anom_list.append(y_w - y_w.mean('T'))

    x_anom = xr.merge(x_anom_list)['prcp'].sortby('T')
    y_anom = xr.merge(y_anom_list)['prcp'].sortby('T')

    obs_unet  = xr.open_dataarray(f'{GEFS_DIR}/unet_ytest_raw_wk3-4.nc')
    preds_elr = xr.open_dataarray(f'{GEFS_DIR}/ELR_predictions_wk3-4.nc')
    real_X, real_Y = obs_unet.X, obs_unet.Y
    lm_obs = y_anom.isnull().all(dim='T').assign_coords(X=real_X, Y=real_Y)
    lm_elr = preds_elr.isel(bootstrap=0, category=0).isnull().all(dim='T').assign_coords(X=real_X, Y=real_Y)
    land = (lm_obs | lm_elr)

    def acc_to_rpss(mask):
        acc = xr.corr(x_anom.isel(T=mask), y_anom.isel(T=mask), dim='T')
        return float((1 - np.sqrt(1 - acc**2)).where(~land).mean())

    dates = pd.to_datetime(x_anom['T'].values)
    out = {'overall': acc_to_rpss(np.ones(len(dates), dtype=bool)),
           'enso': {}, 'mjo': {}}
    enso = enso_mode(dates)
    phase = mjo_phase(dates)
    for m in [-1, 0, 1]:
        msk = (enso == m)
        out['enso'][m] = (acc_to_rpss(msk), int(msk.sum()))
    for p in range(1, 9):
        msk = (phase == p)
        out['mjo'][p] = (acc_to_rpss(msk), int(msk.sum()))
    return out

# =====================================================================
# ELR / U-Net: probabilistic RPSS
# =====================================================================
def _build_oheM(obs_file, preds_file, is_elr):
    obs_raw = xr.open_dataarray(obs_file)
    preds   = xr.open_dataarray(preds_file)
    obs_unet  = xr.open_dataarray(f'{GEFS_DIR}/unet_ytest_raw_wk3-4.nc')
    real_X, real_Y = obs_unet.X, obs_unet.Y

    if is_elr:
        obs_list, preds_list = [], []
        for b in range(len(obs_raw.bootstrap)):
            ob = obs_raw.isel(bootstrap=b)
            vm = ~ob.isnull().all(dim=("Y","X","category"))
            vt = ob['T'][vm].values
            obs_list.append(ob.isel(T=vm))
            preds_list.append(preds.isel(bootstrap=b).assign_coords(T=vt))
    else:
        valid_T_per_b = {}
        for b in obs_unet.bootstrap.values:
            da = obs_unet.sel(bootstrap=b)
            valid_T_per_b[b] = da['T'][~da.isnull().all(dim=("Y","X"))].values
        obs_list, preds_list = [], []
        for b in range(len(obs_raw.bootstrap)):
            obs_list.append(obs_raw.isel(bootstrap=b).assign_coords(T=valid_T_per_b[b]))
            preds_list.append(preds.isel(bootstrap=b).assign_coords(T=valid_T_per_b[b]))

    oheM = xr.concat(obs_list, dim="bootstrap").mean(dim="bootstrap")
    prdM = xr.concat(preds_list, dim="bootstrap").mean(dim="bootstrap")
    vt   = ~oheM.isnull().all(dim=("Y","X","category"))
    oheM = oheM.sel(T=vt).assign_coords(X=real_X, Y=real_Y)
    prdM = prdM.sel(T=oheM['T']).assign_coords(X=real_X, Y=real_Y)
    lm_elr = xr.open_dataarray(f'{GEFS_DIR}/ELR_predictions_wk3-4.nc').isel(
        bootstrap=0, category=0).isnull().all(dim='T').assign_coords(X=real_X, Y=real_Y)
    return oheM, prdM, lm_elr

def compute_prob(obs_file, preds_file, label, is_elr):
    print(f"=== {label} ===")
    oheM, prdM, land = _build_oheM(obs_file, preds_file, is_elr=is_elr)
    obs_r = oheM.transpose('T','Y','X','category')
    fcast = prdM.transpose('T','Y','X','category')
    climo = xr.full_like(fcast, 1/3)

    def rpss(mask):
        o = obs_r.isel(T=mask); f = fcast.isel(T=mask); c = climo.isel(T=mask)
        rm = xskillscore.rps(o, f, dim='T', category_edges=None, input_distributions='p')
        rc = xskillscore.rps(o, c, dim='T', category_edges=None, input_distributions='p')
        return float((1 - rm/rc).where(~land).mean())

    dates = pd.to_datetime(oheM['T'].values)
    enso  = enso_mode(dates)
    phase = mjo_phase(dates)
    return {'overall': rpss(np.ones(len(dates), dtype=bool)),
            'enso': {m: (rpss(enso == m), int((enso == m).sum())) for m in [-1, 0, 1]},
            'mjo':  {p: (rpss(phase == p), int((phase == p).sum())) for p in range(1, 9)}}

# =====================================================================
# Run
# =====================================================================
results = {
    'Raw':   compute_raw(),
    'ELR':   compute_prob(f'{GEFS_DIR}/ELR_y_test_wk3-4.nc',
                          f'{GEFS_DIR}/ELR_predictions_wk3-4.nc', 'ELR', is_elr=True),
    'U-Net': compute_prob(f'{GEFS_DIR}/unet_y_test_wk3-4.nc',
                          f'{GEFS_DIR}/unet_predictions_wk3-4.nc', 'U-Net', is_elr=False),
}
print("\n=== Summary ===")
for m, r in results.items():
    print(f"{m}: overall = {r['overall']:.4f}")

# =====================================================================
# Color scheme: per category, three shades for [Raw, ELR, U-Net]
#   Raw  = lightest, ELR = medium, U-Net = darkest
# =====================================================================
MODELS = ['Raw', 'ELR', 'U-Net']  # light -> dark

def shades(base_hex, n=3):
    """Return n colors from lightest (mix-with-white) to darkest (mix-with-black) of base.
    Order: index 0 = lightest, index n-1 = darkest."""
    r, g, b = to_rgb(base_hex)
    # tints: 0 = pure white, 1 = pure base; shades: 0 = pure base, 1 = pure black
    # We'll go light tint -> base -> slight shade across n=3.
    levels = [0.55, 0.0, -0.40]  # positive = mix with white, negative = mix with black
    out = []
    for L in levels[:n]:
        if L > 0:
            cr = r + (1 - r) * L
            cg = g + (1 - g) * L
            cb = b + (1 - b) * L
        else:
            f = 1 + L  # 1 -> 0 as L -> -1
            cr = r * f; cg = g * f; cb = b * f
        out.append((cr, cg, cb))
    return out

# ENSO category base colors — softer, slightly desaturated
ENSO_BASE = {-1: '#3B6FB4',  # La Niña – muted ocean blue
              0: '#8E8E93',  # Neutral  – warm gray
              1: '#C44A4E'}  # El Niño  – muted brick red
enso_palette = {k: shades(v, n=3) for k, v in ENSO_BASE.items()}

# MJO: 8 hues sampled from matplotlib's Spectral colormap (a perceptually
# smooth red->yellow->blue spectrum), cycling phases P1..P8 around the rainbow.
_spec = mpl.colormaps['Spectral']
MJO_BASE = {p: mpl.colors.to_hex(_spec((p - 1) / 7.0)) for p in range(1, 9)}
mjo_palette = {k: shades(v, n=3) for k, v in MJO_BASE.items()}

enso_labels = {-1: 'La Niña', 0: 'Neutral', 1: 'El Niño'}
enso_keys   = [-1, 0, 1]
mjo_keys    = list(range(1, 9))

# =====================================================================
# Three-row figure: one row per model (Raw / ELR / U-Net)
# =====================================================================
# Use the medium (i=1) shade for each category so bars are uniformly saturated.
enso_solid = {k: enso_palette[k][1] for k in enso_keys}
mjo_solid  = {k: mjo_palette[k][1]  for k in mjo_keys}

fig2 = plt.figure(figsize=(5.8, 4.6))
gs2 = fig2.add_gridspec(3, 2, width_ratios=[1, 2.4], wspace=0.18, hspace=0.30,
                        left=0.10, right=0.985, bottom=0.09, top=0.93)

axes_e = []
axes_m = []
ROW_ORDER = ['U-Net', 'ELR', 'Raw']
for row, model in enumerate(ROW_ORDER):
    # Share y across all rows and across ENSO/MJO panels within a row.
    sharey_e = axes_e[0] if axes_e else None
    ax_e = fig2.add_subplot(gs2[row, 0], sharey=sharey_e)
    ax_m = fig2.add_subplot(gs2[row, 1], sharey=ax_e)
    axes_e.append(ax_e); axes_m.append(ax_m)

    # ENSO bars
    for xi, k in enumerate(enso_keys):
        v = results[model]['enso'][k][0]
        ax_e.bar(xi, v, width=0.58, color=enso_solid[k], edgecolor='white', linewidth=0.5)
    ax_e.axhline(0, color='black', linewidth=0.4)
    ax_e.set_xticks(np.arange(len(enso_keys)))
    ax_e.set_ylabel('RPSS')
    # Model name annotation in top-left corner of the ENSO panel
    ax_e.text(0.02, 0.95, model, transform=ax_e.transAxes,
              ha='left', va='top', fontweight='bold', fontsize=10)

    # MJO bars
    for xi, k in enumerate(mjo_keys):
        v = results[model]['mjo'][k][0]
        ax_m.bar(xi, v, width=0.58, color=mjo_solid[k], edgecolor='white', linewidth=0.5)
    ax_m.axhline(0, color='black', linewidth=0.4)
    ax_m.set_xticks(np.arange(len(mjo_keys)))
    ax_m.tick_params(axis='y', labelleft=False)

    # Top-row gets bold panel titles; x-axis tick labels are kept on every row
    if row == 0:
        ax_e.set_title('ENSO mode', loc='center', fontweight='bold')
        ax_m.set_title('MJO phase', loc='center', fontweight='bold')
    ax_e.set_xticklabels([enso_labels[k] for k in enso_keys])
    ax_m.set_xticklabels([f'P{k}' for k in mjo_keys])
    if row == len(ROW_ORDER) - 1:
        ax_e.set_xlabel('ENSO mode (Niño-4)')
        ax_m.set_xlabel('MJO phase')

# Global y-lim across all rows
all_vals = []
for model in ROW_ORDER:
    all_vals += [results[model]['enso'][k][0] for k in enso_keys]
    all_vals += [results[model]['mjo'][k][0]  for k in mjo_keys]
y_hi = max(all_vals); y_lo = min(min(all_vals), 0)
pad = (y_hi - y_lo) * 0.10 + 1e-4
axes_e[0].set_ylim(y_lo - pad, y_hi + pad)  # propagates via sharey

# Save
out_pdf_3 = os.path.join(FIG_DIR, 'RPSS_drivers_combined_by_model.pdf')
out_png_3 = os.path.join(FIG_DIR, 'RPSS_drivers_combined_by_model.png')
plt.savefig(out_pdf_3, bbox_inches='tight')
plt.savefig(out_png_3, bbox_inches='tight', dpi=300)
print(f"Wrote {out_pdf_3}\nWrote {out_png_3}")

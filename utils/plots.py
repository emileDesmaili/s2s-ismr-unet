

from matplotlib.patches import Patch, Polygon
from operator import sub
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader


def compute_reliability_score(y_true, y_pred, num_bins=10):
    """
    Compute the reliability score for probabilistic forecasts.

    Parameters:
    y_true (numpy array): Array of true binary outcomes (0 or 1).
    y_pred (numpy array): Array of predicted probabilities (between 0 and 1).
    num_bins (int): Number of bins to use for grouping predicted probabilities.

    Returns:
    float: The reliability score.
    """
    
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Initialize arrays to hold the average predicted probabilities and observed frequencies for each bin
    bin_avg_pred = np.zeros(num_bins)
    bin_avg_true = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    # Bin edges
    bin_edges = np.linspace(0, 1, num_bins + 1)
    
    # Assign each prediction to a bin
    bin_indices = np.digitize(y_pred, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure indices are within valid range
    
    # Calculate bin statistics
    for i in range(num_bins):
        # Indices of predictions in the current bin
        bin_mask = (bin_indices == i)
        bin_count = np.sum(bin_mask)
        
        if bin_count > 0:
            # Average predicted probability for the current bin
            bin_avg_pred[i] = np.mean(y_pred[bin_mask])
            # Observed frequency (empirical probability) for the current bin
            bin_avg_true[i] = np.mean(y_true[bin_mask])
            bin_counts[i] = bin_count
    
    # Calculate the reliability score
    reliability_score = np.nansum((bin_avg_pred - bin_avg_true)**2 * bin_counts) / np.sum(bin_counts)
    
    return reliability_score

def compute_brier_skill_score(y_pred, t):
    """
    Compute the Brier Skill Score (BSS) for probabilistic forecasts.

    Parameters:
    y_pred (numpy array): Predicted probabilities.
    t (numpy array): True binary outcomes (0 or 1).

    Returns:
    float: The Brier Skill Score (BSS).
    """
    # Adjust y_pred to avoid edge cases
    y_pred = y_pred * 0.9999999999999
    
    # Mask to remove NaNs
    msk = np.where(~np.isnan(y_pred + t))
    y_pred = y_pred[msk]
    t = t[msk]
    
    # Calculate the base rate (climatological mean)
    base_rate = np.nanmean(t)
    base_rate = np.ones_like(np.nanmean(t)) * 1/3
    
    # Calculate the Brier Score (BS)
    bs = np.nanmean((y_pred - t)**2)
    
    # Calculate the Brier Score for the reference forecast (BR)
    br = np.nanmean((base_rate - t)**2)
    
    # Calculate the Brier Skill Score (BSS)
    bss = 1 - (bs / br)
    
    return bss

def compute_resolution_score(predictions, observations, num_bins=10):
    """
    Compute the resolution score for probabilistic forecasts.

    Parameters:
    predictions (numpy array): Predicted probabilities.
    observations (numpy array): True binary outcomes (0 or 1).
    num_bins (int): Number of bins to use for grouping predicted probabilities.

    Returns:
    float: The resolution score.
    """
    # Adjust predictions to avoid edge cases
    predictions = predictions * 0.9999999999999
    
    # Mask to remove NaNs
    mask = np.where(~np.isnan(predictions + observations))
    predictions = predictions[mask]
    observations = observations[mask]
    
    # Calculate the base rate (climatological mean)
    base_rate = np.nanmean(observations)
    
    # Initialize arrays to hold the average observed frequencies and counts for each bin
    bin_obs_freq = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    # Bin edges
    bin_edges = np.linspace(0, 1, num_bins + 1)
    
    # Assign each prediction to a bin
    bin_indices = np.digitize(predictions, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure indices are within valid range
    
    # Calculate bin statistics
    for i in range(num_bins):
        bin_mask = (bin_indices == i)
        bin_counts[i] = bin_mask.sum()
        
        if bin_counts[i] > 0:
            bin_obs_freq[i] = observations[bin_mask].mean()
    
    # Calculate the resolution score
    resolution = np.nansum(bin_counts * (bin_obs_freq - base_rate)**2) / np.sum(bin_counts)
    
    return resolution

def reliability_diagram(ypred, t, title=None, perfect_reliability_line=True, plot_hist=True, fig=None, ax=None, bin_minimum_pct=0.01, tercile_skill_area=True, scores=True):
    """
    Compute and plot a reliability diagram (calibration curve) with a normalized histogram of the forecast probabilities.

    Parameters:
    ypred (numpy array): Predicted probabilities.
    t (numpy array): True binary outcomes (0 or 1).
    title (str): Title for the plot.
    perfect_reliability_line (bool): Whether to plot the perfect reliability line.
    plot_hist (bool): Whether to plot the histogram of forecast probabilities.
    fig (matplotlib Figure): Figure object to plot on.
    ax (matplotlib Axes): Axes object to plot on.
    bin_minimum_pct (float): Minimum percentage of samples in a bin to include it in the plot.
    """


    ypred = ypred * 0.9999999999999  # Avoids edge cases with digitize
    assert ypred.shape == t.shape, 'Inconsistent shapes between ypred and t - {} vs {}'.format(ypred.shape, t.shape)

    # Mask to remove NaNs
    msk = np.where(~np.isnan(ypred + t))
    ypred = ypred[msk]
    t = t[msk]

    # Total number of non-NaN samples
    countnonnan = np.ones_like(ypred).sum()

    # Initialize arrays
    bin_avg_pred = np.zeros(10)
    bin_obs_freq = np.zeros(10)
    bin_counts = np.zeros(10)

    # Iterate over bins
    for i in range(10):
        bin_mask = (ypred >= i / 10.0) & (ypred < (i / 10.0 + 0.1))
        bin_counts[i] = bin_mask.sum()
        
        if bin_counts[i] > 0:
            bin_avg_pred[i] = ypred[bin_mask].mean()
            bin_obs_freq[i] = t[bin_mask].mean()

    # Compute bin centers
    bin_centers = (np.arange(10) + 0.5) / 10.0

    # Mask bins with insufficient data
    valid_bins = bin_counts / countnonnan >= bin_minimum_pct
    bin_centers = bin_centers[valid_bins]
    bin_avg_pred = bin_avg_pred[valid_bins]
    bin_obs_freq = bin_obs_freq[valid_bins]
    bin_counts = bin_counts[valid_bins]

    # Normalize bin counts for the histogram
    bin_counts = bin_counts / countnonnan

    if ax is None:
        fig, ax = plt.subplots()

    b1, t1 = ax.set_ylim(0, 1)
    l, r = ax.set_xlim(0, 1)

    #plt.hist(epoelm_xval[:, 0], bins=11)
    if tercile_skill_area:
        ur = Polygon([[0.33, 0.33 ], [0.33, 1], [1,1], [1, 1.33/2.0]], facecolor='gray', alpha=0.25)
        bl = Polygon([[0.33, 0.33 ], [0.33, 0], [0,0], [0, 0.33/2.0]], facecolor='gray', alpha=0.25)
        ax.add_patch(ur)
        ax.add_patch(bl)

        ax.text(0.66, 0.28, 'No Resolution')
        noresolution = ax.plot([0, 1], [0.33,0.33], lw=0.5, linestyle='dotted')

        noskill = ax.plot([0, 1], [0.33/2.0,1.33/2.0], lw=0.5, linestyle='dotted')
        figW, figH = ax.get_figure().get_size_inches()
        _, _, w, h = ax.get_position().bounds
        disp_ratio = (figH * h) / (figW * w)
        data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
        angle = (180.0/np.pi)*np.arctan(disp_ratio / data_ratio)
        ax.text(0.66, 0.45, 'No Skill', rotation=angle*0.5)

        ax.plot([0.33, 0.33], [0,1], lw=0.5, linestyle='dotted')

    # Plot the reliability diagram
    ax.plot(bin_centers, bin_obs_freq, marker='o', linestyle='-', color='red', label='Observed Frequency')

    # Plot the perfect reliability line
    if perfect_reliability_line:
        ax.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Perfect Reliability')

    # Plot normalized histogram
    if plot_hist:
        ax.bar(bin_centers, bin_counts, width=0.1, color='violet', alpha=0.5, label='Normalized Counts')

    if scores:
        bss = compute_brier_skill_score(ypred, t)
        rel = compute_reliability_score(t, ypred)
        res = compute_resolution_score(ypred, t)
        ax.text( 0.7, 0.11, 'BSS: {:0.04f}'.format(bss))
        ax.text( 0.7, 0.06, 'REL: {:0.04f}'.format(rel))
        ax.text( 0.7, 0.01, 'RES: {:0.04f}'.format(res))

    ax.set_xlabel('Forecast Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='upper left')


    plt.show()


def reliability_diagram_compare(y_pred_NN, t_NN, y_pred_elr, t_elr, title=None, perfect_reliability_line=True, plot_hist=True, fig=None, ax=None, bin_minimum_pct=0.01, tercile_skill_area=True, scores=True, week=None,
                                model=None, obs=None,dir=None):
    """
    Compute and plot a reliability diagram (calibration curve) with a normalized histogram of the forecast probabilities
    for both NN and ELR models.

    Parameters:
    y_pred_NN (numpy array): Predicted probabilities from NN model.
    t_NN (numpy array): True binary outcomes from NN model.
    y_pred_elr (numpy array): Predicted probabilities from ELR model.
    t_elr (numpy array): True binary outcomes from ELR model.
    title (str): Title for the plot.
    perfect_reliability_line (bool): Whether to plot the perfect reliability line.
    plot_hist (bool): Whether to plot the histogram of forecast probabilities.
    fig (matplotlib Figure): Figure object to plot on.
    ax (matplotlib Axes): Axes object to plot on.
    bin_minimum_pct (float): Minimum percentage of samples in a bin to include it in the plot.
    """

    def compute_bin_stats(y_pred, t, num_bins=10):
        y_pred = y_pred * 0.9999999999999  # Avoids edge cases with digitize
        assert y_pred.shape == t.shape, 'Inconsistent shapes between ypred and t - {} vs {}'.format(y_pred.shape, t.shape)

        # Mask to remove NaNs
        msk = np.where(~np.isnan(y_pred + t))
        y_pred = y_pred[msk]
        t = t[msk]

        # Total number of non-NaN samples
        countnonnan = np.ones_like(y_pred).sum()

        # Initialize arrays
        bin_avg_pred = np.zeros(num_bins)
        bin_obs_freq = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)

        # Iterate over bins
        for i in range(num_bins):
            bin_mask = (y_pred >= i / 10.0) & (y_pred < (i / 10.0 + 0.1))
            bin_counts[i] = bin_mask.sum()

            if bin_counts[i] > 0:
                bin_avg_pred[i] = y_pred[bin_mask].mean()
                bin_obs_freq[i] = t[bin_mask].mean()

        # Compute bin centers
        bin_centers = (np.arange(num_bins) + 0.5) / num_bins

        # Mask bins with insufficient data
        valid_bins = bin_counts / countnonnan >= bin_minimum_pct
        bin_centers = bin_centers[valid_bins]
        bin_avg_pred = bin_avg_pred[valid_bins]
        bin_obs_freq = bin_obs_freq[valid_bins]
        bin_counts = bin_counts[valid_bins]

        # Normalize bin counts for the histogram
        bin_counts = bin_counts / countnonnan

        return bin_centers, bin_avg_pred, bin_obs_freq, bin_counts

    # Compute bin statistics for both NN and ELR
    bin_centers_NN, bin_avg_pred_NN, bin_obs_freq_NN, bin_counts_NN = compute_bin_stats(y_pred_NN, t_NN)
    bin_centers_elr, bin_avg_pred_elr, bin_obs_freq_elr, bin_counts_elr = compute_bin_stats(y_pred_elr, t_elr)

    if ax is None:
        fig, ax = plt.subplots()

    # Plot the shaded "No Skill" and "No Resolution" areas
    if tercile_skill_area:
        ur = Polygon([[0.33, 0.33 ], [0.33, 1], [1,1], [1, 1.33/2.0]], facecolor='gray', alpha=0.25)
        bl = Polygon([[0.33, 0.33 ], [0.33, 0], [0,0], [0, 0.33/2.0]], facecolor='gray', alpha=0.25)
        ax.add_patch(ur)
        ax.add_patch(bl)

        ax.text(0.66, 0.28, 'No Resolution')
        noresolution = ax.plot([0, 1], [0.33,0.33], lw=0.5, linestyle='dotted')

        noskill = ax.plot([0, 1], [0.33/2.0,1.33/2.0], lw=0.5, linestyle='dotted')
        figW, figH = ax.get_figure().get_size_inches()
        _, _, w, h = ax.get_position().bounds
        disp_ratio = (figH * h) / (figW * w)
        data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
        angle = (180.0/np.pi)*np.arctan(disp_ratio / data_ratio)
        ax.text(0.66, 0.45, 'No Skill', rotation=angle*0.5)

        ax.plot([0.33, 0.33], [0,1], lw=0.5, linestyle='dotted')

    # Plot the reliability diagram for NN
    ax.plot(bin_centers_NN, bin_obs_freq_NN, marker='o', linestyle='-', color='#008080', label='NN')

    # Plot the reliability diagram for ELR
    ax.plot(bin_centers_elr, bin_obs_freq_elr, marker='o', linestyle='-', color='#FF1493', label='ELR')

    # Plot the perfect reliability line
    if perfect_reliability_line:
        ax.plot([0, 1], [0, 1], linestyle='--', color='blue', label='Perfect Reliability')

    # Plot normalized histogram as step plot
    if plot_hist:
        ax.bar(bin_centers_NN, bin_counts_NN, width=0.1, color='#008080', alpha=0.5, label='NN Normalized')
        ax.bar(bin_centers_elr, bin_counts_elr, width=0.1, color='#FF1493', alpha=0.5, label='ELR Normalized')

    if scores:
        bss_NN = compute_brier_skill_score(y_pred_NN, t_NN)
        rel_NN = compute_reliability_score(t_NN, y_pred_NN)
        res_NN = compute_resolution_score(y_pred_NN, t_NN)

        bss_elr = compute_brier_skill_score(y_pred_elr, t_elr)
        rel_elr = compute_reliability_score(t_elr, y_pred_elr)
        res_elr = compute_resolution_score(y_pred_elr, t_elr)

        ax.text(0.75, 0.48, f'NN BSS: {bss_NN:.4f}\nNN REL: {rel_NN:.4f}\nNN RES: {res_NN:.4f}', 
                verticalalignment='top', transform=ax.transAxes, color='#008080')
        ax.text(0.75, 0.25, f'ELR BSS: {bss_elr:.4f}\nELR REL: {rel_elr:.4f}\nELR RES: {res_elr:.4f}', 
                verticalalignment='top', transform=ax.transAxes, color='#FF1493')

    ax.set_xlabel('Forecast Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='upper left')
    
    # if len(str(week))>1:
    #     plt.title(f'Weeks {str(week)[0]}&{str(week)[1]} Reliability Diagram')
    # else:
    #     plt.title(f'Week {str(week)} Reliability Diagram')
    
    # plt.savefig(f'./plots/Reliability Diagram - Week {week}.jpg', dpi=600)

    title_str = title.replace("-","_")
    title_str = title_str.replace(" ","_")

    plt.savefig('figures/'+ (dir or '') + f'{model}_{obs}/{title_str}.pdf', dpi=300, transparent=True)

    plt.show()
    
    return bin_centers_NN, bin_avg_pred_NN, bin_obs_freq_NN, bin_counts_NN



def plot_rpss_elr(rpss_train_list, rpss_test_list,
                    levels = None,
                    vmin = -0.2,vmax=0.2,
                    model='GEFS',
                    dir=None,
                    obs="IMD",week="wk3-4",cbar_kwargs={'shrink': 0.8,'spacing':'proportional', 'orientation': 'vertical'}):
        # Create the figure and two subplots with Cartopy projection
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={'projection': ccrs.PlateCarree()})

    rpss_train = xr.concat(rpss_train_list, dim='bootstrap').mean(dim='bootstrap')
    rpss_test = xr.concat(rpss_test_list, dim='bootstrap').mean(dim='bootstrap')


    rpss_train_mean = rpss_train.mean()
    rpss_mean = rpss_test.mean()

    rpss_train_min = rpss_train.min()
    rpss_min = rpss_test.min()

    rpss_train_max = rpss_train.max()
    rpss_max = rpss_test.max()

    # Loop over both subplots to add custom shapes
    for a in ax:  # Loop over both axes
        for shape_name in ['indian_borders', 'sd_boundary']:  # Shape names
            reader = cartopy.io.shapereader.Reader(f'shapes/{shape_name}.shp')  # Adjust path if needed
            a.add_geometries(reader.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black')



    # Plot RPSS with first subplot
    rpss_train.plot(ax=ax[0],
            levels=levels,
            vmin=vmin, vmax=vmax,
            cmap='bwr',
            cbar_kwargs=cbar_kwargs)

    # Plot RPSS for training with second subplot
    rpss_test.plot(ax=ax[1],
                    levels=levels,
                    vmin=vmin, vmax=vmax,
                    cmap='bwr',
                    cbar_kwargs=cbar_kwargs)


    # Set titles for the subplots with bold font
    ax[0].set_title(f'Train:\n mean:{rpss_train_mean.values:.2f}, max:{rpss_train_max.values:.2f}, min: {rpss_train_min.values:.2f}', fontweight='bold')
    ax[1].set_title(f'Test: \nmean:{rpss_mean.values:.2f}, max:{rpss_max.values:.2f}, min: {rpss_min.values:.2f}', fontweight='bold')


    # Show the plot

    title = f"{model}-{obs} - {week} RPSS ELR"
    #plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    week_title = week.replace("-","")
    file_str = f"{week_title}_RPSS_ELR"
    
    for a in ax:
        a.set_xlabel("Longitude", fontsize=11)
        a.set_ylabel("Latitude", fontsize=11)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    
    plt.savefig('figures/'+ (dir or '') + f'{model}_{obs}/{file_str}.png', dpi=300, format = 'png')
    plt.show()


def plot_rpss_deepnet(rpss_train_list, rpss_val_list, rpss_test_list,mask=None, 
                      levels = None,
                      vmin = -0.2,vmax=0.2,
                      custom_title = None,
                      model = 'GEFS',
                      dir = None,
                      obs="IMD",week="wk3-4",architecture="unet", cbar_kwargs={'shrink': 0.8,'spacing':'proportional', 'orientation': 'vertical'}):
    # Create the figure and two subplots with Cartopy projection
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': ccrs.PlateCarree()})
    #put architecture in caps
    architecture = architecture.upper()

        # Loop over both subplots to add custom shapes
    for a in ax:  # Loop over both axes
        for shape_name in ['indian_borders', 'sd_boundary']:  # Shape names
            reader = cartopy.io.shapereader.Reader(f'shapes/{shape_name}.shp')  # Adjust path if needed
            a.add_geometries(reader.geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='black')

    

    rpss_train = xr.concat(rpss_train_list, dim='bootstrap').mean(dim='bootstrap')
    rpss_test = xr.concat(rpss_test_list, dim='bootstrap').mean(dim='bootstrap')
    rpss_val = xr.concat(rpss_val_list, dim='bootstrap').mean(dim='bootstrap')

    #apply mask to rpss to turn masked into 0
    rpss_train = rpss_train.where(~mask)
    rpss_test = rpss_test.where(~mask)
    rpss_val = rpss_val.where(~mask)

    rpss_train_mean = rpss_train.mean()
    rpss_mean = rpss_test.mean()
    rpss_val_mean = rpss_val.mean()

    rpss_train_min = rpss_train.min()
    rpss_min = rpss_test.min()
    rpss_val_min = rpss_val.min()

    rpss_train_max = rpss_train.max()
    rpss_max = rpss_test.max()
    rpss_val_max = rpss_val.max()

    # Plot RPSS with first subplot
    rpss_train.plot(ax=ax[0],
            #vmin=-0.2, vmax=0.2,
            cmap='bwr',
            levels=levels,
            vmin=vmin, vmax=vmax,
            cbar_kwargs=cbar_kwargs)

    rpss_val.plot(ax=ax[1],
                    cmap='bwr',
                    levels=levels,
                    vmin=vmin, vmax=vmax,
                    cbar_kwargs=cbar_kwargs)
    
    rpss_test.plot(ax=ax[2],
                    cmap='bwr',
                levels=levels,
                vmin=vmin, vmax=vmax,
                    cbar_kwargs=cbar_kwargs)

    # Set titles for the subplots with bold font
    ax[0].set_title(f'Train:\n mean:{rpss_train_mean.values:.2f}, max:{rpss_train_max.values:.2f}, min: {rpss_train_min.values:.2f}', fontweight='bold')
    ax[1].set_title(f'Validation: \nmean:{rpss_val_mean.values:.2f}, max:{rpss_val_max.values:.2f}, min: {rpss_val_min.values:.2f}', fontweight='bold')
    ax[2].set_title(f'Test: \nmean:{rpss_mean.values:.2f}, max:{rpss_max.values:.2f}, min: {rpss_min.values:.2f}', fontweight='bold')


    # Show the plot
    if custom_title is not None:
        title = f"{model}-{obs} - {week} RPSS {architecture} {custom_title}"
    else:
        title = f"{model}-{obs} - {week} RPSS {architecture}"

    #plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    week_title = week.replace("-","")
    if custom_title is not None:
        file_str = f"{week_title}_RPSS_{architecture}_{custom_title}"
    else:
        file_str = f"{week_title}_RPSS_{architecture}"

    for a in ax:
        a.set_xlabel("Longitude", fontsize=11)
        a.set_ylabel("Latitude", fontsize=11)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.savefig('figures/'+ (dir or '') + f'{model}_{obs}/{file_str}.png', dpi=300, format = 'png')
    plt.show()
    #plt.tight_layout()
    plt.show()
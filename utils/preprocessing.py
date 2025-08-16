import cartopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import xarray as xr
import xskillscore
import tensorflow as tf


def make_tercile_labeler(observations):
    edges = observations.quantile(q=[1/3, 2/3], dim='T')
    def labeler(y):
        return (
            xr.where(y < edges.isel(quantile=0), 0,   # label as 0 if below first tercile
                      xr.where(y > edges.isel(quantile=1), 2,  # label as 2 if above second tercile
                               1))  # otherwise, label as 1 (between the two edges)
        ).where(y.notnull()) 
    return labeler

def create_mean_predictor_images(xt):
    mean_xt = xt.mean(dim='M')
    return mean_xt.values

def create_multi_predictor_images(xt):
    stacked_xt = xt.transpose('T', 'Y', 'X', 'M')
    return stacked_xt.values

def create_stacked_predictor_images(xt,yt):
    stacked_xt = xt.stack(MT=('M', 'T')).transpose('MT', 'Y', 'X')
    len_M = len(xt['M'])
    #stack y M times to match the number of predictors
    yt_stacked = np.tile(yt.values, (len_M, 1, 1))
    yt_stacked_xr = xr.DataArray(yt_stacked, dims=['MT', 'Y', 'X'], coords={'MT': stacked_xt['MT'], 'Y': stacked_xt['Y'], 'X': stacked_xt['X']})
    return stacked_xt, yt_stacked, yt_stacked_xr


def convert_to_ndarray(xt, yt, type="mean"):
    if type == "mean":
        X_train = create_mean_predictor_images(xt)
        yt_np = yt.values
        return X_train, yt_np
    elif type == "multi_predictor":
        X_train = create_multi_predictor_images(xt)
        yt_np = yt.values
        return X_train, yt_np
    elif type =="stacked":
        X_train_xr, yt_np, yt_xr = create_stacked_predictor_images(xt,yt)
        return X_train_xr, yt_np, yt_xr
    
    

def rolling_labeler(observations,window=1):

    """
    Creates a labeling function that dynamically assigns tercile-based labels to data based on a rolling window 
    of weeks, using quantiles computed from similar weeks.

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        The input dataset containing the `T` dimension (time) and the data to compute quantile edges. 
        The `T` dimension must represent timestamps, which will be converted to ISO calendar week numbers.

    window : int, optional
        The size of the rolling window (in weeks) used to compute quantiles. For example, a window of 1 
        includes 1 week before and 1 week after the target week, along with the target week itself. 
        Default is 1.

    Returns
    -------
    labeler : function
        A function that takes an xarray object `y` (with a `T` dimension) as input and returns the same object 
        with labels assigned. Labels are computed based on the quantile edges derived from the rolling 
        window of weeks.

    Labeler Parameters
    ------------------
    y : xarray.Dataset or xarray.DataArray
        The input data to label, containing a `T` dimension with timestamps. The `T` dimension must align with 
        the `T` dimension in the `observations` dataset.

    Returns
    -------
    y_labeled : xarray.DataArray
        A labeled xarray object where each value is assigned one of three labels:
        - 0: Below the first quantile edge (lower tercile)
        - 1: Between the first and second quantile edges (middle tercile)
        - 2: Above the second quantile edge (upper tercile)
        - NaN: If the quantile edges could not be computed for the corresponding week or the week has insufficient data.

    Notes
    -----
    1. The function uses ISO calendar weeks to group and process data, with weeks ranging from 1 to 53.
    2. When computing the rolling window, weeks wrap around, ensuring that week 1 considers week 52 from the previous year.
    3. The function handles cases where quantile edges cannot be computed due to insufficient or degenerate data, 
       assigning NaN to these points.

    """

    observations['T'] = pd.to_datetime(observations['T'].values)

    # Extract week numbers from 'T'
    week_values = observations['T'].dt.isocalendar().week.values

    # Add week as a coordinate variable
    observations = observations.assign_coords(week=('T', week_values))

    edges_list = []

    # Iterate over each unique week to compute quantiles
    for week in np.unique(week_values):
        # Select data for the specific week, 3 weeks before and 3 weeks after (6-week window)
        week_window = [(week + i) % 53 or 53 for i in range(-window, 1 + window)]

        observations_weekly = observations.sel(T=observations['week'].isin(week_window))

        # Compute quantiles for the selected week window
        edges = observations_weekly.quantile([1/3, 2/3], dim='T')
        
        # Store the quantiles with 'week' as a dimension
        edges = edges.assign_coords(week=week)
        edges_list.append(edges)

    # Concatenate edges across all weeks along the 'week' dimension
    edges = xr.concat(edges_list, dim='week')

    def labeler(y):
        # Ensure the 'T' coordinate is a datetime index
        y['T'] = pd.to_datetime(y['T'].values)
        
        # Extract week numbers from the 'T' dimension
        week_values = y['T'].dt.isocalendar().week.values
        y = y.assign_coords(week=('T', week_values))
        
        labeled_list = []
        
        # Iterate over each unique week
        for week in np.unique(week_values):
            # Select the corresponding edges for the current week
            edges_weekly = edges.sel(week=week, method='nearest')
            
            # Select the observations for the current week
            y_weekly = y.sel(T=y['week'].isin(week))

            mask = (
            edges_weekly.isnull().any('quantile') 
            # degenerate points where there aren't enough unique
            # values to identify tercile
            #(edges_weekly.isel(quantile=0) == 0) 
            #(edges_weekly.isel(quantile=0) == edges_weekly.isel(quantile=1))
                        )
            
            # Label the data based on quantile edges
            y_weekly_labeled = xr.where(
                y_weekly < edges_weekly.isel(quantile=0), 0,
                xr.where(y_weekly > edges_weekly.isel(quantile=1), 2, 1)
            ).where(np.logical_not(mask))
            
            # Append the labeled data for the current week
            labeled_list.append(y_weekly_labeled)
            
        # Concatenate the labeled data across all weeks
        y_labeled = xr.concat(labeled_list, dim='T')
        return y_labeled.sortby('T')
    
    return labeler




def rolling_labeler_logistic(observations,window=0):
    observations['T'] = pd.to_datetime(observations['T'].values)

    # Extract week numbers from 'T'
    week_values = observations['T'].dt.isocalendar().week.values

    # Add week as a coordinate variable
    observations = observations.assign_coords(week=('T', week_values))

    edges_list = []

    # Iterate over each unique week to compute quantiles
    for week in np.unique(week_values):
        # Select data for the specific week, 3 weeks before and 3 weeks after (6-week window)
        #week_window = [(week + i) % 52 if (week + i) > 0 else 52 for i in range(-window, 1+window)]  # Handle 6-week cycle with wrap-around
        week_window = [(week + i) % 53 or 53 for i in range(-window, 1 + window)]

        observations_weekly = observations.sel(T=observations['week'].isin(week_window))

        # Compute quantiles for the selected week window
        edges = observations_weekly.quantile([1/3, 2/3], dim='T')
        
        # Store the quantiles with 'week' as a dimension
        edges = edges.assign_coords(week=week)
        edges_list.append(edges)

    # Concatenate edges across all weeks along the 'week' dimension
    edges = xr.concat(edges_list, dim='week')

    def labeler(y):
        # Ensure the 'T' coordinate is a datetime index
        y['T'] = pd.to_datetime(y['T'].values)
        
        # Extract week numbers from the 'T' dimension
        week_values = y['T'].dt.isocalendar().week.values
        y = y.assign_coords(week=('T', week_values))
        
        labeled_list = []
        
        # Iterate over each unique week
        for week in np.unique(week_values):
            # Select the corresponding edges for the current week
            edges_weekly = edges.sel(week=week,method='nearest')
            
            # Select the observations for the current week
            y_weekly = y.sel(T=y['week'].isin(week))

            mask = (
                edges_weekly.isnull().any('quantile') |
                # Points where quantile 0 equals quantile 1 (degenerate case)
                (edges_weekly.isel(quantile=0) == 0) | 
                (edges_weekly.isel(quantile=0) == edges_weekly.isel(quantile=1))
            )

            # Adjust the labeling
            y_weekly_labeled = xr.where(
                # Label -1 where y_weekly < edges_weekly.isel(quantile=0) but quantile 0 != 0
                (y_weekly < edges_weekly.isel(quantile=0)), 0,
                xr.where(
                    # Label 1 where y_weekly > edges_weekly.isel(quantile=1)
                    (y_weekly > edges_weekly.isel(quantile=1)), 2, 1
                )
            ).where(np.logical_not(mask))


            # Append the labeled data for the current week
            labeled_list.append(y_weekly_labeled)
            
        # Concatenate the labeled data across all weeks
        y_labeled = xr.concat(labeled_list, dim='T')
        return y_labeled.sortby('T')
    
    return labeler

def rolling_labeler_ELR(observations,window=1):
    observations['T'] = pd.to_datetime(observations['T'].values)

    edges_full = observations.quantile([1/3, 2/3], dim='T')

    # Extract week numbers from 'T'
    week_values = observations['T'].dt.isocalendar().week.values

    # Add week as a coordinate variable
    observations = observations.assign_coords(week=('T', week_values))

    edges_list = []

    # Iterate over each unique week to compute quantiles
    for week in np.unique(week_values):
        # Select data for the specific week, 3 weeks before and 3 weeks after (6-week window)
        week_window = [(week + i) % 53 or 53 for i in range(-window, 1 + window)]
        observations_weekly = observations.sel(T=observations['week'].isin(week_window))

        # Compute quantiles for the selected week window
        edges = observations_weekly.quantile([1/3, 2/3], dim='T')
        
        # Store the quantiles with 'week' as a dimension
        edges = edges.assign_coords(week=week)
        edges_list.append(edges)

    # Concatenate edges across all weeks along the 'week' dimension
    edges = xr.concat(edges_list, dim='week')


    def labeler(y):
        # Ensure the 'T' coordinate is a datetime index
        y['T'] = pd.to_datetime(y['T'].values)
        
        # Extract week numbers from the 'T' dimension
        week_values = y['T'].dt.isocalendar().week.values
        y = y.assign_coords(week=('T', week_values))
        
        labeled_list = []
        edges_list_t = []
        y_elr_list = []
        
        # Iterate over each unique week
        for week in np.unique(week_values):
            # Select the corresponding edges for the current week
            edges_weekly = edges.sel(week=week, method='nearest')
            
            # Select the observations for the current week
            y_weekly = y.sel(T=y['week'].isin(week))

            edges_weekly_t = edges_weekly.expand_dims(T=y_weekly['T'])
            edges_weekly_t = edges_weekly_t.assign_coords(T=y_weekly['T'])
            edges_list_t.append(edges_weekly_t)

            mask = (
                edges_weekly.isnull().any('quantile') |
                # Points where quantile 0 equals quantile 1 (degenerate case)
                (edges_weekly.isel(quantile=0) == 0) | 
                (edges_weekly.isel(quantile=0) == edges_weekly.isel(quantile=1))
            )

            # Adjust the labeling
            y_weekly_labeled = xr.where(
                # Label -1 where y_weekly < edges_weekly.isel(quantile=0) but quantile 0 != 0
                (y_weekly < edges_weekly.isel(quantile=0)), 0,
                xr.where(
                    # Label 1 where y_weekly > edges_weekly.isel(quantile=1)
                    (y_weekly > edges_weekly.isel(quantile=1)), 2, 1
                )
            ).where(np.logical_not(mask))

            y_below_33 = xr.where( (y_weekly <= edges_weekly.isel(quantile=0)), 1,0)
            y_below_66 = xr.where( (y_weekly <= edges_weekly.isel(quantile=1)), 1,0)
            #put on top of each other
            y_weekly_elr = xr.concat([y_below_33,y_below_66],dim='quantile').where(np.logical_not(mask))

            # Append the labeled data for the current week
            labeled_list.append(y_weekly_labeled)
            y_elr_list.append(y_weekly_elr)
            
        # Concatenate the labeled data across all weeks
        y_labeled = xr.concat(labeled_list, dim='T')
        edges_t = xr.concat(edges_list_t, dim='T')
        #edges_t = edges_full
        y_elr = xr.concat(y_elr_list, dim='T')
        return y_labeled.sortby('T'), edges_t.sortby('T'), y_elr.sortby('T')
    
    return labeler

def bootstrap_splits(x, y, n_bootstraps=10, frac_valid=0.2, frac_test=0.1,standardize=False):

    #standardize
    if standardize==True:
        x = (x - x.mean(dim='T')) / (x.std(dim='T')+1e-6)
        y = (y - y.mean(dim='T')) / (y.std(dim='T')+1e-6)
    # Fill NaN values with 0
    x = x.fillna(0)
    y = y.fillna(0)

    # Ensure 'T' is in datetime format
    x['T'] = pd.to_datetime(x['T'].values)
    y['T'] = pd.to_datetime(y['T'].values)

    # Extract year information
    years = x['T'].dt.year
    unique_years = np.unique(years)

    # Initialize lists to store bootstrapped datasets
    xtrain_list, ytrain_list = [], []
    xval_list, yval_list = [], []
    xtest_list, ytest_list = [], []

    # Loop through the number of bootstraps
    for i in range(n_bootstraps):
        np.random.seed(i)  # Change seed per bootstrap to ensure randomness
        shuffled_years = np.random.permutation(unique_years)

        # Calculate the number of years for validation and test sets
        n_years = len(shuffled_years)
        n_valid = int(frac_valid * n_years)
        n_test = int(frac_test * n_years)

        # Split years into train, validation, and test
        valid_years = shuffled_years[:n_valid]
        test_years = shuffled_years[n_valid:n_valid + n_test]
        train_years = shuffled_years[n_valid + n_test:]

        # Select data based on the year split
        xtrain = x.where(x['T'].dt.year.isin(train_years), drop=True).sortby('T')
        ytrain = y.where(y['T'].dt.year.isin(train_years), drop=True).sortby('T')

        xval = x.where(x['T'].dt.year.isin(valid_years), drop=True).sortby('T')
        yval = y.where(y['T'].dt.year.isin(valid_years), drop=True).sortby('T')

        xtest = x.where(x['T'].dt.year.isin(test_years), drop=True).sortby('T')
        ytest = y.where(y['T'].dt.year.isin(test_years), drop=True).sortby('T')

        # Append the bootstrapped datasets to the lists
        xtrain_list.append(xtrain)
        ytrain_list.append(ytrain)
        xval_list.append(xval)
        yval_list.append(yval)
        xtest_list.append(xtest)
        ytest_list.append(ytest)

    return xtrain_list, ytrain_list, xval_list, yval_list, xtest_list, ytest_list

def preprocess(xtrain, ytrain, xval, yval, xtest, ytest):
    """Preprocess the data for the deep net models

    Args:
        xtrain (_type_): training data predictor xarray
        ytrain (_type_): training data predictand xarray
        xval (_type_): validation data predictor xarray
        yval (_type_): validation data predictand xarray
        xtest (_type_): testing data predictor xarray
        ytest (_type_): testing data predictand xarray

    Returns:
        _type_: tuple of preprocessed data
        X_train (_type_): training data predictor numpy array
        Y_train_oh (_type_): training data predictand one-hot encoded numpy array
        X_val (_type_): validation data predictor numpy array
        Y_val_oh (_type_): validation data predictand one-hot encoded numpy array
        X_test (_type_): testing data predictor numpy array
        Y_test_oh (_type_): testing data predictand one-hot encoded numpy array
        y_train_terciled (_type_): training data predictand terciled xarray (0-1-2 labels)
        y_val_terciled (_type_): validation data predictand terciled xarray (0-1-2 labels)
    """
    labeler_train = rolling_labeler(ytrain,window=1)

    num_classes = 3
    y_train_terciled = labeler_train(ytrain) 
    y_val_terciled = labeler_train(yval) 
    y_test_terciled = labeler_train(ytest)

    X_train, Y_train_terciled = convert_to_ndarray(xtrain, y_train_terciled, "mean")
    X_val, Y_val_terciled = convert_to_ndarray(xval, y_val_terciled, "mean")
    X_test, Y_test_terciled = convert_to_ndarray(xtest, y_test_terciled, "mean")

    Y_train_oh = tf.keras.utils.to_categorical(Y_train_terciled, num_classes)
    Y_val_oh = tf.keras.utils.to_categorical(Y_val_terciled, num_classes)
    Y_test_oh = tf.keras.utils.to_categorical(Y_test_terciled, num_classes)

    return X_train, Y_train_oh, X_val, Y_val_oh, X_test, Y_test_oh, y_train_terciled, y_val_terciled, y_test_terciled


def preprocess_stacked(xtrain, ytrain, xval, yval, xtest, ytest):
    labeler_train = rolling_labeler(ytrain, window=1)

    num_classes = 3
    y_train_terciled = labeler_train(ytrain) 
    y_val_terciled = labeler_train(yval) 
    y_test_terciled = labeler_train(ytest)

    X_train, Y_train_terciled, y_train_terciled = convert_to_ndarray(xtrain, y_train_terciled, "stacked")
    X_val, Y_val_terciled, y_val_terciled = convert_to_ndarray(xval, y_val_terciled, "stacked")
    X_test, Y_test_terciled, y_test_terciled = convert_to_ndarray(xtest, y_test_terciled, "stacked")

    Y_train_oh = tf.keras.utils.to_categorical(Y_train_terciled, num_classes)
    Y_val_oh = tf.keras.utils.to_categorical(Y_val_terciled, num_classes)
    Y_test_oh = tf.keras.utils.to_categorical(Y_test_terciled, num_classes)

    return X_train, Y_train_oh, X_val, Y_val_oh, X_test, Y_test_oh, y_train_terciled, y_val_terciled, y_test_terciled


def bootstrap_splits_ELR(x, y, n_bootstraps=10, frac_test=0.3, standardize=False):
    # standardize x and y 
    if standardize==True:
        x = (x - x.mean(dim='T')) / (x.std(dim='T')+1e-6)
        y = (y - y.mean(dim='T')) / (y.std(dim='T')+1e-6)

    # Ensure 'T' is in datetime format
    x['T'] = pd.to_datetime(x['T'].values)
    y['T'] = pd.to_datetime(y['T'].values)

    # Extract year information
    years = x['T'].dt.year
    unique_years = np.unique(years)

    # Initialize lists to store bootstrapped datasets
    xtrain_list, ytrain_list = [], []
    xtest_list, ytest_list = [], []

    # Loop through the number of bootstraps
    for i in range(n_bootstraps):
        np.random.seed(i)  # Change seed per bootstrap to ensure randomness
        shuffled_years = np.random.permutation(unique_years)

        # Calculate the number of years for validation and test sets
        n_years = len(shuffled_years)
        n_test = int(n_years * frac_test)

        # Split years into train and test
        train_years = shuffled_years[:-n_test]
        test_years = shuffled_years[-n_test:]

        # Select data based on the year split
        xtrain = x.where(x['T'].dt.year.isin(train_years), drop=True).sortby('T')
        ytrain = y.where(y['T'].dt.year.isin(train_years), drop=True).sortby('T')


        xtest = x.where(x['T'].dt.year.isin(test_years), drop=True).sortby('T')
        ytest = y.where(y['T'].dt.year.isin(test_years), drop=True).sortby('T')

        # Append the bootstrapped datasets to the lists
        xtrain_list.append(xtrain)
        ytrain_list.append(ytrain)
        xtest_list.append(xtest)
        ytest_list.append(ytest)

    return xtrain_list, ytrain_list, xtest_list, ytest_list


def bootstrap_splits_ELR_mme(x_dict, y, n_bootstraps=10, frac_test=0.3, standardize=False):
    """
    Perform bootstrapped train-test splits for multiple models, where x is a dictionary of xarray.DataArray objects.
    
    Args:
        x_dict (dict): Dictionary where keys are model names (str) and values are xarray.DataArray objects.
        y (xarray.DataArray): Target variable, shared across all models.
        n_bootstraps (int, optional): Number of bootstrap splits. Defaults to 10.
        frac_test (float, optional): Fraction of years to include in the test set. Defaults to 0.3.
        standardize (bool, optional): Whether to standardize x and y. Defaults to False.

    Returns:
        dict: Dictionary containing bootstrapped train-test splits for each model with structure:
              {
                  'train': {model_name: [xtrain_1, xtrain_2, ...]},
                  'test': {model_name: [xtest_1, xtest_2, ...]},
                  'y_train': [ytrain_1, ytrain_2, ...],
                  'y_test': [ytest_1, ytest_2, ...]
              }
    """
    # Standardize x and y if required
    if standardize:
        x_dict = {model: (x - x.mean(dim='T')) / (x.std(dim='T') + 1e-6) for model, x in x_dict.items()}
        y = (y - y.mean(dim='T')) / (y.std(dim='T') + 1e-6)


    # Ensure 'T' is in datetime format
    y['T'] = pd.to_datetime(y['T'].values)
    for model in x_dict:
        x_dict[model]['T'] = pd.to_datetime(x_dict[model]['T'].values)

    # Extract year information
    years = y['T'].dt.year
    unique_years = np.unique(years)

    # Initialize lists to store bootstrapped datasets
    xtrain_dict, xtest_dict = {model: [] for model in x_dict}, {model: [] for model in x_dict}
    ytrain_list, ytest_list = [], []

    # Loop through the number of bootstraps
    for i in range(n_bootstraps):
        np.random.seed(i)  # Change seed per bootstrap to ensure randomness
        shuffled_years = np.random.permutation(unique_years)

        # Calculate the number of years for validation and test sets
        n_years = len(shuffled_years)
        n_test = int(n_years * frac_test)

        # Split years into train and test
        train_years = shuffled_years[:-n_test]
        test_years = shuffled_years[-n_test:]

        # Select data based on the year split for each model
        for model, x in x_dict.items():
            xtrain_dict[model].append(x.where(x['T'].dt.year.isin(train_years), drop=True).sortby('T'))
            xtest_dict[model].append(x.where(x['T'].dt.year.isin(test_years), drop=True).sortby('T'))

        # Select data based on the year split for y
        ytrain_list.append(y.where(y['T'].dt.year.isin(train_years), drop=True).sortby('T'))
        ytest_list.append(y.where(y['T'].dt.year.isin(test_years), drop=True).sortby('T'))

    return xtrain_dict,  xtest_dict,  ytrain_list,  ytest_list


def bootstrap_splits_mme(x_dict, y, n_bootstraps=10, frac_valid=0.2, frac_test=0.1, standardize=False):
    """
    Perform bootstrapped train-validation-test splits for multiple models.
    
    Args:
        x_dict (dict): Dictionary where keys are model names (str) and values are xarray.DataArray objects.
        y (xarray.DataArray): Target variable, shared across all models.
        n_bootstraps (int, optional): Number of bootstrap splits. Defaults to 10.
        frac_valid (float, optional): Fraction of years to include in the validation set. Defaults to 0.2.
        frac_test (float, optional): Fraction of years to include in the test set. Defaults to 0.1.
        standardize (bool, optional): Whether to standardize x and y. Defaults to False.

    Returns:
        dict: Dictionary containing bootstrapped train-validation-test splits for each model with structure:
              {
                  'train': {model_name: [xtrain_1, xtrain_2, ...]},
                  'valid': {model_name: [xval_1, xval_2, ...]},
                  'test': {model_name: [xtest_1, xtest_2, ...]},
                  'y_train': [ytrain_1, ytrain_2, ...],
                  'y_valid': [yval_1, yval_2, ...],
                  'y_test': [ytest_1, ytest_2, ...]
              }
    """
    # Standardize x and y if required
    if standardize:
        x_dict = {model: (x - x.mean(dim='T')) / (x.std(dim='T') + 1e-6) for model, x in x_dict.items()}
        y = (y - y.mean(dim='T')) / (y.std(dim='T') + 1e-6)

    #fill nan values with 0
    for model in x_dict:
        x_dict[model] = x_dict[model].fillna(0)
    y = y.fillna(0)

    # Ensure 'T' is in datetime format
    y['T'] = pd.to_datetime(y['T'].values)
    for model in x_dict:
        x_dict[model]['T'] = pd.to_datetime(x_dict[model]['T'].values)

    # Extract year information
    years = y['T'].dt.year
    unique_years = np.unique(years)

    # Initialize lists to store bootstrapped datasets
    xtrain_dict = {model: [] for model in x_dict}
    xval_dict = {model: [] for model in x_dict}
    xtest_dict = {model: [] for model in x_dict}
    ytrain_list, yval_list, ytest_list = [], [], []

    # Loop through the number of bootstraps
    for i in range(n_bootstraps):
        np.random.seed(i)  # Change seed per bootstrap to ensure randomness
        shuffled_years = np.random.permutation(unique_years)

        # Calculate the number of years for validation and test sets
        n_years = len(shuffled_years)
        n_valid = int(n_years * frac_valid)
        n_test = int(n_years * frac_test)

        # Split years into train, validation, and test
        valid_years = shuffled_years[:n_valid]
        test_years = shuffled_years[n_valid:n_valid + n_test]
        train_years = shuffled_years[n_valid + n_test:]

        # Select data based on the year split for each model
        for model, x in x_dict.items():
            xtrain_dict[model].append(x.where(x['T'].dt.year.isin(train_years), drop=True).sortby('T'))
            xval_dict[model].append(x.where(x['T'].dt.year.isin(valid_years), drop=True).sortby('T'))
            xtest_dict[model].append(x.where(x['T'].dt.year.isin(test_years), drop=True).sortby('T'))

        # Select data based on the year split for y
        ytrain_list.append(y.where(y['T'].dt.year.isin(train_years), drop=True).sortby('T'))
        yval_list.append(y.where(y['T'].dt.year.isin(valid_years), drop=True).sortby('T'))
        ytest_list.append(y.where(y['T'].dt.year.isin(test_years), drop=True).sortby('T'))

    return xtrain_dict, xval_dict, xtest_dict, ytrain_list, yval_list, ytest_list
    
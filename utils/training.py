import os

os.environ['PYTHONHASHSEED']=str(42)

import numpy as np
import xarray as xr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import utils.preprocessing as preprocessing
import utils.performance_metrics as performance_metrics
import utils.deep_nn_models as deep_nn_models
import tensorflow as tf
from keras import layers, models, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import itertools
import statsmodels.api as sm
from joblib import Parallel, delayed
import itertools
import random

import keras

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(42)
   tf.random.set_seed(42)
   np.random.seed(42)
   random.seed(42)


def train_single_bootstrap_deepnet(i, xtrain_list, ytrain_list, xval_list, yval_list, xtest_list, ytest_list,
                           architecture_params, tuning_grid, architecture, training_type,
                           predictor, modname, obs, week, epochs, batch_size, learning_rate,dir):

    xtrain = xtrain_list[i]
    ytrain = ytrain_list[i]
    xval = xval_list[i]
    yval = yval_list[i]
    xtest = xtest_list[i]
    ytest = ytest_list[i]

    best_params_dict = {}

    reset_random_seeds()

    if predictor == "mean":


        X_train, Y_train_oh, X_val, Y_val_oh, X_test, Y_test_oh, ytrain_terciled, yval_terciled, ytest_terciled = preprocessing.preprocess(xtrain, ytrain,
                                                                                                                                            xval, yval,
                                                                                                                                                xtest, ytest)
        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        
        if architecture == "unet":
            if architecture_params is not None:
                ct_kernel = architecture_params['ct_kernel']
                n_blocks = architecture_params['n_blocks']
                filters = architecture_params['filters']
            model = deep_nn_models.Unet("", ct_kernel=ct_kernel,
                                        n_blocks=n_blocks, filters=filters,
                                        train_patches=False, weighted_loss=False).build_model(input_shape, dg_train_weight_target=None)
        elif architecture == "cnn":
            model = deep_nn_models.CNN(input_shape = (input_shape[0], input_shape[1], 1))
        elif architecture == "mlp":
            model = deep_nn_models.MLP(input_shape = (input_shape[0], input_shape[1]))

        optim = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=['accuracy'])
        # Train the model
        checkpoint = ModelCheckpoint('models/'+ (dir or '') + f'{modname}_{obs}/{week}/best_model_{architecture}_{i}.keras', save_best_only=True, save_weights_only=False, monitor='val_loss',
                                        mode='min', verbose=0)
            
        if training_type == "tune":

            #define hyperparameter grid
            batch_sizes = tuning_grid['batch_sizes']
            learning_rates = tuning_grid['learning_rates']
            ct_kernels = tuning_grid['ct_kernels']
            n_filters = tuning_grid['n_filters']
            n_blocks = tuning_grid['n_blocks']
            patience = tuning_grid['patience']

            best_val_loss = float("inf")
            best_model_path = None
            best_params = None

            # Iterate over all combinations of hyperparameters
            for trial_num, (bs, lr, ct_kernel, n_filter, n_block) in enumerate(itertools.product(batch_sizes, learning_rates, ct_kernels, n_filters, n_blocks)):
                print(f'Trial {trial_num+1}/ {len(batch_sizes)*len(learning_rates)*len(ct_kernels)*len(n_filters)*len(n_blocks)}')
                print(f'Tuning Combination: Batch size={bs}, LR={lr}, Kernel={ct_kernel}, Filters={n_filter}, Blocks={n_block}')

                model = deep_nn_models.Unet("", ct_kernel=ct_kernel,
                                n_blocks=n_block, filters=n_filter,
                            train_patches=False, weighted_loss=False).build_model(input_shape, dg_train_weight_target=None)
                                            
                optim = optimizers.Adam(learning_rate=lr)
                model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=['accuracy'])

                checkpoint_path = 'models/'+ (dir or '') + f'{modname}_{obs}/{week}/best_model_{architecture}_bootstrap_{i+1}_trial_{trial_num+1}.keras'
                checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min', verbose=0)
                early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

                history = model.fit(x=X_train, y=Y_train_oh, validation_data=(X_val, Y_val_oh), epochs=epochs, batch_size=bs,
                                    callbacks=[checkpoint, early_stopping], shuffle=True, verbose=0)

                # Track the best model based on validation loss
                val_loss = min(history.history['val_loss'])
                print(f'Validation loss for bootstrap {i+1}, trial {trial_num+1}: {val_loss}')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = checkpoint_path
                    best_params = (bs, lr, ct_kernel, n_filter, n_block)

            # Load the best model
            best_model = models.load_model(best_model_path)
            best_model.save('models/'+ (dir or '') + f'{modname}_{obs}/{week}/best_model_{architecture}_{i}_tuned.keras')
            best_params_dict[i] = {'batch_size': best_params[0], 'lr': best_params[1], 'ct_kernel': best_params[2], 'filters': best_params[3], 'blocks': best_params[4], 'val_loss': best_val_loss}
            print(f'Best hyperparameters for bootstrap {i+1}: {best_params_dict[i]}')

        elif training_type == "train":
            history = model.fit(x=X_train, y=Y_train_oh, validation_data=(X_val, Y_val_oh), epochs=epochs, batch_size=batch_size, 
                                callbacks=[checkpoint],
                                shuffle=True, verbose=0)


            best_model = models.load_model('models/'+ (dir or '') + f'{modname}_{obs}/{week}/best_model_{architecture}_{i}.keras')

        elif training_type == "load":
            try:
                best_model = models.load_model('models/'+ (dir or '') + f'{modname}_{obs}/{week}/best_model_{architecture}_{i}_tuned.keras')
            except:
                best_model = models.load_model('models/'+ (dir or '') + f'{modname}_{obs}/{week}/best_model_{architecture}_{i}.keras')

        predictions = best_model.predict(X_test, verbose=0)
        train_predictions = best_model.predict(X_train, verbose=0)
        val_predictions = best_model.predict(X_val, verbose=0)

        dims = ('T', 'Y', 'X', 'category')
        test_predictions_xarray = xr.DataArray(predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})
        train_predictions_xarray = xr.DataArray(train_predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})
        val_predictions_xarray = xr.DataArray(val_predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})

        Y_test_oh_xr = xr.DataArray(Y_test_oh, dims=dims, coords={'category': ['below', 'normal', 'above']})



    if predictor == "stacked":


        X_train, Y_train_oh, X_val, Y_val_oh, X_test, Y_test_oh, ytrain_terciled, yval_terciled, ytest_terciled = preprocessing.preprocess_stacked(xtrain, ytrain,
                                                                                                                                                        xval, yval,
                                                                                                                                                            xtest, ytest)
        input_shape = (X_train.shape[1], X_train.shape[2],1)


        if architecture == "unet":
            if architecture_params is not None:
                ct_kernel = architecture_params['ct_kernel']
                n_blocks = architecture_params['n_blocks']
                filters = architecture_params['filters']
            model = deep_nn_models.Unet("", ct_kernel=ct_kernel,
                                        n_blocks=n_blocks, filters=filters,
                                        train_patches=False, weighted_loss=False).build_model(input_shape, dg_train_weight_target=None)
        elif architecture == "cnn":
            model = deep_nn_models.CNN(input_shape = (input_shape[0], input_shape[1], 1))
        elif architecture == "mlp":
            model = deep_nn_models.MLP(input_shape = (input_shape[0], input_shape[1]))
            
        model = deep_nn_models.Unet("",train_patches=False, weighted_loss=False).build_model(input_shape, dg_train_weight_target=None)
        optim = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=['accuracy'])
        # Train the model
        checkpoint = ModelCheckpoint('models/'+ (dir or '') + f'{modname}_{obs}/{week}/best_model_stacked_{architecture}_{i}.keras', save_best_only=True, save_weights_only=False, monitor='val_loss',
                                        mode='min', verbose=0)

            
        if training_type == "tune":
            #define hyperparameter grid
            batch_sizes = tuning_grid['batch_sizes']
            learning_rates = tuning_grid['learning_rates']
            ct_kernels = tuning_grid['ct_kernels']
            n_filters = tuning_grid['n_filters']
            n_blocks = tuning_grid['n_blocks']

            patience = tuning_grid['patience']

            best_val_loss = float("inf")
            best_model_path = None
            best_params = None

            # Iterate over all combinations of hyperparameters
            for trial_num, (bs, lr, ct_kernel, n_filter, n_block) in enumerate(itertools.product(batch_sizes, learning_rates, ct_kernels, n_filters, n_blocks)):
                print(f'Trial {trial_num+1}/ {len(batch_sizes)*len(learning_rates)*len(ct_kernels)*len(n_filters)*len(n_blocks)}')
                print(f'Tuning Combination: Batch size={bs}, LR={lr}, Kernel={ct_kernel}, Filters={n_filter}, Blocks={n_block}')

                model = deep_nn_models.Unet("", ct_kernel=ct_kernel,
                                n_blocks=n_block, filters=n_filter,
                            train_patches=False, weighted_loss=False).build_model(input_shape, dg_train_weight_target=None)
                optim = optimizers.Adam(learning_rate=lr)
                model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=['accuracy'])

                checkpoint_path = 'models/'+ (dir or '') + f'{modname}_{obs}/{week}/best_model_stacked_{architecture}_bootstrap_{i+1}_trial_{trial_num+1}.keras'
                checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min', verbose=0)

                early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

                history = model.fit(x=X_train.values, y=Y_train_oh, validation_data=(X_val.values, Y_val_oh), epochs=epochs, batch_size=bs,
                                    callbacks=[checkpoint, early_stopping], shuffle=True, verbose=0)

                # Track the best model based on validation loss
                val_loss = min(history.history['val_loss'])
                print(f'Validation loss for bootstrap {i+1}, trial {trial_num+1}: {val_loss}')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = checkpoint_path
                    best_params = (bs, lr, ct_kernel, n_filter, n_block)

            # Load the best model
            best_model = models.load_model(best_model_path)
            best_params_dict[i] = {'batch_size': best_params[0], 'lr': best_params[1], 'ct_kernel': best_params[2], 'filters': best_params[3], 'blocks': best_params[4], 'val_loss': best_val_loss}
            print(f'Best hyperparameters for bootstrap {i+1}: {best_params_dict[i]}')

        elif training_type == "train":
            history = model.fit(x=X_train.values, y=Y_train_oh, validation_data=(X_val.values, Y_val_oh), epochs=epochs, batch_size=batch_size,
                                    callbacks=[checkpoint],
                                shuffle=True, verbose=0)


            best_model = models.load_model('models/'+ (dir or '') + f'{modname}_{obs}/{week}/best_model_{architecture}_{i}.keras')

        predictions = best_model.predict(X_test.values, verbose=0)
        train_predictions = best_model.predict(X_train.values, verbose=0)
        val_predictions = best_model.predict(X_val.values, verbose=0)

        dims = ('MT', 'Y', 'X', 'category')
        test_predictions_xarray = xr.DataArray(predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})
        train_predictions_xarray = xr.DataArray(train_predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})
        val_predictions_xarray = xr.DataArray(val_predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})
        Y_test_oh_xr = xr.DataArray(Y_test_oh, dims=dims, coords={'category': ['below', 'normal', 'above']})



    return train_predictions_xarray, val_predictions_xarray, test_predictions_xarray, Y_test_oh_xr, X_train, X_val, X_test, ytrain_terciled, yval_terciled, ytest_terciled


def train_deepnet(xtrain_list, ytrain_list, xval_list, yval_list, xtest_list, ytest_list,
                           architecture_params=None, tuning_grid=None, architecture="unet",
                           training_type="train", predictor="mean", modname = "GEFS", obs="IMD", week="wk3-4",
                           epochs=100, batch_size=16, learning_rate=1e-3,dir=None):
    rpss_test_list = []
    rpss_train_list = []
    rpss_val_list = []

    predictions_list = []
    y_test_oh_list = []

    n_bootstraps = len(xtrain_list)
    for i in range(n_bootstraps):
        print(f'Bootstrap {i+1}')
        train_predictions_xarray, val_predictions_xarray, test_predictions_xarray, Y_test_oh_xr, X_train, X_val, X_test, ytrain_terciled, yval_terciled, ytest_terciled = train_single_bootstrap_deepnet(i, xtrain_list, ytrain_list, 
                                                                                                                    xval_list, yval_list, 
                                                                                                                    xtest_list, ytest_list,
                                                                                architecture_params, tuning_grid, architecture, training_type,
                                                                                predictor, modname, obs, week, epochs, batch_size, learning_rate,dir)
        
        predictions_list.append(test_predictions_xarray)
        y_test_oh_list.append(Y_test_oh_xr)

        if predictor == "mean":
            fcast_train = performance_metrics.climo_predict(xtrain_list[i], predictor)
            fcast_val = performance_metrics.climo_predict(xval_list[i], predictor)
            fcast_test= performance_metrics.climo_predict(xtest_list[i], predictor)
        elif predictor == "stacked":
            fcast_train = performance_metrics.climo_predict(X_train, predictor)
            fcast_val = performance_metrics.climo_predict(X_val, predictor)
            fcast_test= performance_metrics.climo_predict(X_test, predictor)

        rpss_train = performance_metrics.rpss(fcast_train, train_predictions_xarray, ytrain_terciled,predictor)
        rpss_val = performance_metrics.rpss(fcast_val, val_predictions_xarray, yval_terciled,predictor)
        rpss_test = performance_metrics.rpss(fcast_test, test_predictions_xarray, ytest_terciled,predictor)

        rpss_train_list.append(rpss_train)
        rpss_val_list.append(rpss_val)
        rpss_test_list.append(rpss_test)
        


    return rpss_train_list, rpss_val_list, rpss_test_list, predictions_list, y_test_oh_list


def train_deepnet_parallel(xtrain_list, ytrain_list, xval_list, yval_list, xtest_list, ytest_list,
                           architecture_params=None, tuning_grid=None, architecture="unet",
                           training_type="train", predictor="mean", modname = "GEFS", obs="IMD", week="wk3-4",
                           epochs=100, batch_size=16, learning_rate=1e-3, n_jobs=-1):
    n_bootstraps = len(xtrain_list)
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_single_bootstrap)(i, xtrain_list, ytrain_list, xval_list, yval_list, xtest_list, ytest_list,
                                        architecture_params, tuning_grid, architecture, training_type,
                                        predictor, modname, obs, week, epochs, batch_size, learning_rate)
        for i in range(n_bootstraps)
    )
    rpss_test_list, rpss_train_list, rpss_val_list, predictions_list, y_test_oh_list = zip(*results)
    return rpss_test_list, rpss_train_list, rpss_val_list, predictions_list, y_test_oh_list


def train_deepnet_mme(xtrain_dict, ytrain_list, xval_dict, yval_list, xtest_dict, ytest_list,
                           architecture_params=None, tuning_grid=None, architecture="unet",
                           training_type="train", predictor="mean", obs="IMD", week="wk3-4",
                           epochs=100, batch_size=16, learning_rate=1e-3,dir=None):
    rpss_test_list = []
    rpss_train_list = []
    rpss_val_list = []

    predictions_list = []
    y_test_oh_list = []

    n_bootstraps = len(ytrain_list)
    for i in range(n_bootstraps):
        print(f'### Bootstrap {i+1} ####')
        train_preds_list = []
        val_preds_list = []
        test_preds_list = []
        for name, model in xtrain_dict.items():
            print(f'----- Model {name}')
            xtrain_list = xtrain_dict[name]
            xval_list = xval_dict[name]
            xtest_list = xtest_dict[name]
            train_predictions_xarray, val_predictions_xarray, test_predictions_xarray, Y_test_oh_xr, X_train, X_val, X_test, ytrain_terciled, yval_terciled, ytest_terciled = train_single_bootstrap_deepnet(i=i, xtrain_list=xtrain_list,
                                                                                                                         ytrain_list = ytrain_list,
                                                                                                                    xval_list=xval_list, yval_list=yval_list,
                                                                                                                    xtest_list= xtest_list, ytest_list=ytest_list,
                                                                                                                    architecture_params=architecture_params, 
                                                                                                                tuning_grid=tuning_grid,
                                                                                                                  architecture=architecture, 
                                                                                                                  training_type=training_type,
                                                                                                                    modname=name, predictor=predictor, obs=obs,
                                                                                                                      week=week, epochs=epochs,
                                                                                                                        batch_size=batch_size, 
                                                                                                                        learning_rate=learning_rate,
                                                                                                                        dir=dir)
            train_preds_list.append(train_predictions_xarray)
            val_preds_list.append(val_predictions_xarray)
            test_preds_list.append(test_predictions_xarray)

        train_preds = xr.concat(train_preds_list, dim='model').mean('model')
        val_preds = xr.concat(val_preds_list, dim='model').mean('model')
        test_preds = xr.concat(test_preds_list, dim='model').mean('model')
        #normalize probabilites
        train_preds = train_preds / train_preds.sum('category')
        val_preds = val_preds / val_preds.sum('category')
        test_preds = test_preds / test_preds.sum('category')

        predictions_list.append(test_preds)
        y_test_oh_list.append(Y_test_oh_xr)


        if predictor == "mean":
            fcast_train = performance_metrics.climo_predict(xtrain_list[i], predictor)
            fcast_val = performance_metrics.climo_predict(xval_list[i], predictor)
            fcast_test= performance_metrics.climo_predict(xtest_list[i], predictor)
        elif predictor == "stacked":
            fcast_train = performance_metrics.climo_predict(X_train, predictor)
            fcast_val = performance_metrics.climo_predict(X_val, predictor)
            fcast_test= performance_metrics.climo_predict(X_test, predictor)

        rpss_train = performance_metrics.rpss(fcast_train, train_preds, ytrain_terciled,predictor)
        rpss_val = performance_metrics.rpss(fcast_val, val_preds, yval_terciled,predictor)
        rpss_test = performance_metrics.rpss(fcast_test, test_preds, ytest_terciled,predictor)

        rpss_train_list.append(rpss_train)
        rpss_val_list.append(rpss_val)
        rpss_test_list.append(rpss_test)
        


    return rpss_train_list, rpss_val_list, rpss_test_list, predictions_list, y_test_oh_list

def terciled_to_ohe_xr(y):
    """Converts terciled predictand to one-hot encoded xarray for ELR

    Args:
        y (_type_): terciled predictand xarray
    """
    # Create a mask for non-NaN values
    non_nan_mask = ~np.isnan(y.values)
    # Placeholder for the one-hot encoded array with NaN where needed
    y_oh = np.full((*y.shape, 3), np.nan)  # Add category dimension
    # Apply one-hot encoding only to non-NaN values
    valid_values = y.values[non_nan_mask].astype(int)  # Filter valid values
    one_hot_encoded = tf.keras.utils.to_categorical(valid_values, num_classes=3)  # Encode
    # Insert the one-hot encoded values back into the full array
    y_oh[non_nan_mask] = one_hot_encoded
    # Convert back to an xarray DataArray with a new "category" dimension
    y_oh_xr = xr.DataArray(
        y_oh,
        dims=(*y.dims, "category"),
        coords={**y.coords, "category": ["below", "normal", "above"]}
    )
    return y_oh_xr



def train_single_bootstrap_ELR(xtrain, ytrain, xtest, ytest):
    train_predictions_storage = np.full((len(xtrain['T']), len(xtrain.Y), len(xtrain.X), 3), np.nan)
    test_predictions_storage = np.full((len(xtest['T']), len(xtest.Y), len(xtest.X), 3), np.nan)

    labeler = preprocessing.rolling_labeler_ELR(ytrain, window=1)
    y_train_terciled, edges_train, y_elr_train = labeler(ytrain)
    y_test_terciled, edges_test, y_elr_test = labeler(ytest)
    #flatten
    y_elr_train = y_elr_train.stack(QT=('quantile', 'T')).transpose('QT', 'Y', 'X')
    y_elr_test = y_elr_test.stack(QT=('quantile', 'T')).transpose('QT', 'Y', 'X')

    edges_train_qt = edges_train.stack(QT=('quantile', 'T')).transpose('QT', 'Y', 'X')
    edges_test_qt = edges_test.stack(QT=('quantile', 'T')).transpose('QT', 'Y', 'X')
    xtrain_m = xtrain.mean('M')
    xtrain_m = np.tile(xtrain_m, reps=(2,1,1))
    xtrain_m = xr.DataArray(xtrain_m, dims=('QT', 'Y', 'X'), coords={
                                                                    'Y': xtrain.Y.values,
                                                                    'X': xtrain.X.values})

    xtest_m = xtest.mean('M')
    xtest_m = np.tile(xtest_m, reps=(2,1,1))
    xtest_m = xr.DataArray(xtest_m, dims=('QT', 'Y', 'X'), coords={
                                                                    'Y': xtest.Y.values,
                                                                    'X': xtest.X.values})

    X_ranges = xtrain.X.values
    Y_ranges = xtrain.Y.values

    for i in range (len(Y_ranges)):
        for j in range (len(X_ranges)):

            ytrain_grid_raw = ytrain.isel(X=j, Y=i).values

            if np.isnan(ytrain_grid_raw).any():
                continue 
            

            ## train grids
            xtrain_grid = xtrain_m.sel(X=X_ranges[j], Y=Y_ranges[i]).values
            edges_train_qt_grid = edges_train_qt.sel(X=X_ranges[j], Y=Y_ranges[i]).values
            total_elements = edges_train_qt_grid.shape[0]
            midpoint = total_elements // 2
            
            edges_train_qt_grid[:midpoint] = 33
            edges_train_qt_grid[midpoint:] = 67

            Xtrain_qt_grid = np.stack([xtrain_grid, edges_train_qt_grid], axis=1) 
            ytrain_grid = y_elr_train.sel(X=X_ranges[j], Y=Y_ranges[i]).values

            ## test grids

            xtest_grid = xtest_m.sel(X=X_ranges[j], Y=Y_ranges[i]).values
            edges_test_qt_grid = edges_test_qt.sel(X=X_ranges[j], Y=Y_ranges[i]).values
            total_elements = edges_test_qt_grid.shape[0]
            midpoint = total_elements // 2


            edges_test_qt_grid[:midpoint] = 33
            edges_test_qt_grid[midpoint:] = 67
        
            Xtest_qt_grid = np.stack([xtest_grid, edges_test_qt_grid], axis=1)
            ytest_grid = y_elr_test.sel(X=X_ranges[j], Y=Y_ranges[i]).values
        
            if np.isnan(ytrain_grid).all():
                continue

            #drop nas from xtrain_grid and ytrain_grid
            valid_indices_train = ~np.isnan(ytrain_grid)
            valid_indices_test = ~np.isnan(ytest_grid)

            Xtrain_qt_grid = Xtrain_qt_grid[valid_indices_train]
            ytrain_grid = ytrain_grid[valid_indices_train]
            Xtest_qt_grid = Xtest_qt_grid[valid_indices_test]
            ytest_grid = ytest_grid[valid_indices_test]

            if np.isnan(Xtrain_qt_grid).any() or np.isnan(Xtest_qt_grid).any():
                continue
            
            if len(ytrain_grid) <= 2 or len(ytest_grid) <= 2:
                print(f'Not enough data for grid {i}, {j}')
                continue

            valid_indices_train_half = valid_indices_train[:len(valid_indices_train)//2]
            valid_indices_test_half = valid_indices_test[:len(valid_indices_test)//2]

            # Fit the logistic regression model
  
            model = sm.GLM(ytrain_grid, sm.add_constant(Xtrain_qt_grid,has_constant='add'),
                                family=sm.families.Binomial())
            model = model.fit()

            try:
                train_predictions = model.predict(exog=sm.add_constant(Xtrain_qt_grid))
            except:
                print(Xtrain_qt_grid)
                print(f'Error for grid ', i, j)
                print('Xtrain_qt_grid shape:', Xtrain_qt_grid.shape)
                print(model.summary())
                print('ytrain_grid shape:', ytrain_grid.shape)

            #split into two (odd and even indices)
            train_probas1 = train_predictions[:len(train_predictions)//2]
            train_probas2 = train_predictions[len(train_predictions)//2:]

            train_predictions_storage[valid_indices_train_half, i, j, 0] = train_probas1
            train_predictions_storage[valid_indices_train_half, i, j, 1] = train_probas2 - train_probas1
            train_predictions_storage[valid_indices_train_half, i, j, 2] = 1 - train_probas2
            #fill rest of indices with 1/3 for each class
            train_predictions_storage[:, i, j, :] = np.nan_to_num(train_predictions_storage[:, i, j, :], nan=1/3)

            try:
                test_predictions = model.predict(exog = sm.add_constant(Xtest_qt_grid,has_constant='add'))
            except:
                print(f'Error for grid ', i, j)
                continue

            test_probas1 = test_predictions[:len(test_predictions)//2]
            test_probas2 = test_predictions[len(test_predictions)//2:]

            test_predictions_storage[valid_indices_test_half, i, j, 0] = test_probas1
            test_predictions_storage[valid_indices_test_half, i, j, 1] = test_probas2 - test_probas1
            test_predictions_storage[valid_indices_test_half, i, j, 2] = 1 - test_probas2
            test_predictions_storage[:, i, j, :] = np.nan_to_num(test_predictions_storage[:, i, j, :], nan=1/3)   

    dims = ('T', 'Y', 'X', 'category')
    train_predictions_xarray = xr.DataArray(train_predictions_storage, dims=dims, coords={'category': ['below', 'normal', 'above']})
    test_predictions_xarray = xr.DataArray(test_predictions_storage, dims=dims, coords={'category': ['below', 'normal', 'above']})

    return train_predictions_xarray, test_predictions_xarray, y_train_terciled, y_test_terciled


def train_elr(xtrain_list_elr, ytrain_list_elr, xtest_list_elr, ytest_list_elr):
    """trains ELR model on bootstrapped data

    Args:
        xtrain_list_elr (_type_): list of xarray DataArrays of predictor training data
        ytrain_list_elr (_type_): list of xarray DataArrays of predictand training data
        xtest_list_elr (_type_): list of xarray DataArrays of predictor testing data
        ytest_list_elr (_type_): list of xarray DataArrays of predictand testing data

    Returns:
        _type_: rpss lists of training, and testing data, test predictions and one-hot encoded testing predictand
    """
    rpss_test_list = []
    rpss_train_list = []

    predictions_list = []
    y_test_oh_list = []

    n_bootstraps = len(xtrain_list_elr)

    for i in range(n_bootstraps):

        print(f'Bootstrap {i+1}')
        xtrain = xtrain_list_elr[i]
        ytrain = ytrain_list_elr[i]

        xtest = xtest_list_elr[i]
        ytest = ytest_list_elr[i]

        train_predictions_xarray, test_predictions_xarray, y_train_terciled, y_test_terciled = train_single_bootstrap_ELR(xtrain, ytrain, xtest, ytest)

        predictions_list.append(test_predictions_xarray)

        y_test_oh_xr = terciled_to_ohe_xr(y_test_terciled)
        y_test_oh_list.append(y_test_oh_xr)


        fcast_test = performance_metrics.climo_predict(xtest)
        fcast_train = performance_metrics.climo_predict(xtrain)

        rpss_train = performance_metrics.rpss(fcast_train, train_predictions_xarray, y_train_terciled)
        rpss_test = performance_metrics.rpss(fcast_test, test_predictions_xarray, y_test_terciled)

        rpss_test_list.append(rpss_test)
        rpss_train_list.append(rpss_train)

    return rpss_train_list, rpss_test_list, predictions_list, y_test_oh_list



def train_elr_mme(xtrain_dict_elr, ytrain_list_elr, xtest_dict_elr, ytest_list_elr):
    """trains ELR model on bootstrapped data

    Args:
        xtrain_dict_elr (_type_): dict of list of xarray DataArrays of predictor training data
        ytrain_list_elr (_type_): list of xarray DataArrays of predictand training data
        xtest_dict_elr (_type_): dict of list of xarray DataArrays of predictor testing data
        ytest_list_elr (_type_): list of xarray DataArrays of predictand testing data

    Returns:
        _type_: rpss lists of training, and testing data, test predictions and one-hot encoded testing predictand
    """
    rpss_test_list = []
    rpss_train_list = []

    predictions_list = []
    y_test_oh_list = []

    n_bootstraps = len(ytrain_list_elr)

    for i in range(n_bootstraps):
        print(f'### Bootstrap {i+1} ###')
        train_preds_list = []
        test_preds_list = []
        for name, model in xtrain_dict_elr.items():
            print(f'----- Model {name}')
            xtrain_list = model
            xtest_list = xtest_dict_elr[name]

            xtrain = xtrain_list[i]
            ytrain = ytrain_list_elr[i]
            xtest = xtest_list[i]
            ytest = ytest_list_elr[i]
            train_predictions_xarray, test_predictions_xarray, y_train_terciled, y_test_terciled = train_single_bootstrap_ELR(xtrain, ytrain, xtest, ytest)

            train_preds_list.append(train_predictions_xarray)
            test_preds_list.append(test_predictions_xarray)


        train_preds = xr.concat(train_preds_list, dim='model').mean('model')
        test_preds = xr.concat(test_preds_list, dim='model').mean('model')
        #normalize probabilites
        train_preds = train_preds / train_preds.sum('category')
        test_preds = test_preds / test_preds.sum('category')
            


        predictions_list.append(test_preds)

        y_test_oh_xr = terciled_to_ohe_xr(y_test_terciled)
        y_test_oh_list.append(y_test_oh_xr)


        fcast_test = performance_metrics.climo_predict(xtest)
        fcast_train = performance_metrics.climo_predict(xtrain)

        rpss_train = performance_metrics.rpss(fcast_train, train_preds, y_train_terciled)
        rpss_test = performance_metrics.rpss(fcast_test, test_preds, y_test_terciled)

        rpss_test_list.append(rpss_test)
        rpss_train_list.append(rpss_train)

    return rpss_train_list, rpss_test_list, predictions_list, y_test_oh_list
































# def train_deepnet(xtrain_list, ytrain_list, xval_list,yval_list, xtest_list, ytest_list, 
#                   architecture_params=None,
#                   tuning_grid = None,
#                   architecture="unet",
#                   training_type = "train",
#                   predictor="mean",modname="GEFS",
#                   obs="IMD", week="wk3-4",epochs=100,batch_size=16,learning_rate=1e-3):
#         """trains deep learning model on bootstrapped data

#         Args:
#             xtrain_list (_type_): list of xarray DataArrays of predictor training data
#             ytrain_list (_type_): list of xarray DataArrays of predictand training data
#             xval_list (_type_): list of xarray DataArrays of predictor validation data
#             yval_list (_type_): list of xarray DataArrays of predictand validation data
#             xtest_list (_type_): list of xarray DataArrays of predictor testing data
#             ytest_list (_type_): list of xarray DataArrays of predictand testing data
#             architecture_params (_type_, optional): dictionary of architecture parameters. Defaults to None.
#             tuning_grid (_type_, optional): dictionary of hyperparameters for tuning. Defaults to None.
#             architecture (str, optional): architecture of the model. Defaults to "unet".
#             training_type (str, optional): type of training. Defaults to "train".
#             predictor (str, optional): type of predictor. Defaults to "mean".
#             modname (str, optional): model name. Defaults to "GEFS".
#             obs (str, optional): observation. Defaults to "IMD".
#             week (str, optional): forecast week. Defaults to "wk3-4".
#             epochs (int, optional): number of epochs. Defaults to 100.
#             batch_size (int, optional): batch size. Defaults to 16.
#             learning_rate (float, optional): learning rate. Defaults to 1e-3.
        
#         Returns:
#             _type_: rpss lists of training, validation, and testing data, test predictions and one-hot encoded testing predictand
#         """
#         rpss_test_list = []
#         rpss_train_list = []
#         rpss_val_list = []

#         predictions_list = []
#         y_test_oh_list = []

#         best_params_dict ={}

#         n_bootstraps = len(xtrain_list)


#         if predictor == "mean":

#             for i in range(n_bootstraps):
#                 print(f'Bootstrap {i+1}')
#                 xtrain = xtrain_list[i]
#                 ytrain = ytrain_list[i]
#                 xval = xval_list[i]
#                 yval = yval_list[i]
#                 xtest = xtest_list[i]
#                 ytest = ytest_list[i]

#                 X_train, Y_train_oh, X_val, Y_val_oh, X_test, Y_test_oh, y_train_terciled, y_val_terciled, y = preprocessing.preprocess(xtrain, ytrain,
#                                                                                                                                                     xval, yval,
#                                                                                                                                                         xtest, ytest)
#                 input_shape = (X_train.shape[1], X_train.shape[2], 1)
                
#                 if architecture == "unet":
#                     if architecture_params is not None:
#                         ct_kernel = architecture_params['ct_kernel']
#                         n_blocks = architecture_params['n_blocks']
#                         filters = architecture_params['filters']
#                     model = deep_nn_models.Unet("", ct_kernel=ct_kernel,
#                                                 n_blocks=n_blocks, filters=filters,
#                                                 train_patches=False, weighted_loss=False).build_model(input_shape, dg_train_weight_target=None)
#                 elif architecture == "cnn":
#                     model = deep_nn_models.CNN(input_shape = (input_shape[0], input_shape[1], 1))
#                 elif architecture == "mlp":
#                     model = deep_nn_models.MLP(input_shape = (input_shape[0], input_shape[1]))

#                 optim = optimizers.Adam(learning_rate=learning_rate)
#                 model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=['accuracy'])
#                 # Train the model
#                 checkpoint = ModelCheckpoint(f'models/{modname}_{obs}/{week}/best_model_{architecture}_{i}.keras', save_best_only=True, save_weights_only=False, monitor='val_loss',
#                                               mode='min', verbose=0)
                    
#                 if training_type == "tune":

#                     #define hyperparameter grid
#                     batch_sizes = tuning_grid['batch_sizes']
#                     learning_rates = tuning_grid['learning_rates']
#                     ct_kernels = tuning_grid['ct_kernels']
#                     n_filters = tuning_grid['n_filters']
#                     n_blocks = tuning_grid['n_blocks']
#                     patience = tuning_grid['patience']
   
#                     best_val_loss = float("inf")
#                     best_model_path = None
#                     best_params = None

#                     # Iterate over all combinations of hyperparameters
#                     for trial_num, (bs, lr, ct_kernel, n_filter, n_block) in enumerate(itertools.product(batch_sizes, learning_rates, ct_kernels, n_filters, n_blocks)):
#                         print(f'Trial {trial_num+1}')
#                         print(f'Tuning Combination: Batch size={bs}, LR={lr}, Kernel={ct_kernel}, Filters={n_filter}, Blocks={n_block}')

#                         model = deep_nn_models.Unet("", ct_kernel=ct_kernel,
#                                         n_blocks=n_block, filters=n_filter,
#                                     train_patches=False, weighted_loss=False).build_model(input_shape, dg_train_weight_target=None)
                                                    
#                         optim = optimizers.Adam(learning_rate=lr)
#                         model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=['accuracy'])

#                         checkpoint_path = f'models/{modname}_{obs}/{week}/best_model_{architecture}_bootstrap_{i+1}_trial_{trial_num+1}.keras'
#                         checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min', verbose=0)
#                         early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

#                         history = model.fit(x=X_train, y=Y_train_oh, validation_data=(X_val, Y_val_oh), epochs=epochs, batch_size=batch_size,
#                                             callbacks=[checkpoint, early_stopping], shuffle=True, verbose=0)

#                         # Track the best model based on validation loss
#                         val_loss = min(history.history['val_loss'])
#                         print(f'Validation loss for bootstrap {i+1}, trial {trial_num+1}: {val_loss}')
#                         if val_loss < best_val_loss:
#                             best_val_loss = val_loss
#                             best_model_path = checkpoint_path
#                             best_params = (bs, lr, ct_kernel, n_filter, n_block)

#                     # Load the best model
#                     best_model = models.load_model(best_model_path)
#                     best_model.save(f'models/{modname}_{obs}/{week}/best_model_{architecture}_{i}_tuned.keras')
#                     best_params_dict[i] = {'batch_size': best_params[0], 'lr': best_params[1], 'ct_kernel': best_params[2], 'filters': best_params[3], 'blocks': best_params[4], 'val_loss': best_val_loss}
#                     print(f'Best hyperparameters for bootstrap {i+1}: {best_params_dict[i]}')

#                 elif training_type == "train":
#                     history = model.fit(x=X_train, y=Y_train_oh, validation_data=(X_val, Y_val_oh), epochs=epochs, batch_size=batch_size, 
#                                         callbacks=[checkpoint],
#                                         shuffle=True, verbose=0)


#                     best_model = models.load_model(f'models/{modname}_{obs}/{week}/best_model_{architecture}_{i}.keras')

#                 elif training_type == "load":
#                     try:
#                         best_model = models.load_model(f'models/{modname}_{obs}/{week}/best_model_{architecture}_{i}_tuned.keras')
#                     except:
#                         best_model = models.load_model(f'models/{modname}_{obs}/{week}/best_model_{architecture}_{i}.keras')

#                 predictions = best_model.predict(X_test, verbose=0)
#                 train_predictions = best_model.predict(X_train, verbose=0)
#                 val_predictions = best_model.predict(X_val, verbose=0)

#                 dims = ('T', 'Y', 'X', 'category')
#                 predictions_xarray = xr.DataArray(predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})
#                 train_predictions_xarray = xr.DataArray(train_predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})
#                 val_predictions_xarray = xr.DataArray(val_predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})

#                 predictions_list.append(predictions_xarray)
#                 Y_test_oh_xr = xr.DataArray(Y_test_oh, dims=dims, coords={'category': ['below', 'normal', 'above']})
#                 y_test_oh_list.append(Y_test_oh_xr)

#                 fcast_test= performance_metrics.climo_predict(xtest)
#                 fcast_train = performance_metrics.climo_predict(xtrain)
#                 fcast_val = performance_metrics.climo_predict(xval)

#                 rpss_test = performance_metrics.rpss(fcast_test, predictions_xarray, y)
#                 rpss_train = performance_metrics.rpss(fcast_train, train_predictions_xarray, y_train_terciled)
#                 rpss_val = performance_metrics.rpss(fcast_val, val_predictions_xarray, y_val_terciled)

#                 rpss_test_list.append(rpss_test)
#                 rpss_train_list.append(rpss_train)
#                 rpss_val_list.append(rpss_val)

#         if predictor == "stacked":

#             for i in range(n_bootstraps):

#                 print(f'Bootstrap {i+1}')
#                 xtrain = xtrain_list[i]
#                 ytrain = ytrain_list[i]
#                 xval = xval_list[i]
#                 yval = yval_list[i]
#                 xtest = xtest_list[i]
#                 ytest = ytest_list[i]

#                 X_train, Y_train_oh, X_val, Y_val_oh, X_test, Y_test_oh, y_train_terciled, y_val_terciled, y = preprocessing.preprocess_stacked(xtrain, ytrain,
#                                                                                                                                                                xval, yval,
#                                                                                                                                                                  xtest, ytest)
#                 input_shape = (X_train.shape[1], X_train.shape[2])


#                 if architecture == "unet":
#                     if architecture_params is not None:
#                         ct_kernel = architecture_params['ct_kernel']
#                         n_blocks = architecture_params['n_blocks']
#                         filters = architecture_params['filters']
#                     model = deep_nn_models.Unet("", ct_kernel=ct_kernel,
#                                                 n_blocks=n_blocks, filters=filters,
#                                                 train_patches=False, weighted_loss=False).build_model(input_shape, dg_train_weight_target=None)
#                 elif architecture == "cnn":
#                     model = deep_nn_models.CNN(input_shape = (input_shape[0], input_shape[1], 1))
#                 elif architecture == "mlp":
#                     model = deep_nn_models.MLP(input_shape = (input_shape[0], input_shape[1]))
                    
#                 model = deep_nn_models.Unet("",train_patches=False, weighted_loss=False).build_model(input_shape, dg_train_weight_target=None)
#                 optim = optimizers.Adam(learning_rate=learning_rate)
#                 model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=['accuracy'])
#                 # Train the model
#                 checkpoint = ModelCheckpoint(f'models/{modname}_{obs}/{week}/best_model_stacked_{architecture}_{i}.keras', save_best_only=True, save_weights_only=False, monitor='val_loss',
#                                               mode='min', verbose=0)

                    
#                 if training_type == "tune":
#                     #define hyperparameter grid
#                     batch_sizes = tuning_grid['batch_sizes']
#                     learning_rates = tuning_grid['learning_rates']
#                     ct_kernels = tuning_grid['ct_kernels']
#                     n_filters = tuning_grid['n_filters']
#                     n_blocks = tuning_grid['n_blocks']

#                     patience = tuning_grid['patience']
   
#                     best_val_loss = float("inf")
#                     best_model_path = None
#                     best_params = None

#                     # Iterate over all combinations of hyperparameters
#                     for trial_num, (bs, lr, ct_kernel, n_filter, n_block) in enumerate(itertools.product(batch_sizes, learning_rates, ct_kernels, n_filters, n_blocks)):
#                         print(f'Trial {trial_num+1}')
#                         print(f'Tuning Combination: Batch size={bs}, LR={lr}, Kernel={ct_kernel}, Filters={n_filter}, Blocks={n_block}')

#                         model = deep_nn_models.Unet("", ct_kernel=ct_kernel,
#                                         n_blocks=n_block, filters=n_filter,
#                                     train_patches=False, weighted_loss=False).build_model(input_shape, dg_train_weight_target=None)
#                         optim = optimizers.Adam(learning_rate=lr)
#                         model.compile(optimizer=optim, loss="categorical_crossentropy", metrics=['accuracy'])

#                         checkpoint_path = f'models/{modname}_{obs}/{week}/best_model_stacked_{architecture}_bootstrap_{i+1}_trial_{trial_num+1}.keras'
#                         checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=False, monitor='val_loss', mode='min', verbose=0)

#                         early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

#                         history = model.fit(x=X_train, y=Y_train_oh, validation_data=(X_val, Y_val_oh), epochs=epochs, batch_size=bs,
#                                             callbacks=[checkpoint, early_stopping], shuffle=True, verbose=0)

#                         # Track the best model based on validation loss
#                         val_loss = min(history.history['val_loss'])
#                         print(f'Validation loss for bootstrap {i+1}, trial {trial_num+1}: {val_loss}')
#                         if val_loss < best_val_loss:
#                             best_val_loss = val_loss
#                             best_model_path = checkpoint_path
#                             best_params = (bs, lr, ct_kernel, n_filter, n_block)

#                     # Load the best model
#                     best_model = models.load_model(best_model_path)
#                     best_params_dict[i] = {'batch_size': best_params[0], 'lr': best_params[1], 'ct_kernel': best_params[2], 'filters': best_params[3], 'blocks': best_params[4], 'val_loss': best_val_loss}
#                     print(f'Best hyperparameters for bootstrap {i+1}: {best_params_dict[i]}')

#                 elif training_type == "train":
#                     history = model.fit(x=X_train.values, y=Y_train_oh, validation_data=(X_val.values, Y_val_oh), epochs=epochs, batch_size=batch_size,
#                                          callbacks=[checkpoint],
#                                         shuffle=True, verbose=0)


#                     best_model = models.load_model(f'models/{modname}_{obs}/{week}/best_model_{architecture}_{i}.keras')

#                 predictions = best_model.predict(X_test.values, verbose=0)
#                 train_predictions = best_model.predict(X_train.values, verbose=0)
#                 val_predictions = best_model.predict(X_val.values, verbose=0)

#                 dims = ('MT', 'Y', 'X', 'category')
#                 predictions_xarray = xr.DataArray(predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})
#                 train_predictions_xarray = xr.DataArray(train_predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})
#                 val_predictions_xarray = xr.DataArray(val_predictions, dims=dims, coords={'category': ['below', 'normal', 'above']})

#                 predictions_list.append(predictions_xarray)
#                 Y_test_oh_xr = xr.DataArray(Y_test_oh, dims=dims, coords={'category': ['below', 'normal', 'above']})
#                 y_test_oh_list.append(Y_test_oh_xr)

#                 fcast_test= performance_metrics.climo_predict_stacked(X_test)
#                 fcast_train = performance_metrics.climo_predict_stacked(X_train)
#                 fcast_val = performance_metrics.climo_predict_stacked(X_val)

#                 rpss_test = performance_metrics.rpss_stacked(fcast_test, predictions_xarray, y)
#                 rpss_train = performance_metrics.rpss_stacked(fcast_train, train_predictions_xarray, y_train_terciled)
#                 rpss_val = performance_metrics.rpss_stacked(fcast_val, val_predictions_xarray, y_val_terciled)

#                 rpss_test_list.append(rpss_test)
#                 rpss_train_list.append(rpss_train)
#                 rpss_val_list.append(rpss_val)



#         return rpss_train_list, rpss_val_list, rpss_test_list, predictions_list, y_test_oh_list
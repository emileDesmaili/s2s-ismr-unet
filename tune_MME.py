
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

import utils.dataloader as dataloader
import utils.plots as plots
import utils.training as training
import utils.preprocessing as preprocessing
import os
import numpy as np
import xarray as xr
import pandas as pd
import time
import tensorflow as tf
import shutil

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
#check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



def main():


    dir = "MME/"
    obs = "IMD"
    models = ["GEFS","IITM", "ECMWF"] # can be GEFS IITM ECMWF for multi-model training, put a list of strings
    domain = [67, 98, 7, 38] # west east south north. for Unet's check that lat and lot make a square divisible by 8, ie 24x24, 32x32, 64x64
    season = "May-Sep"
    n_bootstraps = 10
    years = (2003, 2018)
    week = "wk3-4"  #wk1, wk2 or wk3-4


    print(f' #######  TUNING MME for {obs} for {week} lead time  ##########')

    download = True


    for model in models:
        os.makedirs('models/' + (dir or '') + f'{model}_{obs}', exist_ok=True)
        if download:
            os.makedirs(f'download/{model}_{obs}', exist_ok=True)
    os.makedirs('figures/' + (dir or '') + f'MME_{obs}', exist_ok=True)
    os.makedirs('outputs/' + (dir or '') + f'MME_{obs}', exist_ok=True)

    custom_leads = {'GEFS':(16,29), 'IITM':(16,29), 'ECMWF':(16,30)} # custom lead times for each model
    custom_seasons = {'GEFS':"May-Sep", 'IITM':"May-Sep", 'ECMWF':"May-Sep"} # custom season for each model


    x, y = dataloader.get_data_ensemble(years=years, download = download,week=week,obs=obs, domain=domain, season=season, 
                            models=models, custom_leads=custom_leads, custom_seasons=custom_seasons)
    
    # x['GEFS'] = x['GEFS'].isel(T=slice(0, -1))  
    # x['IITM'] = x['IITM'].isel(T=slice(1, None))    
    # x['ECMWF'] = x['ECMWF'].isel(T=slice(1, None))   

    # y['GEFS'] = y['GEFS'].isel(T=slice(0, -1))  
    # y['IITM'] = y['IITM'].isel(T=slice(1, None))    
    # y['ECMWF'] = y['ECMWF'].isel(T=slice(1, None))  


        #create a T_midpoint which is the midpoint of all T dates in the ensemble
    T1 = x[models[0]]['T'].values
    T2 = x[models[1]]['T'].values
    T_midpoint = T1 + (T2 - T1) / 2
    #convert to datetime
    T_midpoint = pd.to_datetime(T_midpoint)
    #applt midpoint to all models
    for name, model in x.items():
        x[name]['T'] = T_midpoint
        y[name]['T'] = T_midpoint

    #average y across models
    y_mme = xr.concat([y[model] for model in models], dim='model').mean(dim='model') 

    #check if at any point the models have different T values
    for name, model in x.items():
        assert np.all(model['T'].values == T_midpoint), f"Model {name} has different T values"

    print("############### ELR ###############")
    xtrain_dict, xval_dict, ytrain_list, ytest_list = preprocessing.bootstrap_splits_ELR_mme(x, y_mme, n_bootstraps= n_bootstraps)

    rpss_train_list_elr, rpss_test_list_elr, predictions_list_elr, y_test_oh_list_elr = training.train_elr_mme(xtrain_dict, ytrain_list, xval_dict, ytest_list)

    #save rpss values
    rpss_train_elr = xr.concat(rpss_train_list_elr, dim='bootstrap')
    rpss_test_elr = xr.concat(rpss_test_list_elr, dim='bootstrap')

    rpss_train_elr.to_netcdf('outputs/' + (dir or '') + f'MME_{obs}/ELR_rpss_train_{week}.nc')
    rpss_test_elr.to_netcdf('outputs/' + (dir or '') + f'MME_{obs}/ELR_rpss_test_{week}.nc')

    plots.plot_rpss_elr(rpss_train_list_elr, rpss_test_list_elr, week=week, obs=obs,model="MME",levels=None,dir=dir)

    print("############### ELR DONE ###############")

    print ("############### Neural Network ###############")

    xtrain_dict, xval_dict, xtest_dict, ytrain_list, yval_list, ytest_list = preprocessing.bootstrap_splits_mme(x, y_mme, n_bootstraps= n_bootstraps)

    #print train val test years for each bootstrap
    for i in range(n_bootstraps):
        print('Bootstrap', i+1)
        print('Train years:', set(xtrain_dict['GEFS'][i]['T'].dt.year.values))
        print('Validation years:', set(xval_dict['GEFS'][i]['T'].dt.year.values))
        print('Test years:', set(xtest_dict['GEFS'][i]['T'].dt.year.values))
        print('')

    architecture = "unet"   #unet or cnn or mlp
    #for unet you can specifiy the architecture parameters if training
    architecture_params = {"n_blocks": 3, "filters": 2, "ct_kernel": (3,3)} #if unet
    # you can also tune the architecture parameters, takes very long
    tuning_grid = {"n_blocks": [3,4,5], "n_filters": [2,3], "ct_kernels": [(2,2),(3,3),(5,5)], "batch_sizes": [16], "learning_rates": [1e-3],
                "patience": 10}
    
    rpss_train_list, rpss_val_list, rpss_test_list, predictions_list_nn, y_test_oh_list_nn = training.train_deepnet_mme(xtrain_dict, ytrain_list, xval_dict, yval_list, xtest_dict, ytest_list,
                                                                                                    training_type="tune", #train, tune or load
                                                                                                  architecture=architecture,
                                                                                                  architecture_params=architecture_params, #if train
                                                                                                  tuning_grid=tuning_grid, #if tune
                                                                                                  predictor="mean", #mean or stacked 
                                                                                                obs=obs, week=week,
                                                                                                epochs=100,
                                                                                                batch_size=16, #if not tuning
                                                                                                learning_rate=1e-3, #if not tuning
                                                                                                dir=dir
                                                                                                )
    
    rpss_train = xr.concat(rpss_train_list, dim='bootstrap')
    rpss_test = xr.concat(rpss_test_list, dim='bootstrap')
    rpss_val = xr.concat(rpss_val_list, dim='bootstrap')
    #save rpss values
    rpss_train.to_netcdf('outputs/' + (dir or '') + f'MME_{obs}/{architecture}_rpss_train_{week}.nc')
    rpss_val.to_netcdf('outputs/' + (dir or '') + f'MME_{obs}/{architecture}_rpss_val_{week}.nc')
    rpss_test.to_netcdf('outputs/' + (dir or '') + f'MME_{obs}/{architecture}_rpss_test_{week}.nc')
    
    #make a mask based on the training data with less than 3 labels
    def count_unique(values):
        return len(np.unique(values))
    # Apply the function along the time dimension ('T')
    y_test_terciled = y_test_oh_list_nn[0].argmax('category') 
    unique_counts = xr.apply_ufunc(count_unique, y_test_terciled, input_core_dims=[['T']], vectorize=True)
    # Mask grid points with less than 3 unique labels or NaNs
    mask1 = (unique_counts <3)
    mask2 = np.isnan(y_mme).any(dim='T')
    #combine masks
    mask = mask1 | mask2

    cbar_kwargs = {'shrink': 0.7,'spacing': 'proportional'}

    plots.plot_rpss_deepnet(rpss_train_list, rpss_val_list, rpss_test_list,model="MME", obs=obs, week=week, architecture=architecture, mask=mask
                        , cbar_kwargs=cbar_kwargs, custom_title = None,dir=dir)
    
    print ("############### Neural Network DONE ###############")

    predictions_elr = [predictions_list_elr[i]for i in range(n_bootstraps)]
    #stack predictions along bootstrap dimension
    predictions_masked_elr = xr.concat(predictions_elr, dim='T')

    y_test_oh_masked_elr = [y_test_oh_list_elr[i].where(~mask) for i in range(n_bootstraps)]
    #stack predictions along bootstrap dimension
    y_test_oh_masked_elr = xr.concat(y_test_oh_masked_elr, dim='T')


    t_bn_elr = y_test_oh_masked_elr.sel(category='below').values
    y_pred_bn_elr = predictions_masked_elr.sel(category='below').values

    t_n_elr  = y_test_oh_masked_elr.sel(category='normal').values
    y_pred_n_elr = predictions_masked_elr.sel(category='normal').values

    t_an_elr = y_test_oh_masked_elr.sel(category='above').values
    y_pred_an_elr = predictions_masked_elr.sel(category='above').values

    predictions_nn = [predictions_list_nn[i]for i in range(n_bootstraps)]
    #stack predictions along bootstrap dimension
    predictions_masked_nn = xr.concat(predictions_nn, dim='T')

    y_test_oh_masked_nn = [y_test_oh_list_nn[i].where(~mask) for i in range(n_bootstraps)]
    #stack predictions along bootstrap dimension
    y_test_oh_masked_nn = xr.concat(y_test_oh_masked_nn, dim='T')


    t_bn_nn = y_test_oh_masked_nn.sel(category='below').values
    y_pred_bn_nn = predictions_masked_nn.sel(category='below').values

    t_n_nn  = y_test_oh_masked_nn.sel(category='normal').values
    y_pred_n_nn = predictions_masked_nn.sel(category='normal').values

    t_an_nn = y_test_oh_masked_nn.sel(category='above').values
    y_pred_an_nn = predictions_masked_nn.sel(category='above').values

    plots.reliability_diagram_compare(y_pred_bn_nn, t_bn_nn, y_pred_bn_elr, t_bn_elr, model="MME", obs=obs, title=f'{week}-Below Normal',dir=dir)
    plots.reliability_diagram_compare(y_pred_n_nn, t_n_nn, y_pred_n_elr, t_n_elr, model="MME", obs=obs, title=f'{week}-Normal',dir=dir)
    plots.reliability_diagram_compare(y_pred_an_nn, t_an_nn, y_pred_an_elr, t_an_elr, model="MME", obs=obs, title=f'{week}-Above Normal',dir=dir)

    #delete all files in the models directory
    dir_path = os.path.join("models", dir, f"{model}_{obs}/{week}")
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


if __name__ == "__main__":
    #start timer
    start = time.time()
    main()
    end = time.time()
    #print elapsed time in hh:mm:ss
    print(time.strftime("%H:%M:%S", time.gmtime(end-start)))
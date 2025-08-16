
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

import utils.dataloader as dataloader
import utils.plots as plots
import utils.training as training
import utils.preprocessing as preprocessing
import os
import numpy as np
import xarray as xr
import time
import tensorflow as tf
import shutil


warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
#check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def main():

    obs = "IMD" 
    model = "ECMWF"
    domain = [67, 98, 7, 38] # West East South North, for Unet's check that lat and lot make a square divisible by 8, ie 24x24, 32x32, 64x64
    season = "May-Sep"
    n_bootstraps = 10
    years = (2003,2018)
    #choose lead time
    week = "wk3-4"  #wk1, wk2 or wk3-4
    dir = 'Common Period/'

    download = True
    if download:
        os.makedirs(f'download/{model}_{obs}', exist_ok=True)
    os.makedirs('models/' + (dir or '') + f'{model}_{obs}', exist_ok=True)
    os.makedirs('figures/'+ (dir or '') + f'{model}_{obs}', exist_ok=True)
    os.makedirs('outputs/'+ (dir or '') + f'{model}_{obs}', exist_ok=True)



    print(f' #######  TUNING {model} for {obs} for {week} lead time  ##########')

    x, y = dataloader.get_data(years=years, download = download,
                            week=week,obs=obs, custom_lead=(16,30), regrid=1,
                                domain=domain, season=season, 
                            model=model)

    


    ######## ELR ########
    print("############### ELR ###############")

    xtrain_list, ytrain_list, xval_list, yval_list = preprocessing.bootstrap_splits_ELR(x, y, n_bootstraps= n_bootstraps)

    rpss_train_list_elr, rpss_test_list_elr, predictions_list_elr, y_test_oh_list_elr = training.train_elr(xtrain_list, ytrain_list, xval_list, yval_list)

    #save rpss values
    rpss_train_elr = xr.concat(rpss_train_list_elr, dim='bootstrap')
    rpss_test_elr = xr.concat(rpss_test_list_elr, dim='bootstrap')

    rpss_train_elr.to_netcdf('outputs/' + (dir or '') + f'{model}_{obs}/ELR_rpss_train_{week}.nc')
    rpss_test_elr.to_netcdf('outputs/' + (dir or '') + f'{model}_{obs}/ELR_rpss_test_{week}.nc')

        #levels=[-0.3,-0.2,-0.1,-0.05, 0, 0.05,0.1, 0.2,0.4]
    plots.plot_rpss_elr(rpss_train_list_elr, rpss_test_list_elr, week=week, obs=obs,
                        model=model,
                        levels=None, dir = dir)
    
    print("############### ELR DONE ###############")

    print ("############### Neural Network ###############")

    xtrain_list, ytrain_list, xval_list, yval_list, xtest_list, ytest_list = preprocessing.bootstrap_splits(x, y, n_bootstraps=n_bootstraps)

        #print train val test years for each bootstrap
    for i in range(n_bootstraps):
        print('Bootstrap', i+1)
        print('Train years:', set(xtrain_list[i]['T'].dt.year.values))
        print('Validation years:', set(xval_list[i]['T'].dt.year.values))
        print('Test years:', set(xtest_list[i]['T'].dt.year.values))
        print('')


    architecture = "unet"   #unet or cnn or mlp
    #for unet you can specifiy the architecture parameters if training
    architecture_params = {"n_blocks": 3, "filters": 2, "ct_kernel": (3,3)} #if unet
    # you can also tune the architecture parameters, takes very long
    tuning_grid = {"n_blocks": [3], "n_filters": [2,3], "ct_kernels": [(2,2),(3,3),(5,5)], "batch_sizes": [16,32], "learning_rates": [1e-3,1e-4],
                "patience": 15}
    
    rpss_train_list, rpss_val_list, rpss_test_list, predictions_list_nn, y_test_oh_list_nn = training.train_deepnet(xtrain_list, ytrain_list,
                                                                                                  xval_list,yval_list,
                                                                                                    xtest_list, ytest_list,
                                                                                                    training_type="tune", #train, tune or load
                                                                                                  architecture=architecture,
                                                                                                  architecture_params=architecture_params, #if train
                                                                                                  tuning_grid=tuning_grid, #if tune
                                                                                                  predictor="mean", #mean or stacked 
                                                                                                obs=obs, week=week,
                                                                                                modname=model,
                                                                                                epochs=100,
                                                                                                batch_size=16, #if not tuning
                                                                                                learning_rate=1e-3, #if not tuning
                                                                                                #n_jobs=-1
                                                                                                dir=dir
                                                                                                )
    

    print ("############### Neural Network DONE ###############")
    #save rpss values
    rpss_train = xr.concat(rpss_train_list, dim='bootstrap')
    rpss_test = xr.concat(rpss_test_list, dim='bootstrap')
    rpss_val = xr.concat(rpss_val_list, dim='bootstrap')
    #save rpss values

    rpss_train.to_netcdf('outputs/' + (dir or '') + f'{model}_{obs}/{architecture}_rpss_train_{week}.nc')
    rpss_val.to_netcdf('outputs/' + (dir or '') + f'{model}_{obs}/{architecture}_rpss_val_{week}.nc')
    rpss_test.to_netcdf('outputs/' + (dir or '') + f'{model}_{obs}/{architecture}_rpss_test_{week}.nc')

    #make a mask based on the training data with less than 3 labels
    def count_unique(values):
        return len(np.unique(values))
    # Apply the function along the time dimension ('T')
    y_test_terciled = y_test_oh_list_nn[0].argmax('category') 
    unique_counts = xr.apply_ufunc(count_unique, y_test_terciled, input_core_dims=[['T']], vectorize=True)
    # Mask grid points with less than 3 unique labels or NaNs
    mask1 = (unique_counts <3)
    mask2 = np.isnan(y).any(dim='T')
    #combine masks
    mask = mask1 | mask2

    cbar_kwargs = {'shrink': 0.7,'spacing': 'proportional'}

    plots.plot_rpss_deepnet(rpss_train_list, rpss_val_list, rpss_test_list, model=model, obs=obs, week=week, architecture=architecture, mask=mask
                        , cbar_kwargs=cbar_kwargs,dir=dir)
    
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

    plots.reliability_diagram_compare(y_pred_bn_nn, t_bn_nn, y_pred_bn_elr, t_bn_elr, title=f'{week}-Below Normal',
                                      model=model, obs=obs,week=week,dir=dir)
    plots.reliability_diagram_compare(y_pred_n_nn, t_n_nn, y_pred_n_elr, t_n_elr, title=f'{week}-Normal',
                                        model=model, obs=obs,week=week,dir=dir)
    plots.reliability_diagram_compare(y_pred_an_nn, t_an_nn, y_pred_an_elr, t_an_elr, title=f'{week}-Above Normal',
                                        model=model, obs=obs,week=week,dir=dir)
    
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


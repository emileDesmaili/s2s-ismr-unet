
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import xarray as xr


def get_url_template(model, obs,regrid=None):
    """
    Returns predictor and predictand URL templates based on model and obs arguments.
    Raises ValueError if model or obs is invalid.
    """
    # Define supported models and observations
    supported_models = ["GEFS", "IITM1", "IITM2", "IITM3", "IITM4", "IITM5",
                        "IITM6",
                         "ECMWF_perturbed", "ECMWF_control"]
    supported_obs = ["IMD", "GPCP", "CHIRPS", "CPC"]

    # Check if provided arguments are valid
    if model not in supported_models:
        raise ValueError(f"Invalid model: {model}. Supported models are {supported_models}.")
    if obs not in supported_obs:
        raise ValueError(f"Invalid obs: {obs}. Supported obs are {supported_obs}.")

    # Define templates for predictor datasets
    hindcast_datasets = {
        "GEFS": 'SOURCES/.Models/.SubX/.EMC/.GEFSv12_CPC/.hindcast/.weekly/.pr/S/(0000%202%20Jan%20{first_year})/(0000%201%20Dec%20{final_year})/RANGEEDGES/S/7/STEP/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM1": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsasfer/.hindcast/.APCPsfc/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM2": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsasfer_sc/.hindcast/.APCPsfc/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM3": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsaszc/.hindcast/.APCPsfc/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM4": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsaszc_sc/.hindcast/.APCPsfc/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM5": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.sasfer/.hindcast/.APCPsfc/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM6": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.saszc/.hindcast/.APCPsfc/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "ECMWF_perturbed": 'home/.jingyuan/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/VALUES/S/7/STEP/S/({start_window}%20{fcast_year})/VALUES/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert//name/(prcp)/def/-999/setmissing_value/hdate/({first_year})/({final_year})/RANGE',
        "ECMWF_control": 'home/.jingyuan/.ECMWF/.S2S/.ECMF/.reforecast/.control/.sfc_precip/.tp/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/VALUES/S/7/STEP/S/({start_window}%20{fcast_year})/VALUES/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert//name/(prcp)/def/-999/setmissing_value/hdate/({first_year})/({final_year})/RANGE',
    }


    # Define templates for predictor datasets regridded
    hindcast_datasets_regrid = {
        "GEFS": 'SOURCES/.Models/.SubX/.EMC/.GEFSv12_CPC/.hindcast/.weekly/.pr/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/(0000%202%20Jan%20{first_year})/(0000%201%20Dec%20{final_year})/RANGEEDGES/S/7/STEP/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM1": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsasfer/.hindcast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM2": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsasfer_sc/.hindcast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM3": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsaszc/.hindcast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM4": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsaszc_sc/.hindcast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM5": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.sasfer/.hindcast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "IITM6": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.saszc/.hindcast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/%28{first_year}-{final_year}%29VALUES/S/({start_window})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D//keepgrids/average//units/(mm/day)/def//name/(prcp)/def',
        "ECMWF_perturbed": 'home/.jingyuan/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/VALUES/S/7/STEP/S/({start_window}%20{fcast_year})/VALUES/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert//name/(prcp)/def/-999/setmissing_value/hdate/({first_year})/({final_year})/RANGE',
        "ECMWF_control": 'home/.jingyuan/.ECMWF/.S2S/.ECMF/.reforecast/.control/.sfc_precip/.tp/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/VALUES/S/7/STEP/S/({start_window}%20{fcast_year})/VALUES/%5BL%5Ddifferences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert//name/(prcp)/def/-999/setmissing_value/hdate/({first_year})/({final_year})/RANGE',
    }

    time_gridders = {
        'GEFS': '/L/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/',
        'IITM1': '/L/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/',
        'IITM2': '/L/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/',
        'IITM3': '/L/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/',
        'IITM4': '/L/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/',
        'IITM5': '/L/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/',
        'IITM6': '/L/S/add/0/RECHUNK//name//T/def/2/%7Bexch%5BL/S%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/',
        'ECMWF_perturbed': '/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20{fcast_year}-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/',
        'ECMWF_control': '/hdate//pointwidth/0/def/-6/shiftGRID/hdate/(days%20since%201960-01-01)/streamgridunitconvert/S/(days%20since%20{fcast_year}-01-01)/streamgridunitconvert/S//units//days/def/L/hdate/add/add/0/RECHUNK/L/removeGRID//name//T/def/2/%7Bexch%5BS/hdate%5D//I/nchunk/NewIntegerGRID/replaceGRIDstream%7Drepeat/use_as_grid/',
              }


    # Define templates for predictand datasets
    predictand_datasets = {
        "IMD": 'SOURCES/.IMD/.RF0p25/.gridded/.daily/.v1989-2022/.rf/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE',
        "GPCP": 'SOURCES/.NASA/.GPCP/.V1DD/.V1p3/.precip/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE',
        "CHIRPS": 'SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global/.0p25/.prcp/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE',
        "CPC": 'SOURCES/.NOAA/.NCEP/.CPC/.temperature/.daily/.tmin/SOURCES/.NOAA/.NCEP/.CPC/.temperature/.daily/.tmax/add/2/div/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE',
    }


    time_gridder = time_gridders[model]

    if regrid:
        predictor_dataset = hindcast_datasets_regrid[model]
    else:
        predictor_dataset = hindcast_datasets[model]
    
    predictand_dataset = predictand_datasets[obs]
    
    if model == 'ECMWF_perturbed' or model == 'ECMWF_control':
        predictor_url_template = f'https://iridl.ldeo.columbia.edu/{predictor_dataset}{time_gridder}L/removeGRID/data.nc'
    else:
        predictor_url_template = f'https://iridl.ldeo.columbia.edu/{predictor_dataset}/L/removeGRID/data.nc'


    predictand_url_template = f'https://iridl.ldeo.columbia.edu/{predictor_dataset}{time_gridder}{predictand_dataset}/%5BX/Y%5D/regridLinear/T/(days%20since%201960-01-01)/streamgridunitconvert/T/{{lead_end}}/{{lead_start}}/sub/runningAverage/T/2/index/.T/SAMPLE/nip//name/(prcp)/def/data.nc'

    return predictor_url_template, predictand_url_template


def get_file(name, url_template, years, download, lead, model, obs, domain, season,regrid=None):
    """
    Downloads or reads a NetCDF file and extracts data based on the provided domain and dataset.

    Args:
        name (str): Filename prefix for downloaded files.
        url_template (str): URL template for downloading the dataset.
        years (tuple): Range of years to retrieve data for (start, end).
        download (bool): Whether to download the file or use local files.
        lead (tuple): Lead time window in days (start, end).
        model (str): Model name (e.g., "GEFS", "IITM").
        obs (str): Observation dataset name (e.g., "IMD", "GPCP", "CHIRPS", "CPC").
        domain (tuple): Domain boundaries (west, east, south, north).
        season (str): Seasonal window (e.g., "Jan-Apr", "Jul-Sep").
    
    Returns:
        xarray.DataArray: The requested data array.
    """
    west, east, south, north = domain
    #mif model contains IITM in the name string
    if 'IITM' in model:
        model = 'IITM'
    if 'ECMWF' in model:
        model = 'ECMWF'
    fname = f'download/{model}_{obs}/{name}_{years[0]}-{years[1]}.nc'
    
    # Construct dataset URL
    dset_url = url_template.format(
        first_year=years[0],
        final_year=years[1],
        lead_start=lead[0],
        lead_end=lead[1],
        start_window=season,
        west=west,
        east=east,
        south=south,
        north=north,
        fcast_year = 2023,
        regrid=regrid
    )
    
    # Download the file if required
    if download:
        dset_url_no_data = dset_url.replace('data.nc','')
        print(f"Downloading data from: {dset_url_no_data}")
        subprocess.call(['curl', '-b', 'cookies.txt', '-k', dset_url, '-o', fname])
    
    # Load the data based on observation type
    if obs == "CPC":
        da = xr.open_dataset(fname)['temp']
    else:
        da = xr.open_dataset(fname)['prcp']
    
    return da


def get_data(years, download, week, model, obs, domain, season,regrid=None,custom_lead=None):
    """
    Retrieves predictor and predictand data for a specified week and domain.

    Args:
        years (tuple): Range of years (start, end).
        download (bool): Whether to download files or use local files.
        week (str): Target week ("wk1", "wk2", "wk3-4").
        model (str): Model name.
        obs (str): Observation dataset name.
        domain (tuple): Domain boundaries (west, east, south, north).
        season (str): Seasonal window.
        custom_lead (tuple): Custom lead time window in days (start, end).

    Returns:
        tuple: Predictor (x) and predictand (y) data arrays.
    """
    # Map week to lead times
    lead_mapping = {'wk1': (2, 8), 'wk2': (9, 15), 'wk3-4': (16, 29)}
    if custom_lead:
        lead = custom_lead
    else:
        lead = lead_mapping[week]


    # Get the URL templates for the model and observation

    if model == "IITM":
        iitm_models = ['IITM1','IITM2','IITM3','IITM4','IITM5','IITM6']
        x = []
        for iitm_model in iitm_models:
            predictor_url_template, predictand_url_template = get_url_template(iitm_model, obs,regrid=regrid)
            # Fetch predictor data
            xi = get_file(
                f'{iitm_model}_{week}',
                predictor_url_template,
                years=years,
                download=download,
                lead=lead,
                model=iitm_model,
                obs=obs,
                domain=domain,
                season=season,
                regrid=regrid
            )

            x.append(xi)
            # Fetch predictand data
            y = get_file(
                f'IITM_{obs}_{week}',
                predictand_url_template,
                years=years,
                download=download,
                lead=lead,
                model=iitm_model,
                obs=obs,
                domain=domain,
                season=season,
                regrid=regrid
            )

        # Combine predictors and predictands
        x = xr.concat(x, dim='M')
        t = x['S'] + pd.Timedelta(days=(lead[0] + lead[1]) / 2)
        x = x.assign_coords(T=('S', t.values)).swap_dims(S='T')

    elif model == "ECMWF":
        ecmwf_models = ['ECMWF_perturbed','ECMWF_control']
        x = []
        for ecmwf_model in ecmwf_models:
            predictor_url_template, predictand_url_template = get_url_template(ecmwf_model, obs,regrid=regrid)
            # Fetch predictor data
            xi = get_file(
                f'{ecmwf_model}_{week}',
                predictor_url_template,
                years=years,
                download=download,
                lead=lead,
                model=ecmwf_model,
                obs=obs,
                domain=domain,
                season=season,
                regrid=regrid
            )
            #if dim M is not present, add it
            if 'M' not in xi.dims:
                xi = xi.expand_dims("M").assign_coords(M=[11])
            else:
                xi = xi

            x.append(xi)
            # Fetch predictand data
            y = get_file(
                f'ECMWF_{obs}_{week}',
                predictand_url_template,
                years=years,
                download=download,
                lead=lead,
                model=ecmwf_model,
                obs=obs,
                domain=domain,
                season=season,
                regrid=regrid
            )

        # Combine predictors and predictands
        x = xr.concat(x, dim='M')

    else:

        predictor_url_template, predictand_url_template = get_url_template(model, obs,regrid=regrid)

        # Fetch predictor data
        x = get_file(
            f'{model}_{week}',
            predictor_url_template,
            years=years,
            download=download,
            lead=lead,
            model=model,
            obs=obs,
            domain=domain,
            season=season,
            regrid=regrid
        )
        if model != 'ECMWF':
            t = x['S'] + pd.Timedelta(days=(lead[0] + lead[1]) / 2)
            x = x.assign_coords(T=('S', t.values)).swap_dims(S='T')

        # Fetch predictand data
        y = get_file(
            f'{model}_{obs}_{week}',
            predictand_url_template,
            years=years,
            download=download,
            lead=lead,
            model=model,
            obs=obs,
            domain=domain,
            season=season,
            regrid=regrid
        )

        # Ensure matching time dimensions
    assert (x['T'].values == y['T'].values).all(), "Mismatch in time dimensions between x and y."

    # Return transposed predictor and predictand
    return x.transpose('T', 'M', 'Y', 'X'), y

def get_data_ensemble(years, download, week, models, obs, domain, season,regrid=1,custom_leads=None, custom_seasons=None):
    """
    Retrieves data for multiple forecast models and combines predictors.
    
    Parameters:
    - years (tuple): Start and end years for data.
    - download (bool): Whether to download missing data.
    - week (str): Forecast week ('wk1', 'wk2', 'wk3-4').
    - models (list of str): List of forecast models.
    - obs (str): Observation dataset name.
    - domain (tuple): Geographic bounding box (west, east, south, north).
    - season (str): Season filter.
    
    Returns:
    - Tuple of (xarray.DataArray, xarray.DataArray): Ensemble predictors (x) and single target (y).
    """
    x_ensemble = []
    y_ensemble = []
    for model in models:
        custom_lead = custom_leads[model] if custom_leads else None
        season = custom_seasons[model] if custom_seasons else season
        x, y = get_data(years = years, download = download, week = week, model = model, obs = obs,
                         domain = domain, season = season, regrid=regrid, custom_lead=custom_lead)
        x_ensemble.append(x)
        y_ensemble.append(y)

    #make a dict with names of models and xarray data for x
    x_ensemble = dict(zip(models, x_ensemble))
    y = dict(zip(models, y_ensemble))

    return x_ensemble, y


    
    #===================================================================================================
    ###                                  Get forecast data file
    #===================================================================================================

def download_forecast(model,obs,day,month,year,domain,week,dir,download=True,regrid=None,custom_lead=None):
        # Map week to lead times
    lead_mapping = {'wk1': (2, 8), 'wk2': (9, 15), 'wk3-4': (16, 29)}
    if custom_lead:
        lead = custom_lead
    else:
        lead = lead_mapping[week]

    west, east, south, north = domain
    if regrid == None:
        forecast_urls = {"GEFS": 'SOURCES/.Models/.SubC/.EMC/.GEFSv12_CPC/.forecast/.pr/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def/86400/mul',
                        "IITM1": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsasfer/.forecast/.APCPsfc/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
                        "IITM2": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsasfer_sc/.forecast/.APCPsfc/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
                        "IITM3": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsaszc/.forecast/.APCPsfc/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
                        "IITM4": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsaszc_sc/.forecast/.APCPsfc/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
                        "IITM5": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.sasfer/.forecast/.APCPsfc/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
                        "IITM6": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.saszc/.forecast/.APCPsfc/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
                        "ECMWF1": 'SOURCES/.ECMWF/.S2S/.ECMF/.CY41-47/.forecast/.perturbed/.sfc_precip/.tp/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/VALUES/%5B/L%5D/differences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert//name/(prcp)/def/-999/setmissing_value/%5BL%5D/average',
                        "ECMWF2": 'SOURCES/.ECMWF/.S2S/.ECMF/.CY48/.forecast/.perturbed/.sfc_precip/.tp/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/VALUES/%5B/L%5D/differences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert//name/(prcp)/def/-999/setmissing_value/%5BL%5D/average',    
                        }
    else:

        forecast_urls = {"GEFS": 'SOURCES/.Models/.SubC/.EMC/.GEFSv12_CPC/.forecast/.pr/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def/86400/mul',
            "IITM1": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsasfer/.forecast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
            "IITM2": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsasfer_sc/.forecast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
            "IITM3": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsaszc/.forecast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
            "IITM4": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsaszc_sc/.forecast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
            "IITM5": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.sasfer/.forecast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
            "IITM6": 'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.saszc/.forecast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/RANGEEDGES/%5B/L%5D/average//units/(mm/day)/def//name/(prcp)/def',
            "ECMWF1": 'SOURCES/.ECMWF/.S2S/.ECMF/.CY41-47/.forecast/.perturbed/.sfc_precip/.tp/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/VALUES/%5B/L%5D/differences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert//name/(prcp)/def/-999/setmissing_value/%5BL%5D/average',
            "ECMWF2": 'SOURCES/.ECMWF/.S2S/.ECMF/.CY48/.forecast/.perturbed/.sfc_precip/.tp/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/S/(0000%20{day}%20{month}%20{year})/VALUES/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE/L/{lead_start}/{lead_end}/VALUES/%5B/L%5D/differences/c%3A//name//water_density/def/998/(kg/m3)/%3Ac/div//mm/unitconvert//name/(prcp)/def/-999/setmissing_value/%5BL%5D/average',    
            }
    months_dict = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    month_num = months_dict[month]

    def get_forecast_url(model):
        if model == 'ECMWF':
            #if day and month are before or Jun 27, use ECMWF1, else use ECMWF2
            if month_num < 6 or (month_num == 6 and day < 27):
                forecast_url = forecast_urls['ECMWF1']  
            else:
                forecast_url = forecast_urls['ECMWF2']
        else:

            forecast_url = forecast_urls[model]
        
        forecast_url_template =f'https://iridl.ldeo.columbia.edu/{forecast_url}/data.nc'
        return forecast_url_template

    def get_forecast_file(url_template,model, download=download):
        fname = './download/'  + (dir or '') + f'{model}_{obs}/forecast_{day}_{month}_{year}.nc'
        dset_url = url_template.format(

            year=year,
            month=month,
            day=day,
            lead_start=lead[0],
            lead_end=lead[1],
            west=west,
            east=east,
            south=south,
            north=north,
            regrid=regrid
        )
        if download:
            dset_url_no_data = dset_url.replace('data.nc','')
            print(dset_url_no_data)
            subprocess.call(['curl','-b','cookies.txt','-k',dset_url, '-o',fname])

        da = xr.open_dataset(fname)['prcp']
        return da

    def get_forecast_data(model,download=download):
        if model == "IITM":
            iitm_models = ['IITM1','IITM2','IITM3','IITM4','IITM5','IITM6']
            x = []
            for iitm_model in iitm_models:
                forecast_url_template = get_forecast_url(iitm_model)
                xi = get_forecast_file(forecast_url_template,model, download)
                x.append(xi)
            x = xr.concat(x, dim='M')

        else:
            forecast_url_template = get_forecast_url(model)
            x = get_forecast_file(forecast_url_template,model, download)
        t = x['S'] + pd.Timedelta(days=(lead[0] + lead[1]) / 2)
        x = x.assign_coords(T=('S', t.values)).swap_dims(S='T')

        return x

    xf = get_forecast_data(model,download=True)
    
    return xf


def get_obs(model, obs, domain, week, years, season, dir,download=True,regrid=None):
    west, east, south, north = domain
    lead_mapping = {'wk1': (2, 8), 'wk2': (9, 15), 'wk3-4': (16, 29)}
    lead = lead_mapping[week]
    lead_start = lead[0]
    lead_end = lead[1]

    if regrid == None:
        model_urls = {
            "GEFS": 'SOURCES/.Models/.SubX/.EMC/.GEFSv12_CPC/.hindcast/.weekly/.pr/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE',
            "IITM":'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsasfer/.hindcast/.APCPsfc/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE',
            "ECMWF":'home/.jingyuan/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE',
            }
    else:
        model_urls = {
            "GEFS": 'SOURCES/.Models/.SubX/.EMC/.GEFSv12_CPC/.hindcast/.weekly/.pr/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE',
            "IITM":'SOURCES/.IITM/.ERPv2/.r0p5/.CFS/.nsasfer/.hindcast/.APCPsfc/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE',
            "ECMWF":'home/.jingyuan/.ECMWF/.S2S/.ECMF/.reforecast/.perturbed/.sfc_precip/.tp/X/-180/{regrid}/179/GRID/Y/-90/{regrid}/90/GRID/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE',
            }

    # Define templates for predictand datasets
    obs_urls = {
        "IMD": 'SOURCES/.IMD/.RF0p25/.gridded/.daily/.v1989-2022/.rf/Y/{south}/{north}/RANGE/X/{west}/{east}/RANGE',
    }

    obs_url = obs_urls[obs]
    model_url = model_urls[model]
    obs_url_template =  f'https://iridl.ldeo.columbia.edu/{model_url}/{obs_url}/%5BX/Y%5DregridLinear/T/(days%20since%201960-01-01)/streamgridunitconvert/T/{lead_end}/{lead_start}/sub/runningAverage//name/(prcp)/def/data.nc'
    

    def get_obs_file(url, obs, download=True):

        fname = './download/'  + (dir or '') + f'{model}_{obs}/{obs}_{week}.nc'
        url = url.format(
            west=west,
            east=east,
            south=south,
            north=north,
            regrid=regrid
        )
        
        if download:
            print(f"Downloading: {url}")
            subprocess.call(['curl', '-b', 'cookies.txt', '-k', url, '-o', fname])

        da = xr.open_dataset(fname)['prcp']
        return da
    
    y = get_obs_file(obs_url_template, obs, download)
    #keep only Jun to Oct months
    #map seaon string like (Jun-Sep) to months
    months_dict = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    start_month = months_dict[season.split('-')[0]]
    end_month = months_dict[season.split('-')[1]]+1
    month_list = list(range(start_month,end_month+1))

    y = y.sel(T = y['T.month'].isin(month_list))
    #slice years formatted in dd-mm-yyyy
    start_year = '01-01-'+str(years[0])
    end_year = '31-12-'+str(years[1])
    y = y.sel(T = slice(start_year,end_year))

    return y







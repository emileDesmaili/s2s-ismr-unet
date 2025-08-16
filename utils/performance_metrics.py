import cartopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import xarray as xr
import xskillscore



def climo_predict(x,predictor="mean"):
    '''Predict function that always predicts climatological conditions,
    i.e. equal probabilities for all three terciles.'''
    x = x.mean('M')
    fcst = xr.concat(
        [xr.full_like(x, 1/3)] * 3,
        'category'
    ).where(x.notnull())
    if predictor=="mean":
        return fcst
    elif predictor=="stacked":
        return fcst.transpose('MT', 'Y', 'X', 'category')
    return fcst


def rps(obs, fcst,predictor="mean"):
    # Put categorical observations in the shape of a forecast where
    # the probability of one tercile is 1 and the others are 0.
    obs_reshaped = xr.concat(
        [
            (obs == 0).assign_coords(category='below'),
            (obs == 1).assign_coords(category='normal'),
            (obs == 2).assign_coords(category='above'),
        ],
        'category'
    ).where(obs.notnull()).transpose('T', 'Y', 'X', 'category')
    if predictor=="mean":
        return xskillscore.rps(obs_reshaped, fcst, dim='T', category_edges=None, input_distributions='p')
    elif predictor=="stacked":
        return xskillscore.rps(obs_reshaped, fcst, dim='MT', category_edges=None, input_distributions='p')



def rpss(reference, forecast, observations,predictor="mean"):
    return 1 - rps(observations, forecast,predictor) / rps(observations, reference,predictor)











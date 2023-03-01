# PYTHON Imports 
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import ipywidgets as widgets
from matplotlib.colors import LogNorm
import pickle
# ASTROPHY Imports
import astropy 
from astropy.table import Table
from astropy.io import fits
# CIAO Imports
import ciao_contrib.runtool
from ciao_contrib.runtool import *

# Define Custom Functions

def hist2D_representation_bonafide_fun(df_eventfile_input, id_name, nbins_E, nbins_dt, nbins_t, dt_type='lin',normalise = True):
    Emin = np.log(500)
    Emax = np.log(7000)
    # Copy df
    df = df_eventfile_input.copy()
    # IDs
    obsid = id_name.split("_")[0]
    regid = id_name.split("_")[1]
    # Eventfile length and duration for normalisation
    N_length = len(df) 
    T_duration = max(df["time"])-min(df["time"])
    if normalise:
        N = N_length
        T = T_duration
    else:
        N = 1
        T = 1
    # Add E column
    df["E"] = np.log(df["energy"])
    # Add t column
    df["t"] = (df["time"]-min(df["time"]))/(max(df["time"])-min(df["time"]))
    # Add delta_time column
    df["delta_time"] = df['time'].diff()
     # Add dt column
    df = df[df["delta_time"].notna()]
    df["delta_time"] = df["delta_time"].apply(lambda dt: np.where(dt == 0, dt + 0.01, dt))

    if dt_type == 'lin':
        df["dt"] = (N/T * (df["delta_time"]))
    elif dt_type == 'log':
        df["dt"] = np.log(N/T * df["delta_time"])
    elif dt_type == 'log*100':
        df["dt"] = np.log(N/T * 100*df["delta_time"])
    elif dt_type == 'log+1':
        df["dt"] = np.log(N/T * df["delta_time"]+1)
    elif dt_type == 'exp':
        df["dt"] = np.exp(N/T * df["delta_time"])
    elif dt_type == '1.1^x':
        df["dt"] = 1.1 ** (N/T * df["delta_time"])
    elif dt_type == 'inv':
        df["dt"] = 1.1 / (N/T * df["delta_time"])
    elif dt_type == 'x^2':
        df["dt"] = (N/T * df["delta_time"]) **2
    elif dt_type == 'x^-2':
        df["dt"] = (N/T * df["delta_time"]) **(-2)
    elif dt_type == 'x^1.1':
        df["dt"] = (N/T * df["delta_time"]) **1.1
    elif dt_type == 'x^-1.1':
        df["dt"] = (N/T * df["delta_time"]) **(-1.1)


    dt_min = min(df["dt"])
    dt_max = max(df["dt"])
    df["dt"] = (df["dt"]- dt_min)/(dt_max-dt_min)

    # Plot histograms
    fig,axs=plt.subplots(1,3,figsize=(12,2),constrained_layout = True)
    fig.suptitle(f'{dt_type} and normalised: {normalise} (N: {N_length} counts, T: {T_duration}s')
    plt.subplot(1, 3, 1)
    plt.title(f'E vs t for ID: {id_name}')
    Et = plt.hist2d(df["t"],df["E"],range = [[0,1],[Emin, Emax]],bins=(nbins_t,nbins_E),norm=LogNorm(),cmap = 'plasma') 
    plt.subplot(1, 3, 2)
    plt.title(f'E vs dt for ID: {id_name}')
    Edt = plt.hist2d(df["dt"],df["E"],range = [[0,1],[Emin, Emax]],bins=(nbins_dt,nbins_E),norm=LogNorm(),cmap = 'plasma') 
    plt.subplot(1, 3, 3)
    plt.title(f'dt vs t for ID: {id_name}')
    dtt = plt.hist2d(df["t"],df["dt"],range = [[0,1],[0,1]],bins=(nbins_t,nbins_dt),norm=LogNorm(),cmap = 'plasma') 
    plt.show()

    return



# 1. 2D Representation Function (Logarithm)
def data_representation2D_fun(df_eventfiles_input,df_properties_input,global_path,set_id,nbins_E,nbins_dt,colormap='Viridis',showplot=True):
    # Group dataframes by IDs
    df_eventfiles_group = df_eventfiles_input.groupby('obsreg_id')
    df_properties_group = df_properties_input.groupby('obsreg_id')
    # Initialise features, labels and ids lists
    x_features = []
    y_labels = []
    id_pass = []
    id_fail = []
    # Initialise counters
    count = 0
    fails = 0
    count_limit = df_eventfiles_group.ngroups
    # Loop over all eventfiles
    for id_name, dfi in df_eventfiles_group:
        # Add delta_time column
        dfi["delta_time"] = dfi['time'].diff()
        # Remove first row as delta_time = nan
        dfi = dfi[dfi["delta_time"].notna()]
        # Add a constant value "pseudo-count" 0.1 to delta_time = 0 
        dfi["delta_time"] = dfi["delta_time"].apply(lambda dt: np.where(dt == 0, dt + 0.1, dt))
        try:
            # Eventfile length and duration for normalisation
            N = len(dfi) 
            T = max(dfi["time"])-min(dfi["time"])
            # Add dt column (with normalisations applied)
            dfi["dt"] = np.log10(N * dfi["delta_time"]/T)
            dt_min = min(dfi["dt"])
            dt_max = max(dfi["dt"])
            dfi["dt"] = (dfi["dt"]- dt_min)/(dt_max-dt_min)
            # Add E column
            dfi["E"] = np.log10(dfi["energy"])
            # Create histogram representation
            fig, ax = plt.subplots(figsize=(3,3))
            #hist = plt.hist2d(df["dt"],df["E"],range = [[dt_axis_min, dt_axis_max],[np.log10(500.), np.log10(7000.)]],bins=(nbins_dt,nbins_E),norm=LogNorm(),cmap = 'plasma', density = True) 
            hist2D = ax.hist2d(dfi["dt"],dfi["E"],range = [[0, 1],[np.log10(500.), np.log10(7000.)]],bins=(nbins_dt,nbins_E),norm=LogNorm(),cmap = colormap)
            if showplot:
                print(id_name)
                plt.show()
            # Create features 
            x = hist2D[0]/np.max(hist2D[0])
            # Create labels
            y = df_properties_group.get_group(id_name)[['cnts_aper_b','cnts_aperbkg_b','src_cnts_aper_b','flux_aper_b','hard_hm','hard_hs','hard_ms','var_prob_b','var_prob_h','var_prob_m',	'var_prob_s']].to_numpy()
            # Append features, labels and ids lists
            x_features.append(x)
            y_labels.append(y)
            id_pass.append(id_name)
            count = count + 1
            print(f'Counter: {count} / {count_limit}')
        except:     
            id_fail.append(id_name)
            fails = fails + 1
            print(f'Fails: {fails}')
    # Concatenate labels into a matrix for easier use
    y_labels = np.concatenate(y_labels)
    return  x_features, y_labels, id_pass, id_fail
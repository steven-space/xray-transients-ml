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

# Data Representation 

def hist2D_Et(df_eventfile_input, id_name, nbins_E, nbins_t,minmax_norm = True, plot = True):
    # Copy df
    df = df_eventfile_input.copy()
    df.sort_values(by='time', inplace = True) 
    df.reset_index(drop=True, inplace = True)
    # Define histogram boundaries
    E_start = np.log10(500)
    E_end = np.log10(7000)
    t_start = 0
    t_end = 1
    # IDs
    obsid = id_name.split("_")[0]
    regid = id_name.split("_")[1]
    # Eventfile length and duration
    N_length = len(df) 
    T_duration = max(df["time"])-min(df["time"])
    # Add E, t column
    df["E"] = np.log10(df["energy"])
    df["t"] = (df["time"]-min(df["time"]))/(max(df["time"])-min(df["time"]))
    # Add Et histogram
    hist_Et = np.histogram2d(df["t"],df["E"],range = [[t_start,t_end],[E_start, E_end]],bins=(nbins_t,nbins_E)) 
    if minmax_norm == True:
        feature = hist_Et[0]/np.max(hist_Et[0])
    else:
        feature = hist_Et[0]
    if plot == True:
        plt.imshow(feature.T, origin='lower', extent=[0, 1, E_start, E_end], cmap='plasma',norm=LogNorm())
        plt.colorbar()
        plt.xlabel('Time')
        plt.ylabel('Log10(Energy)')
        plt.show()
    return feature







# Actual Data Representation Function

def hist2D_representation_bonafide_fun2(df_eventfile_input, id_name, nbins_E, nbins_dt, nbins_t, dt_type='x3',pre_normalise = True, dt_normalise = True):
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
    # Add E, t column
    df["E"] = np.log(df["energy"])
    df["t"] = (df["time"]-min(df["time"]))/(max(df["time"])-min(df["time"]))
     # Add delta_time column
    df["delta_time"] = df['time'].diff().shift(-1)
    # normalise from -1 to 1
    if pre_normalise:
        df["delta_time"] = 2*(df["delta_time"]-min(df["delta_time"]))/(max(df["delta_time"])-min(df["delta_time"])) - 1
    # Add dt column
    if dt_type == 'x3':
        df["dt"] = ((df["delta_time"])) **3
    if dt_type == 'x5':
        df["dt"] = ((df["delta_time"])) **3
    if dt_type == 'x7':
        df["dt"] = ((df["delta_time"])) **3
    if dt_type == 'x9':
        df["dt"] = ((df["delta_time"])) **3
    #print(df["dt"])
    if dt_normalise:
        dt_min = min(df["dt"])
        dt_max = max(df["dt"])
        df["dt"] = 2*(df["dt"]- dt_min)/(dt_max-dt_min)-1

    # Plot histograms
    fig,axs=plt.subplots(1,3,figsize=(12,2),constrained_layout = True)
    fig.suptitle(f'{dt_type}, (N: {N_length} counts, T: {T_duration}s)')
    plt.subplot(1, 3, 1)
    plt.title(f'E vs t for ID: {id_name}')
    Et = plt.hist2d(df["t"],df["E"],range = [[0,1],[Emin, Emax]],bins=(nbins_t,nbins_E),norm=LogNorm(),cmap = 'plasma') 
    plt.subplot(1, 3, 2)
    plt.title(f'E vs dt for ID: {id_name}')
    Edt = plt.hist2d(df["dt"],df["E"],range = [[-1,1],[Emin, Emax]],bins=(nbins_dt,nbins_E),norm=LogNorm(),cmap = 'plasma') 
    plt.subplot(1, 3, 3)
    plt.title(f'dt vs t for ID: {id_name}')
    dtt = plt.hist2d(df["t"],df["dt"],range = [[0,1],[-1,1]],bins=(nbins_t,nbins_dt),norm=LogNorm(),cmap = 'plasma') 
    plt.show()
    return

# Define Custom Functions

def hist2D_representation_bonafide_fun(df_eventfile_input, id_name, nbins_E, nbins_dt, nbins_t, dt_type='lin',pre_normalise = True, normalise = True, log_time = False):
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
    if log_time:
        df["t"] = np.log(df["time"])
    else: 
        df["t"] = df["time"]
    df["t"] = (df["t"]-min(df["t"]))/(max(df["t"])-min(df["t"]))
    # Add delta_time column
    df["delta_time"] = df['time'].diff().shift(-1)
    df = df[df["delta_time"].notna()]
    if pre_normalise:
        df["delta_time"] = (df["delta_time"]-min(df["delta_time"]))/(max(df["delta_time"])-min(df["delta_time"]))+0.1
        #print(df["delta_time"])
    df["delta_time"] = df["delta_time"].apply(lambda dtx : np.where(dtx == 0, dtx + 0.1, dtx))
    print(df["delta_time"])
     # Add dt column
    if dt_type == 'lin':
        df["dt"] = (N/T * (df["delta_time"]))
    elif dt_type == 'log':
        df["dt"] = np.log(N/T * df["delta_time"])
    elif dt_type == 'log2':
        df["dt"] = np.log2(N/T * df["delta_time"])
    elif dt_type == 'log*100':
        df["dt"] = np.log(N/T * 100*df["delta_time"])
    elif dt_type == 'exp':
        df["dt"] = np.exp(N/T * df["delta_time"])
    elif dt_type == '2^x':
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
    elif dt_type == 'x^1.5':
        df["dt"] = (N/T * df["delta_time"]) **1.5
    elif dt_type == 'x^-1.5':
        df["dt"] = (N/T * df["delta_time"]) **(-1.5)
    elif dt_type == 'sqrt':
        df["dt"] = np.sqrt((N/T * df["delta_time"]))

    #print(df["dt"])
    dt_min = min(df["dt"])
    dt_max = max(df["dt"])
    df["dt"] = (df["dt"]- dt_min)/(dt_max-dt_min)

    # Plot histograms
    fig,axs=plt.subplots(1,3,figsize=(12,2),constrained_layout = True)
    fig.suptitle(f'{dt_type}, normalised: {normalise}, pre_normalised: {pre_normalise}  (N: {N_length} counts, T: {T_duration}s)')
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
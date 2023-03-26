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

def hist2D(df_eventfile_input, id_name, nbins_E, nbins_t, norm = 'minmax', plot = True, colmap = 'plasma', lognorm = True):
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
    # Create feature
    if norm == 'minmax':
        feature = (hist_Et[0]-np.min(hist_Et[0]))/(np.max(hist_Et[0])-np.min(hist_Et[0]))
    elif norm == 'none':
        feature = hist_Et[0]
    if plot == True:
        if lognorm == True:
            plt.imshow(feature.T, origin='lower', extent=[0, 1, E_start, E_end], cmap=colmap,norm=LogNorm())
        elif lognorm == False:
            plt.imshow(feature.T, origin='lower', extent=[0, 1, E_start, E_end], cmap=colmap)
        #plt.colorbar()
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$\epsilon$')
        plt.title(f'ObsID: {obsid}, RegID: {regid}, N: {N_length}, T: {int(T_duration)}s')
        plt.show()
    return feature

def hist3D(df_eventfile_input, id_name, nbins_E, nbins_t, nbins_dt, norm = 'minmax', plot = True, colmap = 'plasma', lognorm = True):
    # Copy df
    df = df_eventfile_input.copy()
    df.sort_values(by='time', inplace = True) 
    df.reset_index(drop=True, inplace = True)
    # Define histogram boundaries
    E_start = np.log10(500)
    E_end = np.log10(7000)
    t_start = 0
    t_end = 1
    dt_start = 0
    dt_end = 1
    # IDs
    obsid = id_name.split("_")[0]
    regid = id_name.split("_")[1]
    # Eventfile length and duration
    N_length = len(df) 
    T_duration = max(df["time"])-min(df["time"])
    # Add E, t, dt columns
    df["E"] = np.log10(df["energy"])
    df["t"] = (df["time"]-min(df["time"]))/(max(df["time"])-min(df["time"]))
    # df["delta_time"] = df['time'].diff().shift(-1)
    # df = df[df["delta_time"].notna()]
    # df["dt"] = (df['delta_time'] - df['delta_time'].mean()) / df['delta_time'].std() 
    # df["dt"] = (df["dt"]-min(df["dt"]))/(max(df["dt"])-min(df["dt"]))
    df["delta_time"] = df['t'].diff().shift(-1)
    df = df[df["delta_time"].notna()]
    df["dt"] = (df['delta_time']-min(df['delta_time']))/(max(df['delta_time'])-min(df['delta_time']))
    # Add Etdt histogram
    hist3D, edges = np.histogramdd((df["t"], df["E"], df["dt"]), range = [[t_start,t_end],[E_start, E_end], [dt_start, dt_end]],bins=(nbins_t,nbins_E, nbins_dt))
    # Create feature
    if norm == 'minmax':
        feature = (hist3D-np.min(hist3D))/(np.max(hist3D)-np.min(hist3D))
    elif norm == 'none':
        feature = hist3D
    # Plot
    if plot == True:
        fig = plt.figure(figsize=(10, 10),constrained_layout = True)
        fig.suptitle(f'ObsID: {obsid}, RegID: {regid}, N: {N_length}, T: {int(T_duration)}s')
        # Plot the E-t projection
        ax1 = fig.add_subplot(2, 2, 1)
        if lognorm == True:
            ax1.imshow(hist3D.sum(axis=2).T, origin='lower', extent=[t_start,t_end, E_start, E_end],cmap=colmap,norm=LogNorm())
        elif lognorm == False:
            ax1.imshow(hist3D.sum(axis=2).T, origin='lower', extent=[t_start,t_end, E_start, E_end],cmap=colmap)
        ax1.set_xlabel(r'$\tau$')
        ax1.set_ylabel(r'$\epsilon$')
        ax1.set_title(r'$\epsilon$ vs $\tau$ Projection')

        # Plot the dt-t projection
        ax2 = fig.add_subplot(2, 2, 2)
        if lognorm == True:
            ax2.imshow(hist3D.sum(axis=1).T, origin='lower', extent=[t_start,t_end, dt_start, dt_end],cmap=colmap,norm=LogNorm())
        elif lognorm == False:
            ax2.imshow(hist3D.sum(axis=1).T, origin='lower', extent=[t_start,t_end, dt_start, dt_end],cmap=colmap)
        ax2.set_xlabel(r'$\tau$')
        ax2.set_ylabel(r'$\delta\tau$')
        ax2.set_title(r'$\delta\tau$ vs $\tau$ Projection')

        # Plot the YZ projection
        ax3 = fig.add_subplot(2, 2, 3)
        if lognorm == True:
            ax3.imshow(hist3D.sum(axis=0), origin='lower', extent=[dt_start,dt_end, E_start, E_end],cmap=colmap,norm=LogNorm())
        elif lognorm == False:
            ax3.imshow(hist3D.sum(axis=0), origin='lower', extent=[dt_start,dt_end, E_start, E_end],cmap=colmap)
        ax3.set_xlabel(r'$\delta\tau$')
        ax3.set_ylabel(r'$\epsilon$')
        ax3.set_title(r'$\epsilon$ vs $\delta\tau$ Projection')

        # Plot 3D Histogram
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        tt, EE, dtdt = np.meshgrid(edges[0][:-1], edges[1][:-1], edges[2][:-1], indexing='ij')
        tt = np.ravel(tt)
        EE = np.ravel(EE)
        dtdt = np.ravel(dtdt)
        h = np.ravel(hist3D)
        if lognorm == True:
            ax4.scatter(dtdt, tt, EE, s=np.log(h*10000), alpha=0.9, edgecolors='none', c=h, cmap=colmap, norm = LogNorm())
        elif lognorm == False:
            ax4.scatter(dtdt, tt, EE, s=np.log(h*10000), alpha=0.9, edgecolors='none', c=h, cmap=colmap)
        ax4.set_xlabel(r'$\delta\tau$')
        ax4.set_ylabel(r'$\tau$')
        ax4.set_zlabel(r'$\epsilon$')
        ax4.set_title('3D Histogram')
        ax4.view_init(elev=30, azim=45)
        ax4.xaxis.set_ticks_position('bottom')
        ax4.yaxis.set_ticks_position('top')
        ax4.zaxis.set_ticks_position('bottom')
        ax4.invert_xaxis()
        plt.show()
    return feature

# Image producer

def hist2D_img(df_eventfile_input, id_name, savefolder, norm = 'none', plot = True, colmap = 'plasma'):
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
    # Freedman-Diaconis rule t
    iqr_E = np.subtract(*np.percentile(df["E"], [75, 25], axis=0)) #IQ range
    binwidth_E = 2 * iqr_E / (len(df["E"]) ** (1/3))
    nbins_E = int(np.ceil((max(df["E"]) - min(df["E"])) / binwidth_E))
    iqr_t = np.subtract(*np.percentile(df["t"], [75, 25], axis=0)) #IQ range
    binwidth_t = 2 * iqr_t / (len(df["t"]) ** (1/3))
    nbins_t = int(np.ceil((max(df["t"]) - min(df["t"])) / binwidth_t))
    # Add Et histogram
    hist_Et = np.histogram2d(df["t"],df["E"],range = [[t_start,t_end],[E_start, E_end]],bins=(nbins_t,nbins_E)) 
    # Create feature
    if norm == 'minmax':
        feature = (hist_Et[0]-np.min(hist_Et[0]))/(np.max(hist_Et[0])-np.min(hist_Et[0]))
    elif norm == 'none':
        feature = hist_Et[0]

    if plot == True:
        plt.imshow(feature.T, origin='lower', extent=[t_start,t_end, E_start, E_end], cmap=colmap,norm=LogNorm())
        #plt.colorbar()
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$\epsilon$')
        plt.title(f'ObsID: {obsid}, RegID: {regid}, N: {N_length}, T: {int(T_duration)}s')
        plt.show()

    # Save the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(feature.T, origin='lower', extent=[t_start,t_end, E_start, E_end], cmap='viridis', norm=LogNorm())
    # Remove x-axis and y-axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.savefig(f'{savefolder}/{id_name}.png',bbox_inches='tight', dpi=300,pad_inches=0)
    return feature




###################### DRAFTS

def hist3D_2(df_eventfile_input, id_name, nbins_E, nbins_t, nbins_dt, dtscale = 'mm', plot = True):
    # Copy df
    df = df_eventfile_input.copy()
    df.sort_values(by='time', inplace = True) 
    df.reset_index(drop=True, inplace = True)
    # Define histogram boundaries
    E_start = np.log10(500)
    E_end = np.log10(7000)
    t_start = 0
    t_end = 1
    dt_start = 0
    dt_end = 1
    # IDs
    obsid = id_name.split("_")[0]
    regid = id_name.split("_")[1]
    # Eventfile length and duration
    N_length = len(df) 
    T_duration = max(df["time"])-min(df["time"])
    # Add E, t, dt columns
    df["E"] = np.log10(df["energy"])
    df["t"] = (df["time"]-min(df["time"]))/(max(df["time"])-min(df["time"]))
    df["delta_time"] = df['t'].diff().shift(-1)
    df = df[df["delta_time"].notna()]
    if dtscale == 'z':
        df["dt"] = (df['delta_time'] - df['delta_time'].mean()) / df['delta_time'].std() 
        df["dt"] = (df['dt']-min(df['dt']))/(max(df['dt'])-min(df['dt']))
    elif dtscale == 'mm':
        df["dt"] = (df['delta_time']-min(df['delta_time']))/(max(df['delta_time'])-min(df['delta_time']))
    # Add Et histogram
    hist3D, edges = np.histogramdd((df["t"], df["E"], df["dt"]), range = [[t_start,t_end],[E_start, E_end], [dt_start, dt_end]],bins=(nbins_t,nbins_E, nbins_dt))
    # Create feature
    feature = hist3D
    # Plot
    if plot == True:
        fig = plt.figure(figsize=(10, 10),constrained_layout = True)
        fig.suptitle(f'ObsID: {obsid}, RegID: {regid}, N: {N_length}, T: {T_duration}')
        # Plot the E-t projection
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(hist3D.sum(axis=2).T, origin='lower', extent=[t_start,t_end, E_start, E_end],cmap='plasma',norm=LogNorm())
        ax1.set_xlabel('t')
        ax1.set_ylabel('E')
        ax1.set_title('E-t Projection')

        # Plot the dt-t projection
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(hist3D.sum(axis=1).T, origin='lower', extent=[t_start,t_end, dt_start, dt_end],cmap='plasma',norm=LogNorm())
        ax2.set_xlabel('t')
        ax2.set_ylabel('dt')
        ax2.set_title('dt-t Projection')

        # Plot the YZ projection
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(hist3D.sum(axis=0), origin='lower', extent=[dt_start,dt_end, E_start, E_end],cmap='plasma',norm=LogNorm())
        ax3.set_xlabel('dt')
        ax3.set_ylabel('E')
        ax3.set_title('E-dt Projection')

        # Plot 3D Histogram
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        x, y, z = np.meshgrid(edges[0][:-1], edges[2][:-1], edges[1][:-1], indexing='ij')
        x = np.ravel(x)
        y = np.ravel(y)
        z = np.ravel(z)
        h = np.ravel(hist3D)
        ax4.scatter(x, y, z, s=h*50, alpha=0.5, edgecolors='none', c=h, cmap='plasma')
        ax4.set_xlabel('t')
        ax4.set_ylabel('dt')
        ax4.set_zlabel('E')
        ax4.set_title('3D Histogram')
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
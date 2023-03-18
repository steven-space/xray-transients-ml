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
import astropy.stats.bayesian_blocks as bb
# CIAO Imports
# import ciao_contrib.runtool
# from ciao_contrib.runtool import *

# Define Custom Functions

# Edt binning plotter
def binning_plotter(bin_list,show_percentiles = True, xlim =[0,100],nbins = 100, farbe = 'schwarz',title = 'XXX'):
    # Define Colour Scheme
    google_blue = '#4285F4'
    google_red = '#DB4437'
    google_yellow = '#F4B400'
    google_green = '#0F9D58'
    google_purple = '#6f2da8'
    if farbe == 'blau':
        colour = google_blue
    elif farbe == 'rot':
        colour = google_red
    elif farbe == 'gelb':
        colour = google_yellow
    elif farbe == 'gruen':
        colour = google_green
    elif farbe == 'lila':
        colour = google_purple
    elif farbe == 'schwarz':
        colour = 'black'
    # Define Font Settings
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['font.family'] = 'sans-serif'
    # Create subplots 
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    # Plot Binning without percentiles
    axs[0].hist(bin_list, color = colour,bins=nbins)
    axs[0].set_xlabel('Optimal Number of Bins')
    axs[0].set_ylabel('Counts')
    axs[0].set_title(title)
    axs[0].set_xlim(xlim)
    # Plot delta_time Binning histograms
    axs[1].hist(bin_list, color = colour,bins=nbins)
    axs[1].set_xlabel('Optimal Number of Bins')
    axs[1].set_ylabel('Counts')
    axs[1].set_title(title)
    axs[1].set_xlim(xlim)
    # Plot Percentiles
    if show_percentiles:
        # Energy statistical values
        n_avg = int(np.ceil(sum(bin_list)/len(bin_list)))
        n_99 = int(np.percentile(bin_list, 99))
        n_95 = int(np.percentile(bin_list, 95))
        n_90 = int(np.percentile(bin_list, 90))
        n_75 = int(np.percentile(bin_list, 75))
        n_50 =  int(np.percentile(bin_list, 50))
        # Plot
        textsize = 8
        axs[1].axvline(n_avg,color='black',linestyle='-')
        plt.text(n_avg+0.3, .85, f'Mean\n{n_avg}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        # axs[1].axvline(n_99,color='black',linestyle='--')
        # plt.text(n_99+0.3, .4, f'99%\n{n_99}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(n_95,color='black',linestyle=':')
        plt.text(n_95+0.3, .40, f'95%\n{n_95}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(n_90,color='black',linestyle='-.')
        plt.text(n_90+0.3, .55, f'90%\n{n_90}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(n_75,color='black',linestyle='--')
        plt.text(n_75+0.3, .7, f'75%\n{n_75}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        # axs[1].axvline(n_75,color='black',linestyle=':')
        # plt.text(n_75+0.3, .7, f'75%\n{n_75}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
    
    return

# t binning plotter
def t_binning_plotter(N_list,T_list,show_percentiles = False,xlim_N = [0,800],xlim_T = [0,180000], farbe1 = 'schwarz', farbe2 = 'schwarz'):
    # Define Colour Scheme
    google_blue = '#4285F4'
    google_red = '#DB4437'
    google_yellow = '#F4B400'
    google_green = '#0F9D58'
    google_purple = '#6f2da8'
    if farbe1 == 'blau':
        colour1 = google_blue
    elif farbe1 == 'rot':
        colour1 = google_red
    elif farbe1 == 'gelb':
        colour1 = google_yellow
    elif farbe1 == 'gruen':
        colour1 = google_green
    elif farbe1 == 'lila':
        colour1 = google_purple
    elif farbe1 == 'schwarz':
        colour1 = 'black'

    if farbe2 == 'blau':
        colour2 = google_blue
    elif farbe2 == 'rot':
        colour2 = google_red
    elif farbe2 == 'gelb':
        colour2 = google_yellow
    elif farbe2 == 'gruen':
        colour2 = google_green
    elif farbe2 == 'lila':
        colour2 = google_purple
    elif farbe2 == 'schwarz':
        colour2 = 'black'
    # Define Font Settings
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['font.family'] = 'sans-serif'

    # Freedman-Diaconis rule N
    iqr_N = np.subtract(*np.percentile(N_list, [75, 25], axis=0)) #IQ range
    binwidth_N = 2 * iqr_N / (len(N_list) ** (1/3))
    nbins_N = int(np.ceil((max(N_list) - min(N_list)) / binwidth_N))
    # Freedman-Diaconis rule T
    iqr_T = np.subtract(*np.percentile(T_list, [75, 25], axis=0)) #IQ range
    binwidth_T = 2 * iqr_T / (len(T_list) ** (1/3))
    nbins_T = int(np.ceil((max(T_list) - min(T_list)) / binwidth_T))
    
    # Create subplots 
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    # Plot Energy Binning histograms
    axs[0].hist(N_list, color = colour1 ,bins = nbins_N)
    axs[0].set_xlabel(r'Eventfile Length')
    axs[0].set_ylabel('Counts')
    axs[0].set_title(r'Eventfile Length Distribution')
    axs[0].set_xlim(xlim_N)
    # Plot delta_time Binning histograms
    axs[1].hist(T_list, color = colour2 ,bins = nbins_T)
    axs[1].set_xlabel(r'Eventfile Duration')
    axs[1].set_ylabel('Counts')
    axs[1].set_title(r'Eventfile Duration Distribution')
    axs[1].set_xlim(xlim_T)
    # Plot Percentiles
    textsize = 8
    N_avg = int(np.ceil(sum(N_list)/len(N_list)))
    T_avg = int(np.ceil(sum(T_list)/len(T_list)))
    axs[0].axvline(N_avg,color='black',linestyle='-')
    plt.text(N_avg+10, .85, f'Mean\n{N_avg}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)
    axs[1].axvline(T_avg,color='black',linestyle='-')
    plt.text(T_avg+2000, .85, f'Mean\n{T_avg}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
    
    if show_percentiles:
        # Energy statistical values
        N_avg = int(np.ceil(sum(N_list)/len(N_list)))
        N_99 = int(np.percentile(N_list, 99))
        N_95 = int(np.percentile(N_list, 95))
        N_90 = int(np.percentile(N_list, 90))
        N_75 = int(np.percentile(N_list, 75))
        N_50 =  int(np.percentile(N_list, 50))
        N_25 =  int(np.percentile(N_list, 25))
        # dt statistical values
        T_avg = int(np.ceil(sum(T_list)/len(T_list)))
        T_99 = int(np.percentile(T_list, 99))
        T_95 = int(np.percentile(T_list, 95))
        T_90 = int(np.percentile(T_list, 90))
        T_75 = int(np.percentile(T_list, 75))
        T_50 = int(np.percentile(T_list, 50))
        T_25 = int(np.percentile(T_list, 25))
        # Plot
        textsize = 8
        #axs[0].axvline(N_99,color='black',linestyle='--')
        #plt.text(N_99+10, .90, f'99%\n{N_99}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)
        axs[0].axvline(N_95,color='black',linestyle=':')
        plt.text(N_95+10, .40, f'95%\n{N_95}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)
        axs[0].axvline(N_90,color='black',linestyle='-.')
        plt.text(N_90+10, .55, f'90%\n{N_90}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)
        axs[0].axvline(N_75,color='black',linestyle='--')
        plt.text(N_75+10, .7, f'75%\n{N_75}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)

        # axs[0].axvline(N_50,color='black',linestyle=':')
        # plt.text(N_50+10, .90, f'50%\n{N_50}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)
        # axs[0].axvline(N_25,color='black',linestyle=':')
        # plt.text(N_25+10, .90, f'25%\n{N_25}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)

        #axs[1].axvline(T_99,color='black',linestyle='--')
        #plt.text(T_99+2000, .90, f'99%\n{T_99}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(T_95,color='black',linestyle=':')
        plt.text(T_95+2000, .40, f'95%\n{T_95}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(T_90,color='black',linestyle='-.')
        plt.text(T_90+2000, .55, f'90%\n{T_90}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(T_75,color='black',linestyle='--')
        plt.text(T_75+2000, .7, f'75%\n{T_75}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)

        # axs[1].axvline(T_50,color='black',linestyle=':')
        # plt.text(T_50+2000, .90, f'50%\n{T_50}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        # axs[1].axvline(T_25,color='black',linestyle=':')
        # plt.text(T_25+2000, .90, f'25%\n{T_25}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
    return



# 6. Lightcurve Plotter Function
def lc_plotter_fun(df_eventfiles_input,id_name,bin_size_sec):
    """
    DESCRIPTION: Plots lightcurves and cumulative counts for given eventfile input dataframe
    INPUT: 1. Original eventfile table, 2. Original properties table, 3. Global Path, 4. Set Name
    OUTPUT: 1. Reduced eventfile table, 2. Reduced properties table
    """
    # Define Colour Scheme
    google_blue = '#4285F4'
    google_red = '#DB4437'
    google_yellow = '#F4B400'
    google_green = '#0F9D58'
    # Define Font Settings
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['font.family'] = 'sans-serif'
    # Create subplots 
    fig, axs = plt.subplots(2, 2, figsize=(12, 6),constrained_layout = True)
    fig.suptitle(f'ObsRegID: {id_name}',fontweight="bold")
    # Create binned lightcurve
    df = df_eventfiles_input.copy()
    df['time'] = df_eventfiles_input['time'] - min(df_eventfiles_input['time'])
    df = df.sort_values(by='time') 
    df = df.reset_index(drop=True)
    df_binned = df.groupby(df['time'] // bin_size_sec * bin_size_sec).count()
    # Plot binned lightcurve
    axs[0,0].plot(df_binned.index/1000, df_binned, color = google_blue)
    axs[0,0].set_xlabel('Time [ks]')
    axs[0,0].set_ylabel('Counts per Bin')
    axs[0,0].set_title(f'Lightcurve with {bin_size_sec}s Bin Size')
    # Create rolling 3-bin averaged lightcurved
    df_rolling = df_binned.rolling(window=3, center=True).mean()
    rolling_std = df_binned.rolling(window=3, center=True).std()
    errors = rolling_std['time']/math.sqrt(3)
    errors.iloc[0] = errors.iloc[0] * math.sqrt(3)/math.sqrt(2)
    errors.iloc[-1] = errors.iloc[-1] * math.sqrt(3)/math.sqrt(2)
    # Plot rolling 3-bin averaged lightcurved
    axs[1,0].plot(df_rolling.index/1000, df_rolling, color = google_red)
    axs[1,0].errorbar(df_rolling.index/1000, df_rolling['time'], yerr = errors, xerr = None,fmt ='.',color = "black",linewidth = .5,capsize = 1)
    axs[1,0].set_xlabel('Time [ks]')
    axs[1,0].set_ylabel('Counts per Bin')
    axs[1,0].set_title('Running Average of 3 Bins')
    # Create cumulative count plot
    df_cumulative = df.copy()
    df_cumulative['count'] = 1
    df_cumulative['cumulative_count'] = df_cumulative['count'].cumsum()
    # Plot cumulative count plot
    axs[0,1].plot(df_cumulative['time']/1000, df_cumulative['cumulative_count'],color = google_green)
    axs[0,1].set_xlabel('Time [ks]')
    axs[0,1].set_ylabel('Cumulative Count')
    axs[0,1].set_title('Cumulative Count over Time')
    # Create normalized gradient plot
    df_grad = df_eventfiles_input.copy()
    df_grad['time'] = df_eventfiles_input['time'] - min(df_eventfiles_input['time'])
    df_grad = df_grad.sort_values(by='time') 
    df_grad = df_grad.reset_index(drop=True)
    df_grad_bin =  df_grad.groupby(df_grad['time'] // bin_size_sec * bin_size_sec).count().cumsum()
    df_grad_bin = df_grad_bin.rename(columns={'time': 'count'})
    df_grad_bin = df_grad_bin[['count']].reset_index()
    max_time = df_grad_bin['time'].max()
    min_time = df_grad_bin['time'].min()
    max_count = df_grad_bin['count'].max()
    min_count = df_grad_bin['count'].min()
    df_grad_bin['time_norm'] = (df_grad_bin['time']- min_time)/(max_time - min_time)
    df_grad_bin['cumulative_count_norm'] = (df_grad_bin['count']-min_count)/(max_count - min_count)
    gradient = np.gradient(df_grad_bin['cumulative_count_norm'], df_grad_bin['time_norm'])
    norm_times = df_grad_bin['time_norm']
    avg = np.mean(gradient)
    lower = avg*1.5
    upper = avg*0.5
    x_high = norm_times[gradient > upper]
    y_high = gradient[gradient > upper]
    x_low = norm_times[gradient < lower]
    y_low = gradient[gradient < lower]
    # Plot normalized gradient plot
    axs[1,1].scatter(norm_times, gradient, color = google_yellow)
    axs[1,1].scatter(x_high, y_high, color = 'black')
    axs[1,1].scatter(x_low, y_low, color = 'black')
    axs[1,1].set_xlabel('Normalised Time')
    axs[1,1].set_ylabel('Normalised Gradient')
    axs[1,1].set_title('Normalised Gradient over Time')
    axs[1,1].set(xlim=(0, 1), ylim=(0, 2*avg))
    axs[1,1].axhline(y=avg,linestyle='-',color='black')
    axs[1,1].axhline(y=upper,linestyle=':',color='black')
    axs[1,1].axhline(y=lower,linestyle=':',color='black')

    plt.show()
    return


# 6. Lightcurve Plotter Function
def lc_plotter_fun_2(df_eventfiles_input,id_name,bin_size_sec):
    """
    DESCRIPTION: Plots lightcurves and cumulative counts for given eventfile input dataframe
    INPUT: 1. Original eventfile table, 2. Original properties table, 3. Global Path, 4. Set Name
    OUTPUT: 1. Reduced eventfile table, 2. Reduced properties table
    """
    # Define Colour Scheme
    google_blue = '#4285F4'
    google_red = '#DB4437'
    google_yellow = '#F4B400'
    google_green = '#0F9D58'
    # Define Font Settings
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['font.family'] = 'sans-serif'
    # Create subplots 
    fig, axs = plt.subplots(3, 1, figsize=(6, 9),constrained_layout = True)
    fig.suptitle(f'ObsRegID: {id_name}',fontweight="bold")
    # Create binned lightcurve
    df = df_eventfiles_input.copy()
    df['time'] = df_eventfiles_input['time'] - min(df_eventfiles_input['time'])
    df = df.sort_values(by='time') 
    df = df.reset_index(drop=True)
    df_binned = df.groupby(df['time'] // bin_size_sec * bin_size_sec).count()
    # Plot binned lightcurve
    axs[0].plot(df_binned.index/1000, df_binned, color = google_blue)
    axs[0].set_xlabel('Time [ks]')
    axs[0].set_ylabel('Counts per Bin')
    axs[0].set_title(f'Lightcurve with {bin_size_sec}s Bin Size')
    # Create rolling 3-bin averaged lightcurved
    df_rolling = df_binned.rolling(window=3, center=True).mean()
    rolling_std = df_binned.rolling(window=3, center=True).std()
    errors = rolling_std['time']/math.sqrt(3)
    errors.iloc[0] = errors.iloc[0] * math.sqrt(3)/math.sqrt(2)
    errors.iloc[-1] = errors.iloc[-1] * math.sqrt(3)/math.sqrt(2)
    # Plot rolling 3-bin averaged lightcurved
    axs[1].plot(df_rolling.index/1000, df_rolling, color = google_red)
    axs[1].errorbar(df_rolling.index/1000, df_rolling['time'], yerr = errors, xerr = None,fmt ='.',color = "black",linewidth = .5,capsize = 1)
    axs[1].set_xlabel('Time [ks]')
    axs[1].set_ylabel('Counts per Bin')
    axs[1].set_title('Running Average of 3 Bins')
    # Create cumulative count plot
    df_cumulative = df.copy()
    df_cumulative['count'] = 1
    df_cumulative['cumulative_count'] = df_cumulative['count'].cumsum()
    # Plot cumulative count plot
    axs[2].plot(df_cumulative['time']/1000, df_cumulative['cumulative_count'],color = google_green)
    axs[2].set_xlabel('Time [ks]')
    axs[2].set_ylabel('Cumulative Count')
    axs[2].set_title('Cumulative Count over Time')

    plt.show()
    return


# CSC Products Plot
def cscproducts_plots_fun(df_eventfiles_input,id_name):
    try:
        # IDs
        obsid = id_name.split("_")[0]
        regid = id_name.split("_")[1]
        # Lightcurve
        plt.subplots(figsize=(15, 3))
        lc_filename = [lcurve for lcurve in glob.iglob(f'{global_path}/{set_id}/eventdata/acisf*lc3.fits.gz') if str(obsid) in lcurve and str(regid) in lcurve][0]
        pha_filename = [spec for spec in glob.iglob(f'{global_path}/{set_id}/eventdata/acisf*pha3.fits.gz') if str(obsid) in spec and str(regid) in spec][0]
        img_filename = [imag for imag in glob.iglob(f'{global_path}/{set_id}/eventdata/acisf*regimg3.fits.gz') if str(obsid) in imag and str(regid) in imag][0]

        
        with fits.open(lc_filename) as hdul_lc:
            lc3 = hdul_lc[1].data
            bg3 = hdul_lc[2].data
            plt.plot(lc3['Time'],lc3['COUNT_RATE'])
            plt.xlabel('Time [s]')
            plt.ylabel('Count rate [counts/s]')
            plt.title(f'ObsID: {obsid}, RegID: {regid}')
            plt.errorbar(lc3['Time'],lc3['COUNT_RATE'],lc3['COUNT_RATE_ERR'])
            plt.show()

        ui.load_pha(pha_filename)
        ui.ignore('0.:0.5,8.0:')
        ui.subtract()
        ui.notice_id(1,0.3,7.)
        ui.group_counts(10)
        ui.set_ylog()
        ui.set_xlog()
        ui.plot_data()
        plt.xlim(1E-1,10)  
        plt.title(f'ObsID: {obsid}, RegID: {regid}')
        plt.show()

        with fits.open(img_filename) as hdul_img:
            img3 = hdul_img[0].data
            plt.imshow(img3, cmap='gray')
            #plt.colorbar()
            plt.title(f'ObsID: {obsid}, RegID: {regid}')
            plt.show()

        
    except: 
        print(f'Failed: {id_name}')
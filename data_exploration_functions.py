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

# 1. List Folder Function
def list_folders_fun(path):
    """
    DESCRIPTION: List folders in a given directory.
    INPUT: Directory path
    OUTPUT: Folder names in a given directory
    """
    folder_list = [f.name for f in Path(path).iterdir() if f.is_dir()]
    return folder_list
    
# 2. Region Filter Function
def region_filter_fun(global_path,set_id):
    """
    DESCRIPTION: Filters eventfiles in a directory with regionfiles and stores filtered files in the same directory.
    INPUT: Directory path
    OUTPUT: Filtered eventfiles
    """
    # Loop over eventdata
    for event_filename in glob.iglob(f'{global_path}/{set_id}/eventdata/acisf*regevt3.fits.gz'):
        # Get ObsID
        obsid = int(event_filename.split('_')[0][-5:])
        # Get RegionID
        try: 
            regionid = int(event_filename.split('_')[2][-4:])
        except: 
            regionid = int(event_filename.split('_')[3][-4:]) 
        # Filter eventfiles with regionfiles and store filtered file
        region_filename = [region for region in glob.iglob(f'{global_path}/{set_id}/eventdata/acisf*reg3.fits.gz') if str(obsid) in region and str(regionid) in region][0]
        filtered_filename = event_filename.replace(".fits", "_filtered.fits")
        try:
            ciao_contrib.runtool.dmcopy(f'{event_filename}[sky=region({region_filename})]', filtered_filename)
        except OSError: 
            print(f'{filtered_filename} already exists!')
    return 

# 3. Create Eventfile Table Function
def create_eventfilestable_fun(global_path,set_id):
    """
    DESCRIPTION: Creates a dataframe (saved as csv) of filtered eventfiles including the following additional filters: GTI filters, 'pha'>40, 'grade'>=0, 'energy'>500, 'energy'<7000
    INPUT: 1. Global Path, 2. Set Name including filtered eventfiles
    OUTPUT: Dataframe of filtered eventfiles
    """
    # Initialise dataframe list of all eventfiles
    list_df_events = []
    # Loop over all eventfiles
    for filename in glob.iglob(f'{global_path}/{set_id}/eventdata/acisf*regevt*filtered*gz'):
        with fits.open(filename) as hdul:
            # Events dataframe
            events = hdul["Events"].data
            events_table = Table(events)
            events_cols = events.columns.names
            df_events = pd.DataFrame.from_records(events_table, columns=events_cols)
            df_events = df_events.sort_values(by=["time"])
            # GTI (Good Time Interval) dataframe
            gti = hdul["GTI"].data
            gti_table = Table(gti)
            gti_cols = gti.columns.names
            df_gti = pd.DataFrame.from_records(gti_table, columns=gti_cols)
            # Apply GTI filter to events
            gti_mask = np.zeros(len(df_events), dtype=bool)
            for i in range(len(df_gti)):
                start = df_gti.iloc[i]['START']
                stop = df_gti.iloc[i]['STOP']
                gti_mask |= (df_events["time"] >= start) & (df_events["time"] < stop)
            df_events = df_events[gti_mask]
            # Apply energy, pha, grade filters to events
            df_events = df_events[(df_events['pha']>40) & (df_events['grade']>=0) & (df_events['energy']>500) & (df_events['energy']<7000)]
            # Add obsid and region_id column (from filename)
            df_events["obsid"] = int(filename.split('_')[0][-5:])
            try: 
                df_events["region_id"] = int(filename.split('_')[2][-4:]) #need to add try except while looping 
            except: 
                df_events["region_id"] = int(filename.split('_')[3][-4:]) 
            # Append to dataframe list
            list_df_events.append(df_events)
    # Combine dfs in dataframe list into one df and save in folder
    df_eventfiles = pd.concat(list_df_events)
    df_eventfiles.to_csv(f'{global_path}/{set_id}/eventfiles-{set_id}.csv',index=False)
    return df_eventfiles
    
# 4. Data Reduction Function
def data_reduction_fun(df_eventfiles,df_properties,global_path,set_id, unique_ids = True, min_counts = 0):
    """
    DESCRIPTION: Reduces evenfiles table and properties table to required columns and adds unique ID, can now be used for data representation function
    INPUT: 1. Original eventfile table, 2. Original properties table, 3. Global Path, 4. Set Name
    OUTPUT: 1. Reduced eventfile table, 2. Reduced properties table
    """
    # Extract important labels and input columns
    df_eventfiles_input = df_eventfiles[['obsid','region_id','time','energy','chipx','chipy']]
    df_properties_input = df_properties[['obsid','region_id','cnts_aper_b','cnts_aperbkg_b','src_cnts_aper_b','flux_aper_b','hard_hm','hard_hs','hard_ms','var_prob_b','var_prob_h','var_prob_m','var_prob_s']]
    # Create unique IDs
    df_eventfiles_input['obsreg_id'] = df_eventfiles_input['obsid'].astype(str) + '_' + df_eventfiles_input['region_id'].astype(str)
    df_properties_input['obsreg_id'] = df_properties_input['obsid'].astype(str) + '_' + df_properties_input['region_id'].astype(str)
    # Drop individual IDs
    df_eventfiles_input = df_eventfiles_input.drop(columns=['obsid', 'region_id'])
    df_properties_input = df_properties_input.drop(columns=['obsid', 'region_id'])
    # Filter observations where there are less than min_counts counts
    counts = df_eventfiles_input['obsreg_id'].value_counts()
    count_mask = df_eventfiles_input['obsreg_id'].isin(counts[counts >= min_counts].index)
    df_eventfiles_input = df_eventfiles_input[count_mask]
    # Unique ID combinations filter
    if unique_ids:
        df_eventfiles_input = df_eventfiles_input[df_eventfiles_input['obsreg_id'].isin(df_properties_input['obsreg_id'].unique())]
        df_properties_input = df_properties_input[df_properties_input['obsreg_id'].isin(df_eventfiles_input['obsreg_id'].unique())]
    # Save new dataframes
    df_eventfiles_input.to_csv(f'{global_path}/{set_id}/eventfiles-input-{set_id}.csv',index=False)
    df_properties_input.to_csv(f'{global_path}/{set_id}/properties-input-{set_id}.csv',index=False)
    return df_eventfiles_input, df_properties_input


# 5. Edt binning plotter
def Edt_binning_plotter(nbin_E_list,nbin_dt_list,binning_rule = 'BINNING RULE NAME',show_percentiles = True, xlim_E=[0,100],xlim_dt=[0,100]):
    # Define Colour Scheme
    google_blue = '#4285F4'
    google_red = '#DB4437'
    google_yellow = '#F4B400'
    google_green = '#0F9D58'
    # Define Font Settings
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['font.family'] = 'sans-serif'

    # Freedman-Diaconis rule ENERGY BINS
    iqr_E = np.subtract(*np.percentile(nbin_E_list, [75, 25], axis=0)) #IQ range
    binwidth_E = 2 * iqr_E / (len(nbin_E_list) ** (1/3))
    nbins_E = int(np.ceil((max(nbin_E_list) - min(nbin_E_list)) / binwidth_E))
    # Freedman-Diaconis rule DT BINS
    iqr_dt = np.subtract(*np.percentile(nbin_dt_list, [75, 25], axis=0)) #IQ range
    binwidth_dt = 2 * iqr_dt / (len(nbin_dt_list) ** (1/3))
    nbins_dt = int(np.ceil((max(nbin_dt_list) - min(nbin_dt_list)) / binwidth_dt))
    
    # Create subplots 
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    fig.suptitle(f'{binning_rule}')
    # Plot Energy Binning histograms
    axs[0].hist(nbin_E_list, color = google_blue,bins=nbins_E)
    axs[0].set_xlabel('Optimal Number of Bins')
    axs[0].set_ylabel('Counts')
    axs[0].set_title('Energy Binning')
    axs[0].set_xlim(xlim_E)
    # Plot delta_time Binning histograms
    axs[1].hist(nbin_dt_list, color = google_red,bins=nbins_dt)
    axs[1].set_xlabel('Optimal Number of Bins')
    axs[1].set_ylabel('Counts')
    axs[1].set_title('dt Binning')
    axs[1].set_xlim(xlim_dt)
    # Plot Percentiles
    if show_percentiles:
        # Energy statistical values
        E_avg = int(np.ceil(sum(nbin_E_list)/len(nbin_E_list)))
        E_99 = int(np.percentile(nbin_E_list, 99))
        E_95 = int(np.percentile(nbin_E_list, 95))
        E_90 = int(np.percentile(nbin_E_list, 90))
        E_75 = int(np.percentile(nbin_E_list, 75))
        E_50 =  int(np.percentile(nbin_E_list, 50))
        # dt statistical values
        dt_avg = int(np.ceil(sum(nbin_dt_list)/len(nbin_dt_list)))
        dt_99 = int(np.percentile(nbin_dt_list, 99))
        dt_95 = int(np.percentile(nbin_dt_list, 95))
        dt_90 = int(np.percentile(nbin_dt_list, 90))
        dt_75 = int(np.percentile(nbin_dt_list, 75))
        dt_50 = int(np.percentile(nbin_dt_list, 50))
        # Plot
        textsize = 8
        axs[0].axvline(E_95,color='black',linestyle='--')
        plt.text(E_95+0.3, .55, f'95%\n{E_95}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)
        axs[0].axvline(E_90,color='black',linestyle='-.')
        plt.text(E_90+0.3, .7, f'90%\n{E_90}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)
        axs[0].axvline(E_75,color='black',linestyle=':')
        plt.text(E_75+0.3, .85, f'75%\n{E_75}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(dt_95,color='black',linestyle='--')
        plt.text(dt_95+0.3, .55, f'95%\n{dt_95}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(dt_90,color='black',linestyle='-.')
        plt.text(dt_90+0.3, .7, f'90%\n{dt_90}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(dt_75,color='black',linestyle=':')
        plt.text(dt_75+0.3, .85, f'75%\n{dt_75}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
   
    return

# 6. t binning plotter
def t_binning_plotter(N_list,T_list,show_percentiles = True,xlim_N = [0,800],xlim_T = [0,180000]):
    # Define Colour Scheme
    google_blue = '#4285F4'
    google_red = '#DB4437'
    google_yellow = '#F4B400'
    google_green = '#0F9D58'
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
    axs[0].hist(N_list, color = google_green,bins = nbins_N)
    axs[0].set_xlabel('Eventfile Length')
    axs[0].set_ylabel('Counts')
    axs[0].set_title('Eventfile Length Distribution')
    axs[0].set_xlim(xlim_N)
    # Plot delta_time Binning histograms
    axs[1].hist(T_list, color = google_yellow,bins = nbins_T)
    axs[1].set_xlabel('Eventfile Duration')
    axs[1].set_ylabel('Counts')
    axs[1].set_title('Eventfile Duration Distribution')
    axs[1].set_xlim(xlim_T)
    # Plot Percentiles
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
        axs[0].axvline(N_95,color='black',linestyle='--')
        plt.text(N_95+10, .90, f'95%\n{N_95}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)
        axs[0].axvline(N_90,color='black',linestyle='-.')
        plt.text(N_90+10, .90, f'90%\n{N_90}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)
        axs[0].axvline(N_75,color='black',linestyle=':')
        plt.text(N_75+10, .90, f'75%\n{N_75}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)


        axs[0].axvline(N_50,color='black',linestyle=':')
        plt.text(N_50+10, .90, f'50%\n{N_50}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)
        axs[0].axvline(N_25,color='black',linestyle=':')
        plt.text(N_25+10, .90, f'25%\n{N_25}', transform=axs[0].get_xaxis_transform(),fontsize=textsize)



        #axs[1].axvline(T_99,color='black',linestyle='--')
        #plt.text(T_99+2000, .90, f'99%\n{T_99}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(T_95,color='black',linestyle='--')
        plt.text(T_95+2000, .90, f'95%\n{T_95}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(T_90,color='black',linestyle='-.')
        plt.text(T_90+2000, .90, f'90%\n{T_90}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(T_75,color='black',linestyle=':')
        plt.text(T_75+2000, .90, f'75%\n{T_75}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
   
        axs[1].axvline(T_50,color='black',linestyle=':')
        plt.text(T_50+2000, .90, f'50%\n{T_50}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
        axs[1].axvline(T_25,color='black',linestyle=':')
        plt.text(T_25+2000, .90, f'25%\n{T_25}', transform=axs[1].get_xaxis_transform(),fontsize=textsize)
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
    # Create normalized cumulative count plot
    max_time = df_cumulative['time'].max()
    min_time = df_cumulative['time'].min()
    max_count = df_cumulative['cumulative_count'].max()
    min_count = df_cumulative['cumulative_count'].min()
    df_cumulative['time_norm'] = df_cumulative['time']/max_time
    df_cumulative['cumulative_count_norm'] = df_cumulative['cumulative_count']/max_count
    df_cumulative['gradient'] = np.gradient(df_cumulative['cumulative_count'], df_cumulative['time'])
    df_cumulative['gradient_norm'] = np.gradient(df_cumulative['cumulative_count_norm'], df_cumulative['time_norm'])
    df_cumu_bin = df_cumulative.copy()
     # Show the plot
    plt.show()
    # df_cumu_bin = df_cumulative.groupby(df_cumulative['time_norm'] // 1000 * 1000).mean()
    # axs[1,1].scatter(df_cumu_bin.index/1000, df_cumu_bin['gradient_norm'], color = google_yellow)
    axs[1,1].scatter(df_cumulative.index/1000, df_cumulative['gradient_norm'], color = google_yellow)
    # axs[1,1].plot(df_cumulative['time_norm'], df_cumulative['gradient'])
    # axs[1,1].set_xlabel('Time')
    # axs[1,1].set_ylabel('Gradient')

    # Plot normalized cumulative count plot
    # parx = axs[0,1].twiny()
    # pary = axs[0,1].twinx()
    # axs[0,1].set_xlim(min_time/1000, max_time/1000)
    # axs[0,1].set_ylim(min_count, max_count)
    # parx.set_xlim(0, 1)
    # pary.set_ylim(0, 1)
    return
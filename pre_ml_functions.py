# PYTHON Imports 
import pandas as pd
import numpy as np
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


def list_folders_fun(path):
    """
    DESCRIPTION: List folders in a given directory.
    INPUT: Directory path
    OUTPUT: Folder names in a given directory
    """
    folder_list = [f.name for f in Path(path).iterdir() if f.is_dir()]
    return folder_list
    
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
    
    
def data_reduction_fun(df_eventfiles,df_properties,global_path,set_id):
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
    # Filter to same size
    df_eventfiles_input = df_eventfiles_input[df_eventfiles_input['obsreg_id'].isin(df_properties_input['obsreg_id'].unique())]
    df_properties_input = df_properties_input[df_properties_input['obsreg_id'].isin(df_eventfiles_input['obsreg_id'].unique())]
    # Save new dataframes
    df_eventfiles_input.to_csv(f'{global_path}/{set_id}/eventfiles-input-{set_id}.csv',index=False)
    df_properties_input.to_csv(f'{global_path}/{set_id}/properties-input-{set_id}.csv',index=False)
    return df_eventfiles_input, df_properties_input






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
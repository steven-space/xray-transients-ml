from crypt import methods
from logging.handlers import BaseRotatingHandler
from pyexpat import ErrorString
from tkinter import N
from tokenize import PlainToken
from ciao_contrib.runtool import *
import os
import glob
import numpy as np
import sys
from re import sub
import subprocess
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
import numpy as np
import random
from scipy.stats import entropy
from astropy.stats import bayesian_blocks
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from matplotlib import axes
from astropy.stats import histogram
from statistics import mean, median, median_high
from scipy.stats import norm
from scipy import interpolate
from numpy import trapz
import astropy.stats as astats

def behr_hr(area,id,binning,conf = '68.00'):
    BEHR_DIR = '/Users/steven/Desktop/BBBB/BEHR'
    DATA_DIR = '/Users/steven/Desktop/CSCData/23022need/'
    evt_file = glob.glob(f'{DATA_DIR}*evt6*')[0]
    regevt = glob.glob(f'{DATA_DIR}*evt6_filtered*')[0]
    src_region = glob.glob(f'{DATA_DIR}*reg6*')[0]
    bkg_region = glob.glob(f'{DATA_DIR}*bkg6*')[0]
    bevt = glob.glob(f'{DATA_DIR}*evt6_back*')[0]

    with fits.open(regevt) as hdul:
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
        time = df_events.time.values[20:]
        start_time = min(time)
        src_times = time - start_time
        src_energies = df_events.energy.values[20:]

    with fits.open(bevt) as hdul:
        # Events dataframe
        bevents = hdul["Events"].data
        bevents_table = Table(bevents)
        bevents_cols = bevents.columns.names
        df_bevents = pd.DataFrame.from_records(bevents_table, columns=bevents_cols)
        df_bevents = df_bevents.sort_values(by=["time"])
        # GTI (Good Time Interval) dataframe
        bgti = hdul["GTI"].data
        bgti_table = Table(bgti)
        bgti_cols = bgti.columns.names
        df_bgti = pd.DataFrame.from_records(bgti_table, columns=bgti_cols)
        # Apply GTI filter to events
        bgti_mask = np.zeros(len(df_bevents), dtype=bool)
        for i in range(len(df_bgti)):
            start = df_bgti.iloc[i]['START']
            stop = df_bgti.iloc[i]['STOP']
            bgti_mask |= (df_bevents["time"] >= start) & (df_bevents["time"] < stop)
        df_bevents = df_bevents[bgti_mask]
        # Apply energy, pha, grade filters to events
        df_bevents = df_bevents[(df_bevents['pha']>40) & (df_bevents['grade']>=0) & (df_bevents['energy']>500) & (df_bevents['energy']<7000)]
        bkg_times = df_bevents.time.values - start_time
        bkg_energies = df_bevents.energy.values

    if binning == 'bb':
        bb_bins = astats.bayesian_blocks(src_times, fitness='events',p0 = 0.01) # p0 = 0.1
        bin_widths = bb_bins[1:] - bb_bins[:-1]
        counts, _ = np.histogram(src_times, bins=bb_bins)
        time_bin = (bb_bins[:-1] + bb_bins[1:]) / 2
    elif binning == 'fixed':
        bin_size = 500
        num_bins = int(src_times[-1] / bin_size) + 1
        binned_counts, bb_bins = np.histogram(src_times, bins=num_bins)
        time_bin = (bb_bins[:-1] + bb_bins[1:]) / 2
    elif binning == 'counts':
        count_per_bin = 12
        counts = np.ones(len(src_times))
        num_bins = len(src_times) // count_per_bin
        bb_bins = [src_times[i * count_per_bin] for i in range(num_bins)]
        bb_bins.append(src_times[-1])
        bb_bins = np.array(bb_bins)
        time_bin = (bb_bins[:-1] + bb_bins[1:]) / 2

    times_and_energies = np.column_stack((src_times,src_energies))
    bkg_te = np.column_stack((bkg_times,bkg_energies))
    # define soft/medium/hard bands
    soft_filter = times_and_energies[:,1] < 1200
    medium_filter = (times_and_energies[:,1] < 2000) *(times_and_energies[:,1] > 1200)
    hard_filter = times_and_energies[:,1] > 2000
    soft_medium_filter = times_and_energies[:,1] < 2000
    soft_filter_b = bkg_te[:,1] < 1200
    medium_filter_b = (bkg_te[:,1] < 2000) *(bkg_te[:,1] > 1200)
    hard_filter_b = bkg_te[:,1] > 2000
    soft_medium_filter_b = bkg_te[:,1] < 2000
    # generate data for each band
    soft_lc = times_and_energies[soft_filter]
    medium_lc = times_and_energies[medium_filter]
    hard_lc =times_and_energies[hard_filter]
    sm_lc = times_and_energies[soft_medium_filter]
    soft_b = bkg_te[soft_filter_b]
    medium_b = bkg_te[medium_filter_b]
    hard_b = bkg_te[hard_filter_b]
    sm_b = bkg_te[soft_medium_filter_b]
    #counts
    counts_s, _ = np.histogram(soft_lc[:,0], bins=   bb_bins)
    counts_m, _ = np.histogram(medium_lc[:,0], bins=   bb_bins)
    counts_h, _ = np.histogram(hard_lc[:,0], bins=   bb_bins)
    counts_sm, _ = np.histogram(sm_lc[:,0], bins=   bb_bins)
    counts_s_b, _ = np.histogram(soft_b[:,0], bins=   bb_bins)
    counts_m_b, _ = np.histogram(medium_b[:,0], bins=   bb_bins)
    counts_h_b, _ = np.histogram(hard_b[:,0], bins=   bb_bins)
    counts_sm_b, _ = np.histogram(sm_b[:,0], bins=   bb_bins)

    outfile = f'{DATA_DIR}behr' 
    for i in range(0,len(counts_s)):
        with open(outfile,'w') as writeto:
            writeto.write(f'cd {BEHR_DIR}')
            writeto.write(f'\n echo "softsrc={counts_sm[i]} hardsrc={counts_h[i]}   softbkg={counts_sm_b[i]}   hardbkg={counts_h_b[i]} softarea={area} outputPr=True algo=quad"')
            writeto.write(f'\n./BEHR softsrc={counts_sm[i]} hardsrc={counts_h[i]}   softbkg={counts_sm_b[i]}   hardbkg={counts_h_b[i]}  softarea={area} output={BEHR_DIR}/new/{i}_block_BEHRresults_{id} level={conf}')

        subprocess.run(f'bash {outfile}', shell = True)

    def behr_open(file):
        with open(file,'r') as data:
            contents = data.read()
            line = contents.splitlines()[2].split()
            print("Option: ", line[0])
            med = line[3]
            lower = line[4]
            upper = line[5]
        return med,upper,lower
    
    medians = []
    uppers = []
    lowers = []

    for i in range(0,len(counts_s)):
        file = f'{BEHR_DIR}/new/{i}_block_BEHRresults_{id}.txt'
        med,upper,lower = behr_open(file)
        uppers.append(upper)
        medians.append(med)
        lowers.append(lower)

    uppers = np.array(uppers).astype('float64')
    lowers = np.array(lowers).astype('float64')
    medians = np.array(medians).astype('float64')

    return time_bin, bb_bins, medians, uppers, lowers,    src_times 



# time_bin, bb_bins, medians, uppers, lowers = behr_hr(33.08,id='test',binning='counts',conf = '68.00')
time_bin, bb_bins, medians, uppers, lowers, src_times = behr_hr(33.08,id='test',binning='counts',conf = '68.00')

print("Time Bin:", time_bin)
print("Bin Edges:", bb_bins)
print("Medians:", medians)
print("Uppers:", uppers)
print("Lowers:", lowers)

print(len(src_times))

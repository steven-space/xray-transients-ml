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


def N_plotter(list,xlim = [25,1000],binning = 'fd', colour = 'bright'):
    # Define Font Settings
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Gill Sans'
    # Define Color  Settings
    dark = '#003170'
    bright = '#cfbd62'
    if colour == 'dark':
        col = dark
    elif colour == 'bright':
        col = bright
    # Freedman-Diaconis rule N
    iqr = np.subtract(*np.percentile(list, [75, 25], axis=0)) #IQ range
    binwidth = 2 * iqr / (len(list) ** (1/3))
    nbins = int(np.ceil((max(list) - min(list)) / binwidth))
    # Create subplots 
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    # Plot Energy Binning histograms
    if binning == 'fd':
        axs.hist(list, color = bright,bins = nbins,edgecolor='white',linewidth=0.5)
    else:
        axs.hist(list, color = bright,edgecolor='white',linewidth=0.5)
    axs.set_xlabel(r'Eventfile Length $N$')
    axs.set_xlim(xlim)
    axs.set_ylabel('Counts')
    axs.minorticks_on()
    axs.tick_params(which='both', direction='in', top=True, right=True)
    return

def T_plotter(list,xlim = [0,180],binning = 'fd', colour = 'dark'):
    # Define Font Settings
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Gill Sans'
    # Define Color  Settings
    dark = '#003170'
    bright = '#cfbd62'
    if colour == 'dark':
        col = dark
    elif colour == 'bright':
        col = bright
    # Freedman-Diaconis rule N
    iqr = np.subtract(*np.percentile(list, [75, 25], axis=0)) #IQ range
    binwidth = 2 * iqr / (len(list) ** (1/3))
    nbins = int(np.ceil((max(list) - min(list)) / binwidth))
    # Create subplots 
    fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    # Plot Energy Binning histograms
    if binning == 'fd':
        axs.hist(list, color = col,bins = nbins,edgecolor='white',linewidth=0.5)
    else:
        axs.hist(list, color = col,edgecolor='white',linewidth=0.5)
    axs.set_xlabel(r'Eventfile Duration $T$ [ks]')
    axs.set_xlim(xlim)
    axs.set_ylabel('Counts')
    axs.minorticks_on()
    axs.tick_params(which='both', direction='in', top=True, right=True)
    return
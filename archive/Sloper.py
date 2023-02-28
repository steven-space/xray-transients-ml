import numpy as np
import scipy
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import math
import pandas as pd
import random
import statistics
import sys
import os
import glob

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Slope Sampler Algorithm
# Idea: Sample slope every 2ks or so, and look for deviations from average.


## CONSTANTS:
INTERVAL = 1    ## in ks
CUTOFF = 7      ## Average count required for analysis
NUMBER_AHEAD_AND_BEHIND = 1

def Sloper(file,counter,galaxy):
    info = pd.read_table(file, " ")
    print("Analyzing...", file)
    length = info.shape[0]
    times = info["TIME"].tolist()
    initial_time = times[0]
    times = [i - initial_time for i in times]

      ## --------------------------------------------TRIMMING ZEROES ----------------------------------------
    

    count_list = info["COUNTS"].tolist()
    still = True
    start_skip = 0
    end_skip = 0
    #     print(count_list)
    #     print("SHSHSH",len(count_list),len(times))
    if len(count_list) != 0:
        for b in range(0,len(count_list)):
            try:
                if count_list[0] == 0:
                    if still == True:
                        del count_list[0]
                        start_skip +=1 
                if count_list[0] != 0:
                    break
            except:
                # print(count_list)
                print("WEIRD")
        end = True
        for b in range(0,len(count_list)):
            if count_list[-1] == 0:
                if end == True:
                    del count_list[-1]
                    end_skip +=1
            if count_list[-1] != 0:
                break

        if start_skip != 0:
            times = times[start_skip:]
        if end_skip != 0:
            times = times[:-end_skip]

    try:
      low = times[0]
    except:
      return False
    times = [i - low for i in times]

    ## ---------------------------------------------- TOTAL COUNTS PLOT ----------------------------------------------
    cumu_counts = []
    tot_counts = 0
    for row in count_list:
      tot_counts += row
      cumu_counts.append(tot_counts)


    bin_times, binned_list = bin(times,count_list,1000)
    try:
        avgs, errors = running_avg(1,binned_list)
    except:
        print("error in running averages")



    avg_count_rate = (tot_counts * 1000) /times[-1]
    # avg_count_rate = med
    if times[-1] < 10000:
      print("Too low of a count rate to analyze.")
    else:
    ## ---------------------------------------------- SLOPE SAMPLING ----------------------------------------------

      num_points = int( (times[-1]) / (1000*INTERVAL))
      selected = np.linspace(0,len(times),num_points+1,endpoint=False)
      selected = [int(i) for i in selected]

      multiplier = times[-1]/tot_counts

      new_cumu_counts = [i*multiplier for i in cumu_counts]
      selected_photons = []
      selected_times = []
      for i in selected:
        selected_photons.append(cumu_counts[i])
        selected_times.append(times[i])
      selected_photons.append(cumu_counts[-1])
      selected_times.append(times[-1])
      spacing = times[-1]/num_points
      
    #   if show == True:
    #     fig,(ax1, ax2)= plt.subplots(1,2)
    #     plt.rcParams["figure.figsize"] = (20,7)
    #     ax1.plot(times, cumu_counts)
    #     ax1.set_xlabel('Time [seconds]')
    #     ax1.set_ylabel('Cumalative Photon Count')

    #     ax1.set_title(f"Normalized Cumulative Photon Count vs Time")
    
    #     ax1.scatter(selected_times,selected_photons,c="orange")
      # ax1.show()
      

  
        
      selected_photons =[i*multiplier for i in selected_photons]
      gradient = np.gradient(selected_photons,spacing)
    #   if show == True:
    #     x = np.linspace(0,num_points+2,num_points+2,endpoint=False)
    #     ax2.scatter(x*(times[-1]/num_points)/1000,gradient)
    #     ax2.set_ylim([0, 2*max(gradient)])
    #     ax2.set_xlabel('Time [kiloseconds]')
    #     ax2.set_ylabel('Gradient')
    #     ax2.set_title(f"Gradient over Time")
      if avg_count_rate < CUTOFF and avg_count_rate > 2:
        for i in gradient:
            if i > 4: 
                return True
      if avg_count_rate >= CUTOFF:
        for i in gradient:
            if i < .25 or i > 3:
                makePlot(file,bin_times,binned_list,1000,avgs,errors,times,cumu_counts,initial_time,counter,galaxy)
                return True


def bin(xs,ys,binsize):
    out_y =  []
    out_x = []
    y_sum = 0
    x_sum = 0
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        if i == 0:
            last_x = 0
        else:
            last_x = xs[i-1]
        x_sum += (x-last_x)
        y_sum += y
        if x_sum >= binsize:
            out_y.append(y_sum)
            out_x.append(x)
            x_sum = 0
            y_sum = 0
    if y_sum != 0:
        out_y.append(y_sum)
        out_x.append(x)
    
    return out_x,out_y


def running_avg(number_ahead_and_behind,binned_list):
  avgs = []
  errors = []
  for i in range(0,len(binned_list)):
    # if i < 2:
    #   sum = binned_list[i]+binned_list[i+1]+binned_list[i+2]
    #   avg = sum /3
    #   avgs.append(avg)
    if i >= number_ahead_and_behind and i <= len(binned_list)-number_ahead_and_behind-1:
      list_of_averages = []
      list_of_averages.append(binned_list[i])
      for j in range(1,number_ahead_and_behind+1):
        list_of_averages.append(binned_list[i-j])
        list_of_averages.append(binned_list[i+j])
      summed = sum(list_of_averages)
      avg = summed / (number_ahead_and_behind*2+1)
      std_dev = statistics.stdev(list_of_averages)
      error = std_dev / (math.sqrt(number_ahead_and_behind*2+1))
      errors.append(error)
      avgs.append(avg)
    if i <= number_ahead_and_behind: 
      list_of_averages = []
      list_of_averages.append(binned_list[i])
      for j in range(1,number_ahead_and_behind+1):
        list_of_averages.append(binned_list[i+j])
      for l in range(1,i+1):
        list_of_averages.append(binned_list[i-l])
      summed = sum(list_of_averages)
      avg = summed / (number_ahead_and_behind*2+1)
      std_dev = statistics.stdev(list_of_averages)
      error = std_dev / (math.sqrt(number_ahead_and_behind*2+1))
      errors.append(error)
      avgs.append(avg)
    if i >= len(binned_list)-number_ahead_and_behind-1: 
      list_of_averages = []
      list_of_averages.append(binned_list[i])
      for j in range(1,number_ahead_and_behind+1):
        list_of_averages.append(binned_list[i-j])
      for l in range(1,len(binned_list)-i-1):
        list_of_averages.append(binned_list[i+l])
      summed = sum(list_of_averages)
      avg = summed / (number_ahead_and_behind*2+1)
      std_dev = statistics.stdev(list_of_averages)
      error = std_dev / (math.sqrt(number_ahead_and_behind*2+1))
      errors.append(error)
      avgs.append(avg)

    
  return avgs, errors

def makePlot(file,bin_times,binned_list,binsize,avgs,errors,times,cumu_counts,initial_time,counter,galaxy):
    print("Plot Being Made!")
    fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    plt.rcParams["figure.figsize"] = (15,10)

    kilo_times = [i/1000 for i in times]
    ax1.plot(kilo_times, cumu_counts)
    ax1.set_xlabel('Time [kiloseconds]')
    ax1.set_ylabel('Cumalative Photon Count')
    ax1.set_title(f"Cumulative Photon Count vs Time")
    
    ax2.step(bin_times, binned_list, linewidth = 1)
    ax2.set_xlabel("Time [kiloseconds]")
    ax2.set_title(f"LightCurve with {binsize}s Bin Size")
    ax2.set_ylabel("Photon Count per Bin")
    # ax2.axhline(med_avg,color="red")
    # print(len(avgs),len(errs))

    ax3.set_xlabel("Time [kiloseconds]")
    ax3.set_title(f"Running Average of {NUMBER_AHEAD_AND_BEHIND*2 +1} bins")
    ax3.set_ylabel("Photon Count per Bin")
    ax3.set_ylim([min(binned_list),max(binned_list)])
    ax3.plot(bin_times,avgs[:-2],color="red")
    ax3.errorbar(bin_times, avgs[:-2],
        yerr = errors[:-2],
        xerr = None,
        fmt ='.',
        color = "black",
        linewidth = .5,
        capsize = 1)

    # makes a filename for easier organization
    parts = file.split("_")
    plt.suptitle(f"{parts[0]}. ObsID  -- ({parts[1]})")
    plt.figtext(0.5, 0.01, f"Initial Time: {initial_time}", wrap=True, horizontalalignment='center', fontsize=12)
    # makes easier to read and then saves file
    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(file.split("/")[-1]+"_lc")
    fig.savefig(galaxy+f"/InterestingSignals{galaxy}/"+str(counter)+".png")
    plt.close()

def SloperSecondary(galaxy):
    # binsize = 500
    files = glob.glob(f'{galaxy}/textfiles/*.txt')
    
    try:
        os.makedirs(f'{galaxy}/InterestingSignals{galaxy}')
        print("Interesting Signals Directory created")
        counter = 0
        for i,file in enumerate(files):
            if Sloper(file,counter,galaxy):
                counter+=1
                print("Interesting Signal Found!")
            else:
                print("nothing")
    except FileExistsError:
        print("Directory already exists")
    

if __name__ == '__main__':
    galaxy = sys.argv[1]
    SloperSecondary(galaxy) 

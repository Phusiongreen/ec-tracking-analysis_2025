#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append("../")
from src.io import read_parameters
from src.computation import _calc_rel_vel, _filter_tracks


cmap = matplotlib.colormaps["seismic_r"]


# In[2]:

# read parameters and key file
parameter_file = "/home/jpa/PycharmProjects/ec-tracking-analysis_2025/data/collectivity/parameters_collectivity.yml"

parameters = read_parameters(parameter_file)       

key_file_path = parameters["key_file"]
key_file = pd.read_csv(key_file_path)

print("key file columns:", key_file.columns)
print("key file conditions:", key_file["condition"].unique())
print(key_file)


# In[4]:

# create output folder

output_folder = Path(parameters["output_folder"])
print("Output folder:", output_folder)

data_folder = Path(output_folder).joinpath("tracking_data")

subfolder = "trajectory_plots"
if not os.path.exists(output_folder.joinpath(subfolder)):
    os.mkdir(output_folder / subfolder)

##for single experiment ID##
#experimentID = "WT_Stat_noVEGF_rep1"
#experiment_df = key_file[key_file["experimentID"] == experimentID]

##for all experiment IDs##
#for experimentID in key_file["experimentID"].unique():
#   print("Processing experiment:", experimentID)
#   experiment_df = key_file[key_file["experimentID"] == experimentID]

experiment_df = key_file.copy()

print(experiment_df.head())

number_of_tracks_per_condition = parameters["number_of_tracks_per_condition"]

observation_time = parameters["observation_time"]
frames_per_hour = parameters["frames_per_hour"]
# Calculate observation time in hours
start_time_hours = observation_time[0] / frames_per_hour
end_time_hours = observation_time[1] / frames_per_hour

# limits for min velocity and max velocity in um/h used for the color map
min_vel_lim = parameters["velocity_colormap_limits"][0]
max_vel_lim = parameters["velocity_colormap_limits"][1]

# extend in microns of the coordinate system used for plotting, 
# range is [0, max_x] in x direction and [0, max_y] in y direction
max_x = 2000.0 # um
max_y = 2000.0 # um

list_of_files = []
experiment_ids = []
condition_ids = []


# uncomment this if you want to select specific files 
#19.06.25 update: I don't understand why this needs to be uncommented as it appends the file list? But only keeps the last one. If uncommented, none of them are used on next cell

for index, row in experiment_df.iterrows():
    condition = row["condition"]
    treatment = row["treatment"]
    print("condition:", condition)

    # read tracking data
    tracking_file = "tracking_data_%s_%s_%s.csv" % (treatment, row["color"], row["experimentID"])
    tracking_data_path = data_folder.joinpath(tracking_file)

    list_of_files.append(str(tracking_data_path))
    experiment_ids.append(row["experimentID"])
    condition_ids.append(condition)
condition_ids = np.unique(condition_ids)

# if you want to plot a specific file for example "tracking_data_1.csv", uncomment the following line and specify the file name:
# list_of_files = ["tracking_data_1.csv"]

print("Files to be analysed")
print(list_of_files)
print("Experiment IDs to be analysed")
print(experiment_ids)
print("Conditions to be analysed")
print(condition_ids)




# In[5]:

for tracking_data_path, experimentID in zip(list_of_files, experiment_ids):    
    max_vel = 0
    min_vel = 0
    number_of_tracks = 0
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    print("Processing tracking data for experiment:", experimentID)    
    tracking_data = pd.read_csv(tracking_data_path, low_memory=False)

    # filter data for observation time and get list of track IDs to be plotted
    observation_period_df, trackID_list = _filter_tracks(tracking_data, parameters)
    print("Available tracks: %s" % len(trackID_list))
    number_of_tracks += len(trackID_list)

    # plot trajectories
    for track_id in trackID_list:
        rel_vel, max_vel, min_vel, single_track_df, _, _ = _calc_rel_vel(observation_period_df, track_id, min_vel_lim, max_vel_lim, max_vel, min_vel, parameters)
        ax.plot(single_track_df["POSITION_X"],single_track_df["POSITION_Y"], color = cmap(rel_vel))
        
    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, label=r'velocity parallel to flow in $\mu m/h$', aspect=10, fraction = 0.05)
    cbar.set_ticks([0.0,0.5,1.0])
    mid_vel = (max_vel_lim + min_vel_lim)/2
    cbar.set_ticklabels([min_vel_lim, mid_vel, max_vel_lim])
    
    print("max velocity: ", max_vel, " um/h: ", " min velocity : ", min_vel, " um/h (parallel to flow)")
    
    # set figure properties
    ax.set_xlim(0,max_x)
    ax.set_ylim(0,max_y)
    ax.axhline(0, color = "red", linestyle = "--")
    ax.axvline(0, color = "red", linestyle = "--")

    # Dynamically get treatment for each experiment/file
    # Find the row in key_file matching the current experimentID
    treatment_row = key_file[key_file["experimentID"] == experimentID]
    if not treatment_row.empty:
        current_treatment = treatment_row["treatment"].iloc[0]
    else:
        current_treatment = "Unknown"

    # Calculate observation time in hours
    start_time_hours = observation_time[0] / frames_per_hour
    end_time_hours = observation_time[1] / frames_per_hour

    title_top = "Treatment: " + current_treatment
    title_bottom = ("sampled tracks: " + str(number_of_tracks) +
                " ; Observation time: t = " + str(np.round(start_time_hours, 1)) +
                " to " + str(np.round(end_time_hours, 1)) + " hours")
    fig.suptitle(title_top)
    ax.set_title(title_bottom)
    ax.set_xlabel(r"x in $\mu m$")
    ax.set_ylabel(r"y in $\mu m$")
    ax.set_aspect(1)
    plt.tight_layout()

    plt.savefig(output_folder / subfolder / f"{experimentID}_ROI_trajectories_from{start_time_hours}h_to_{end_time_hours}h.pdf")
    plt.savefig(output_folder / subfolder / f"{experimentID}_ROI_trajectories_from{start_time_hours}h_to_{end_time_hours}h.png")


# In[6]:

for condition in condition_ids:
    print("Processing condition:", condition)
    key_select_ = key_file[key_file["condition"] == condition]
    
    for treatment in key_select_["treatment"].unique():
        print("Processing treatment:", treatment)
        key_select = key_select_[key_select_["treatment"] == treatment] 
        #key_select = key_select[key_select["color"] == color]
        print(key_select.head())
        #color = key_select["color"].iloc[0]
        
        number_of_tracks_per_file = int(number_of_tracks_per_condition/key_select.shape[0])
        print("sample %s tracks per file" % number_of_tracks_per_file)
        
        track_counter = 0
        center_x = 0
        center_y = 0
        max_vel = 0
        min_vel = 0
        
        fig, ax = plt.subplots(figsize=(9,9))
        
        obs_time_length_frames = observation_time[1] - observation_time[0]
        # Convert frames to hours
        start_time_hours = observation_time[0] / frames_per_hour
        end_time_hours = observation_time[1] / frames_per_hour
        for index, row in key_select.iterrows():
        
            tracking_file = "tracking_data_%s_%s_%s.csv" % (treatment, row["color"], row["experimentID"])
            
            print("Processing tracking file:", tracking_file)
        
            data = pd.read_csv(data_folder / tracking_file, low_memory=False)
        
            observation_period_df, trackID_list = _filter_tracks(data, parameters)
           
            print("Available tracks: %s" % len(trackID_list))
            trackID_list = np.random.choice(trackID_list, number_of_tracks_per_file)
        
            print("sampled %s tracks for file %s" % (len(trackID_list),tracking_file))
            track_counter += len(trackID_list)
        
            center_per_file_x = 0.0
            center_per_file_y = 0.0
            total_dist_per_file = 0.0
        
            for track_id in trackID_list:                   
                rel_vel, max_vel, min_vel, single_track_df, delta_x, delta_y  = _calc_rel_vel(observation_period_df, track_id, min_vel_lim, max_vel_lim, max_vel, min_vel, parameters)
        
                ax.plot(single_track_df["X_from_origin"],single_track_df["Y_from_origin"], color = cmap(rel_vel))
                ax.plot([delta_x],[delta_y], color = "black", marker = "o", alpha=0.5) 
        
        # add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        cbar = fig.colorbar(sm, ax=ax, label=r'velocity parallel to flow in $\mu m/h$', aspect=10, fraction = 0.05)
        cbar.set_ticks([0.0,0.5,1.0])
            
        # Optionally, set tick labels if you want to customize them further
        mid_vel = (max_vel_lim + min_vel_lim)/2
        cbar.set_ticklabels([min_vel_lim, mid_vel, max_vel_lim])
        
        print("sampled %s tracks in total" % track_counter)
        print("max velocity: ", max_vel, " um/h: ", " min velocity : ", min_vel, " um/h (parallel to flow)")
        
        # set figure properties
        ax.set_xlim(-1000,1000)
        ax.set_ylim(-1000,1000)
        ax.axhline(0, color = "red", linestyle = "--")
        ax.axvline(0, color = "red", linestyle = "--")
       
        title_top = ("Treatment: " + treatment)

        title_bottom = (
            "sampled tracks: " + str(track_counter)
            + " ; Observation time: t = " + str(np.round(start_time_hours,1))
            + " to " + str(np.round(end_time_hours,1)) + " hours"
        )
                
        r"""title = ("Treatment: " + treatment 
                 + " sampled tracks: " + str(track_counter)
                 + " , min/max velocity parallel to flow: " + str(np.round(min_vel,3)) 
                 + "/" + str(np.round(max_vel,3)) + " $\mu m/h$" )"""
        fig.suptitle(title_top)
        ax.set_title(title_bottom)
        ax.set_xlabel(r"$\Delta x$ in $\mu m$")
        ax.set_ylabel(r"$\Delta y$ in $\mu m$")
        ax.set_aspect(1)
        plt.tight_layout()
            
        # save and show plot
        plt.savefig(output_folder / subfolder / f"trajectories_{treatment}_{row['experimentID']}.pdf")
        plt.savefig(output_folder / subfolder / f"trajectories_{treatment}_{row['experimentID']}.png")
        plt.show()


# In[ ]:

# Vibe code by GPT-o3, copying the cell above and making it loop over treatments only and ignoring condition


# In[7]:

# Group by treatment (ignoring condition)
for treatment, key_select in key_file.groupby("treatment"):
    print("Processing treatment:", treatment)
    # Calculate how many tracks per file (replicate)
    number_of_tracks_per_file = int(number_of_tracks_per_condition / key_select.shape[0])
    print("sample %s tracks per file" % number_of_tracks_per_file)

    track_counter = 0
    center_x = 0
    center_y = 0
    max_vel = 0
    min_vel = 0

    fig, ax = plt.subplots(figsize=(9,9))

    obs_time_length_frames = observation_time[1] - observation_time[0]
    # Convert start and end frames to hours
    start_time_hours = observation_time[0] / frames_per_hour
    end_time_hours = observation_time[1] / frames_per_hour

    # Loop over each replicate for the treatment
    for index, row in key_select.iterrows():
        tracking_file = "tracking_data_%s_%s_%s.csv" % (treatment, row["color"], row["experimentID"])
        print("Processing tracking file:", tracking_file)
        data = pd.read_csv(data_folder / tracking_file, low_memory=False)

        observation_period_df, trackID_list = _filter_tracks(data, parameters)

        print("Available tracks: %s" % len(trackID_list))
        # Sample the desired number of tracks from this file
        trackID_list = np.random.choice(trackID_list, number_of_tracks_per_file, replace=False)
        print("sampled %s tracks for file %s" % (len(trackID_list), tracking_file))
        track_counter += len(trackID_list)

        # Process each selected track and plot on the same axes
        for track_id in trackID_list:            
            rel_vel, max_vel, min_vel, single_track_df, delta_x, delta_y  = _calc_rel_vel(observation_period_df, track_id, min_vel_lim, max_vel_lim, max_vel, min_vel, parameters)
            
            ax.plot(single_track_df["X_from_origin"],
                    single_track_df["Y_from_origin"],
                    color = cmap(rel_vel))
            ax.plot([delta_x], [delta_y],
                    color = "black", marker = "o", alpha=0.5)

    # After processing all replicates for this treatment, add colorbar and titles
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, label=r'velocity parallel to flow in $\mu m/h$', aspect=10, fraction=0.05)
    cbar.set_ticks([0.0, 0.5, 1.0])
    mid_vel = (max_vel_lim + min_vel_lim)/2
    cbar.set_ticklabels([min_vel_lim, mid_vel, max_vel_lim])

    print("sampled %s tracks in total" % track_counter)
    print("max velocity: ", max_vel, " um/h; min velocity: ", min_vel, " um/h (parallel to flow)")

    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.axhline(0, color="red", linestyle="--")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel(r"$\Delta x$ in $\mu m$")
    ax.set_ylabel(r"$\Delta y$ in $\mu m$")
    ax.set_aspect(1)
    
    title_top = "Treatment: " + treatment + " (n=" + str(key_select.shape[0]) + ")"
    title_bottom = ("sampled tracks: " + str(track_counter)
                    + " ; Observation time: t = " + str(np.round(start_time_hours,1))
                    + " to " + str(np.round(end_time_hours,1)) + " hours")
    fig.suptitle(title_top)
    ax.set_title(title_bottom)
    plt.tight_layout()

    plt.savefig(output_folder / subfolder / f"trajectories_{treatment}_from{start_time_hours}h_to_{end_time_hours}h.pdf")
    plt.savefig(output_folder / subfolder / f"trajectories_{treatment}_from{start_time_hours}h_to_{end_time_hours}h.png")
    plt.show()


# In[ ]:


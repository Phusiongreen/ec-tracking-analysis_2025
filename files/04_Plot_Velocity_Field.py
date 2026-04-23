#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append("../")
from src.io import read_parameters
from src.computation import normalize_speed, build_velocity_dataset
from griottes import generate_delaunay_graph, plot_2D

cmap = matplotlib.colormaps["seismic_r"]


# In[2]:

# read parameters and key file

parameter_file = "/home/jpa/PycharmProjects/ec-tracking-analysis_2025/data/collectivity/parameters_collectivity.yml"
parameters = read_parameters(parameter_file)

key_file_path = parameters["key_file"]
key_file = pd.read_csv(key_file_path)

print("key file columns:", key_file.columns)
print("key file conditions:", key_file["condition"].unique())


# In[4]:

# create output folder

output_folder = Path(parameters["output_folder"])
print("Output folder:", output_folder)

data_folder = output_folder.joinpath("tracking_data")

subfolder = "velocity_field"
if not os.path.exists(output_folder.joinpath(subfolder)):
    os.mkdir(output_folder.joinpath(subfolder))

#tracking_data = pd.read_csv(data_folder / "tracking_data.csv") 


# In[5]:

# plot paramters

#number_of_tracks_per_condition = 1000

observation_time = parameters["observation_time"]

# limits for min velocity and max velocity in um/h used for the color map
min_vel_lim = parameters["velocity_colormap_limits"][0]
max_vel_lim = parameters["velocity_colormap_limits"][1]

# extend in microns of the coordinate system used for plotting, 
# range is [-max_x, max_x] in x direction and [-max_y, max_y] in y direction
max_x = -1  # um - leave -1 to set automatically
max_y = -1  # um - leave -1 to set automatically

obs_time_length_frames = observation_time[1] - observation_time[0]

plot_tracks = False
plot_arrows = True

distance_threshold = parameters["distance_threshold"]


# In[6]:

# plot velocity field for each experiment separately by condition
for condition in key_file["condition"].unique():

    # all experiments of one condition in the key file
    key_select = key_file[key_file["condition"] == condition]

    # loop over all experiments of this condition
    for experimentID in key_select["experimentID"].unique():

        # can only be one entry
        key_exp = key_select[key_select["experimentID"] == experimentID]

        row = key_exp.iloc[0]
        treatment = row["treatment"]

        tracking_file = "tracking_data_%s_%s_%s.csv" % (treatment, row["color"], row["experimentID"])

        data = pd.read_csv(data_folder / tracking_file, low_memory=False)

        _max_x = max_x
        if max_x == -1:
            _max_x = data["POSITION_X"].max()

        _max_y = max_y
        if max_y == -1:
            _max_y = data["POSITION_Y"].max()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_aspect('equal', 'box')
        ax[0].set_xlim(0, _max_x)
        ax[0].set_ylim(0, _max_y)

        ax[1].set_aspect('equal', 'box')
        ax[1].set_xlim(0, _max_x)
        ax[1].set_ylim(0, _max_y)


        print(data["POSITION_X"].min(), data["POSITION_X"].max())

        observation_period_df = data[data["FRAME"] <= observation_time[1]]
        observation_period_df = observation_period_df[observation_period_df["FRAME"] >= observation_time[0]]

        # Get the first frame of each track
        tracks_start = observation_period_df[observation_period_df["FRAME"] == observation_time[0]]

        # determine the min end point of the tracks
        # in case "allow_tracks_shorter_than_observation_time" in the parameters file is set to True
        # -> the min required track length is the min of 'obs_time_length_frames' and 'min_track_length'
        # in case "allow_tracks_shorter_than_observation_time" in the parameters file is set to False
        # -> the end point of each trajectory is the end of the observation time 

        frame_end_point = observation_time[1]
        if parameters["allow_tracks_shorter_than_observation_time"] == True:
            if obs_time_length_frames > parameters["min_track_length"]:
                frame_end_point = observation_time[0] + parameters["min_track_length"]

        # Assign tracks_end using the determined frame
        tracks_end = observation_period_df[observation_period_df["FRAME"] == frame_end_point]

        track_ids_start = np.array(tracks_start["TRACK_ID"].unique())
        track_ids_end = np.array(tracks_end["TRACK_ID"].unique())

        unique_common_track_ids = np.intersect1d(track_ids_start, track_ids_end)

        trackID_list = np.unique(unique_common_track_ids)

        num_tracks = len(trackID_list)

        print("Available tracks: %s" % num_tracks)

        ax[0].set_title("Treatment: %s, Tracks: %s (%s)" % (row["treatment"], num_tracks, row["color"]))
        ax[1].set_title("Velocity (um/h), min: %s, max: %s" % (min_vel_lim, max_vel_lim))

        for trackID in trackID_list:

            single_track_df = observation_period_df[observation_period_df["TRACK_ID"] == trackID]
            start_frame = single_track_df["FRAME"].min()
            end_frame = single_track_df["FRAME"].max()
            delta_frame = end_frame - start_frame

            delta_hour = delta_frame / parameters["frames_per_hour"]

            row_start = single_track_df[single_track_df["FRAME"] == start_frame]
            row_end = single_track_df[single_track_df["FRAME"] == end_frame]

            start_x = np.array(row_start["POSITION_X"])[0]
            start_y = np.array(row_start["POSITION_Y"])[0]
            end_x = np.array(row_end["POSITION_X"])[0]
            end_y = np.array(row_end["POSITION_Y"])[0]
            delta_x = end_x - start_x
            delta_y = end_y - start_y
            delta = np.sqrt(delta_x ** 2 + delta_y ** 2)

            rel_vel = normalize_speed(delta_x / delta_hour, min_vel_lim, max_vel_lim)

            if plot_tracks:
                ax[0].plot(single_track_df["POSITION_X"], single_track_df["POSITION_Y"], color=row["color"], alpha=0.5)
                ax[1].plot(single_track_df["POSITION_X"], single_track_df["POSITION_Y"], color=cmap(rel_vel), alpha=0.5)

            if plot_arrows:
                ax[0].arrow(start_x, start_y, delta_x, delta_y, head_width=10, head_length=10, fc=row["color"],
                            ec='black')
                ax[1].arrow(start_x, start_y, delta_x, delta_y, head_width=10, head_length=10, fc=cmap(rel_vel),
                            ec=cmap(rel_vel))

        # add legend
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=min_vel_lim, vmax=max_vel_lim))
        sm.set_array([])  # needed for colorbar

        pos0 = ax[0].get_position()
        pos1 = ax[1].get_position()

        cbar_width = 0.02
        cbar_pad = 0.01
        cbar_ax = fig.add_axes((
            pos1.x1 + cbar_pad,   # place just to the right of ax[1]
            pos1.y0,              # same bottom (y0)
            cbar_width,           # narrow width
            pos1.height           # same height as ax[1]
        ))

        # now draw the colorbar in that manually positioned axis
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Velocity (µm/h)')


        # adjust the second axis to have the same height as the first one
        ax[0].set_position([
            pos0.x0,
            pos0.y0,
            pos1.width,
            pos1.height
        ])
        ax[1].set_position([
            pos1.x0,
            pos0.y0,  # sync horizontally
            pos1.width,
            pos1.height
        ])


        plt.savefig(output_folder / subfolder / f"{experimentID}_velocity_field.png")
        plt.savefig(output_folder / subfolder / f"{experimentID}_velocity_field.pdf")
        plt.show()
        plt.close()



# In[ ]:




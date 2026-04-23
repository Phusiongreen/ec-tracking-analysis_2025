#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dabest
from pathlib import Path
import sys

sys.path.append("../")

from src.io import read_parameters


# In[3]:

# read parameters and key file

parameter_file = "/home/jpa/PycharmProjects/ec-tracking-analysis_2025/data/collectivity/parameters_collectivity.yml"
parameters = read_parameters(parameter_file)

key_file_path = parameters["key_file"]
key_file = pd.read_csv(key_file_path)

print("key file columns:", key_file.columns)
print("key file conditions:", key_file["condition"].unique())


# In[4]:

# parameter_file  = "../local/parameters.yml"
# parameters = read_parameters(parameter_file)   

# base_folder = parameters["base_folder"]
# tracking_data_files = parameters["tracking_data"]
# data_exclude = parameters["data_exclude"]
output_folder = Path(parameters["output_folder"])
tracking_data_path = output_folder.joinpath("speed_data")

# single_time_point_evaluation = parameters["single_time_point_evaluation"]

interval = parameters["time_lag"]

errorbar = ("ci", parameters["errorbar_ci"])

plt.rcParams.update({'font.size': parameters["font_size"]})

subsample_n = 10  # for speed up of draft plotting
subsample_frac = 0.1  # for speed up of draft plotting


# In[5]:

migration_speed_df = pd.DataFrame()

for index, row in key_file.iterrows():
    print("processing", row["condition"])
    #tracking_data_file = tracking_data_path + row["filename"]
    tracking_data_file = output_folder.joinpath(
        "speed_data",
        "migration_speed_df_%s_%s_%s.csv" % (row["treatment"], row["color"], row["experimentID"])
    )
    tracking_data = pd.read_csv(tracking_data_file)

    tracking_data['color'] = row['color']
    tracking_data['treatment'] = row['treatment']
    tracking_data['experimentID'] = row['experimentID']

    if migration_speed_df.empty:
        migration_speed_df = tracking_data
    else:
        migration_speed_df = pd.concat([migration_speed_df, tracking_data], ignore_index=True)

#migration_speed_df = migration_speed_df[migration_speed_df["time_lag"] == interval]
migration_speed_df


# In[6]:

full_hours = np.arange(0, migration_speed_df["FRAME"].max(), interval)
print(full_hours)
migration_speed_df = migration_speed_df[migration_speed_df["FRAME"].isin(full_hours)]
migration_speed_df


# In[7]:

plot_migration_speeds = migration_speed_df.dropna()

# filter observation time

obs_start = parameters["observation_time"][0]
obs_end = parameters["observation_time"][1]

plot_migration_speeds = plot_migration_speeds[plot_migration_speeds["FRAME"] >= obs_start]
plot_migration_speeds = plot_migration_speeds[plot_migration_speeds["FRAME"] <= obs_end]

plot_mean_migration_speeds = plot_migration_speeds.groupby(["experimentID", "condition", "color"])

plot_mean_migration_speeds = plot_mean_migration_speeds.mean(numeric_only=True).add_suffix('').reset_index()
plot_mean_migration_speeds["condition_color"] = plot_mean_migration_speeds["condition"] + "_" + \
                                                plot_mean_migration_speeds["color"]

print(plot_mean_migration_speeds)


# In[8]:


fig, ax = plt.subplots(1, figsize=(9, 6))

condition_palette = dict()

for cond in plot_mean_migration_speeds["condition_color"].unique():
    if "green" in cond:
        condition_palette[cond] = "limegreen"
    elif "red" in cond:
        condition_palette[cond] = "violet"
    #else:
    #    condition_palette[cond] = "red"

#sns.color_palette("husl", len(plot_mean_migration_speeds["condition_color"].unique()))

properties_dabest = dabest.load(data=plot_mean_migration_speeds, x="condition_color", y="vel_x_mu_per_h",
                                idx=('Stat-siRNA_black', 'Flow-siRNA_black'))

#properties_dabest = dabest.load(data = plot_other_df, x="CONDITION", y="clustering",
#
#                           idx=(( 'mosaic_siCTRL_siCTRL',  'mosaic_siCTRL_siSMAD4',  'mosaic_siCTRL_siAlk1')))

print(properties_dabest.mean_diff)
print(properties_dabest.mean_diff.results)

#properties_dabest = dabest.load(data = summary_data_CTR, x="time_interval", y="labelled_arterial_ECs",
#                          idx=('P8_P9', 'P5_P6', 'P5_P7', 'P5_P8', 'P5_P9'))

# Produce a Cumming estimation plot.
#test = properties_dabest.mean_diff.plot(swarm_ylim=(0, 10), contrast_ylim=(-3, 2));
#test = properties_dabest.mean_diff.plot(swarm_ylim=swarm_ylim, custom_palette = time_condition_palette) #, contrast_ylim=(-4, 2));
test = properties_dabest.mean_diff.plot(ax=ax, custom_palette=condition_palette)
test.savefig(output_folder / "mean_velocity_parallel_to_flow.pdf")
test.savefig(output_folder / "mean_velocity_parallel_to_flow.png")


# In[9]:


fig, ax = plt.subplots(1, figsize=(9, 6))

condition_palette = dict()

for cond in plot_mean_migration_speeds["condition_color"].unique():
    if "green" in cond:
        condition_palette[cond] = "limegreen"
    elif "red" in cond:
        condition_palette[cond] = "violet"
    #else:
    #    condition_palette[cond] = "red"

#sns.color_palette("husl", len(plot_mean_migration_speeds["condition_color"].unique()))

properties_dabest = dabest.load(data=plot_mean_migration_speeds, x="condition_color", y="vel_mu_per_h",
                                idx=(('mosaic_siScr_siScr_green', 'mosaic_siScr_siScr_red'),
                                     ('mosaic_siScr_siCdc42_green', 'mosaic_siScr_siCdc42_red'),
                                     ('mosaic_siScr_siRac1_green', 'mosaic_siScr_siRac1_red')))

#properties_dabest = dabest.load(data = plot_other_df, x="CONDITION", y="clustering",
#
#                           idx=(( 'mosaic_siCTRL_siCTRL',  'mosaic_siCTRL_siSMAD4',  'mosaic_siCTRL_siAlk1')))

print(properties_dabest.mean_diff)
print(properties_dabest.mean_diff.results)

#properties_dabest = dabest.load(data = summary_data_CTR, x="time_interval", y="labelled_arterial_ECs",
#                          idx=('P8_P9', 'P5_P6', 'P5_P7', 'P5_P8', 'P5_P9'))

# Produce a Cumming estimation plot.
#test = properties_dabest.mean_diff.plot(swarm_ylim=(0, 10), contrast_ylim=(-3, 2));
#test = properties_dabest.mean_diff.plot(swarm_ylim=swarm_ylim, custom_palette = time_condition_palette) #, contrast_ylim=(-4, 2));
test = properties_dabest.mean_diff.plot(ax=ax, custom_palette=condition_palette)
test.savefig(output_folder / "mean_velocity.pdf")
test.savefig(output_folder / "mean_velocity.png")


# In[ ]:


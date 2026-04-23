#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
from pathlib import Path

import dabest
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append("../")
from src.io import read_parameters
from griottes import generate_delaunay_graph, plot_2D
from src.computation import build_velocity_dataset

cmap = matplotlib.colormaps["seismic_r"]


# In[2]:

# read parameters and key file

parameter_file = "/home/jpa/PycharmProjects/ec-tracking-analysis_2025/data/collectivity/parameters_collectivity.yml"
parameters = read_parameters(parameter_file)       

key_file_path = parameters["key_file"]
key_file = pd.read_csv(key_file_path)

print("key file columns:", key_file.columns)
print("key file conditions:", key_file["condition"].unique())


# In[3]:

# create output folder

output_folder = Path(parameters["output_folder"])
print("Output folder:", output_folder)

data_folder = output_folder.joinpath("tracking_data")

subfolder = "graph_analysis"
if not os.path.exists(output_folder.joinpath(subfolder)):
    os.mkdir(output_folder.joinpath(subfolder))


# In[4]:

# plot paramters

#condition = 'mosaic_siCTRL_siAlk1'
#condition = 'mosaic_siCTRL_siSMAD4'
# color = 'red'
#treatment = 'siAlk1'

#number_of_tracks_per_condition = 1000

observation_time = parameters["observation_time"]

# limits for min velocity and max velocity in um/h used for the color map
min_vel_lim = parameters["velocity_colormap_limits"][0]
max_vel_lim = parameters["velocity_colormap_limits"][1]

# extend in microns of the coordinate system used for plotting, 
# range is [-max_x, max_x] in x direction and [-max_y, max_y] in y direction
max_x = 600 # um
max_y = 600 # um

obs_time_length_frames = observation_time[1] - observation_time[0]

plot_tracks = False
plot_arrows = True

distance_threshold = parameters["distance_threshold"]


# In[5]:


build_velocity_dataset(
    parameters,
    key_file,
    data_folder,
    observation_time,  # (start_frame, end_frame)
    obs_time_length_frames
)


# In[6]:

velocity_df = pd.read_csv(output_folder.joinpath(subfolder, "velocity_field.csv"))


# In[9]:

#does not work unless there is red and green combination sets?
velocity_corr_df = pd.DataFrame()  #columns=velocity_df.columns)

corr_index = 0

for experimentID in velocity_df["EXPERIMENT_ID"].unique():

    # get condition for this experiment
    condition = velocity_df[velocity_df["EXPERIMENT_ID"] == experimentID]["CONDITION"].unique()[0]
    velocity_df_exp = velocity_df[velocity_df["EXPERIMENT_ID"] == experimentID]

    # rename columns for griottes
    velocity_df_exp = velocity_df_exp.rename(columns={"END_X": "x", "END_Y": "y"})

    print("Generating graph...")
    G_delaunay = generate_delaunay_graph(
        velocity_df_exp[velocity_df_exp.columns],
        descriptors=velocity_df_exp.columns,
        distance=distance_threshold,
        image_is_2D=True
    )
    print("Graph generated!")

    neighbors_count = pd.DataFrame()
    k = 0
    for a, d in G_delaunay.nodes(data=True):

        neighbors = G_delaunay[a]
        # Count the neighbors for all colors
        red_count = sum(1 for neighbor in neighbors if G_delaunay.nodes[neighbor].get('color') == 'red')
        orange_count = sum(1 for neighbor in neighbors if G_delaunay.nodes[neighbor].get('color') == 'orange')
        green_count = sum(1 for neighbor in neighbors if G_delaunay.nodes[neighbor].get('color') == 'green')
        black_count = sum(1 for neighbor in neighbors if G_delaunay.nodes[neighbor].get('color') == 'black')        

        # determine other colors neighbors based on the color of the current node
        if d["color"] == "green":
            other_colors = red_count + orange_count + black_count
        elif d["color"] in ["red"]:
            other_colors = orange_count + green_count + black_count
        elif d["color"] in ["orange"]:
            other_colors = red_count + green_count + black_count
        elif d["color"] in ["black"]:
            other_colors = red_count + orange_count + green_count
        else:
            other_colors = 0
            
        total_neighbors = red_count + orange_count + green_count + black_count
                
        neighbors_count.at[k, "EXPERIMENT_ID"] = experimentID
        neighbors_count.at[k, "CONDITION"] = condition
        neighbors_count.at[k, "nx_label"] = a
        neighbors_count.at[k, "neighbour_count"] = total_neighbors
        neighbors_count.at[k, "red_count"] = red_count
        neighbors_count.at[k, "orange_count"] = orange_count
        neighbors_count.at[k, "green_count"] = green_count
        neighbors_count.at[k, "black_count"] = black_count
        neighbors_count.at[k, "other_count"] = other_colors
        neighbors_count.at[k, "color"] = d["color"]
        
        # increment totals for ratio calculation
        k += 1
   
    neighbors_count = neighbors_count.dropna()
    velocity_corr_df = pd.concat([velocity_corr_df, neighbors_count], ignore_index=True)

    # calculate num figures needed based on whether neighbors exist per color
    num_fig = 1 + sum(1 for color in ['red', 'orange', 'green', 'black'] if not neighbors_count[neighbors_count['color'] == color].empty)

    fig, ax = plt.subplots(1, num_fig, figsize=(5*num_fig, 5))

    nc_min = neighbors_count['neighbour_count'].min()
    nc_max = neighbors_count['neighbour_count'].max()
    if nc_min == nc_max:
        nc_min -= 0.5
        nc_max += 0.5
        
    sns.histplot(
            data=neighbors_count,
            x='neighbour_count',
            binwidth=1,
            binrange=(nc_min, nc_max),
            ax=ax[0],
            kde=False,
            stat='frequency'
    )
    ax[0].set_title(f"Experiment {experimentID} - {condition}", fontsize=12, pad=10)
    
    ax_idx = 1
    for color in ['red', 'orange', 'green', 'black']:
        if neighbors_count[neighbors_count['color'] == color].empty:
            continue
        color_count = f"{color}_count"
        g_min = neighbors_count[color_count].min()
        g_max = neighbors_count[color_count].max()
        if g_min == g_max:
            g_min -= 0.5
            g_max += 0.5
        sns.histplot(
            data=neighbors_count,
            x=color_count,
            binwidth=1,
            binrange=(g_min, g_max),
            ax=ax[ax_idx],
            kde=False,
            stat='frequency'
        )
        ax_idx += 1
        
    # visualize the graph for the experiment
    plot_2D(
        G_delaunay,
        alpha_line=0.6,
        scatterpoint_size=1,
        legend=True,
        legend_fontsize=14,
        figsize=(7, 7)
    )
     # Add title showing the experiment ID and condition
    ax = plt.gca()
    ax.set_title(f"Experiment {experimentID} - {condition}", fontsize=14, pad=15)

    plt.savefig(output_folder / subfolder / f"delaunay_graph_{experimentID}.pdf")
    plt.savefig(output_folder / subfolder / f"delaunay_graph_{experimentID}.png")
    plt.show()


# In[15]:

velocity_corr_df.to_csv(output_folder / subfolder / "velocity_corr.csv", index=False)
velocity_corr_df


# In[18]:

mean_neighbour_count_df = velocity_corr_df.groupby(["EXPERIMENT_ID","CONDITION","color"]).mean(numeric_only=True).add_suffix('').reset_index()

mean_neighbour_count_df


# In[19]:

plot_other_df = mean_neighbour_count_df[mean_neighbour_count_df['color'].isin(["red","orange"])]

# check if empty
if plot_other_df.empty:
    print("Warning: No red/orange data available for clustering plot — skipping.")
else:
    print(plot_other_df.head())

    plot_other_df["green_nbr_ratio"] = plot_other_df["green_count"]/(plot_other_df["neighbour_count"])
    plot_other_df["clustering"] = plot_other_df["green_ratio"]/plot_other_df["green_nbr_ratio"]

    properties_dabest = dabest.load(data = plot_other_df, x="CONDITION", y="clustering", idx=("Stat-siRNA", "Flow-siRNA"))

    fig, ax = plt.subplots(1, figsize=(10, 5))

    # properties_dabest = dabest.load(data = plot_other_df, x="CONDITION", y="clustering",
    #                           idx=(( 'mosaic_siCTRL_siCTRL',  'mosaic_siCTRL_siSMAD4',  'mosaic_siCTRL_siAlk1')))

    print(properties_dabest.mean_diff)
    print(properties_dabest.mean_diff.results)

    #properties_dabest = dabest.load(data = summary_data_CTR, x="time_interval", y="labelled_arterial_ECs",
    #                          idx=('P8_P9', 'P5_P6', 'P5_P7', 'P5_P8', 'P5_P9'))

    # Produce a Cumming estimation plot.
    #test = properties_dabest.mean_diff.plot(swarm_ylim=(0, 10), contrast_ylim=(-3, 2));
    #test = properties_dabest.mean_diff.plot(swarm_ylim=swarm_ylim, custom_palette = time_condition_palette) #, contrast_ylim=(-4, 2));
    test = properties_dabest.mean_diff.plot(ax=ax) 
    test.savefig(output_folder / "clustering.pdf")
    test.savefig(output_folder / "clustering.png")
    #fig, ax = plt.subplots(1, figsize=(15, 5))
    #sns.swarmplot(data = plot_other_df,
    #            x = 'CONDITION',
    #            y = 'clustering',
    #            #hue = 'condition',
    #            ax = ax)


# In[20]:

plot_df = pd.DataFrame(({"EXPERIMENT_ID": [], "CONDITION": [], "ratio": []}))

k = 0
for experimentID in mean_neighbour_count_df["EXPERIMENT_ID"].unique():
    print(k)
    temp_df = mean_neighbour_count_df[mean_neighbour_count_df["EXPERIMENT_ID"]==experimentID]
    
    red_df = temp_df[temp_df["color"].isin(["red","orange"])]
    green_df = temp_df[temp_df["color"] == "green"] 

    if red_df.empty or green_df.empty:
        print(f"Warning: Skipping experiment {experimentID} — missing red/orange or green data.")
        continue

    red_df["red_green_ratio"] = red_df["other_count"]/red_df["green_count"]
    green_df["red_green_ratio"] = green_df["other_count"]/green_df["green_count"]

    plot_df.at[k, "EXPERIMENT_ID"] = experimentID
    plot_df.at[k, "CONDITION"] = temp_df["CONDITION"].iloc[0]
    plot_df.at[k, "ratio"] = red_df["red_green_ratio"].iloc[0]/green_df["red_green_ratio"].iloc[0]  
    k += 1
    
print(plot_df)


# In[21]:

if plot_df.empty:
    print("Warning: No data available for sorting plot — skipping.")
else:
    fig, ax = plt.subplots(1, figsize=(10, 5))

    properties_dabest = dabest.load(data = plot_df, x="CONDITION", y="ratio",
                              idx=(( 'mosaic_siScr_siScr',  'mosaic_siScr_siCdc42',  'mosaic_siScr_siRac1')))

    # properties_dabest = dabest.load(data = plot_df, x="CONDITION", y="ratio",
    #                          idx=(( 'mosaic_siCTRL_siCTRL',  'mosaic_siCTRL_siSMAD4',  'mosaic_siCTRL_siAlk1')))


    print(properties_dabest.mean_diff)
    print(properties_dabest.mean_diff.results)

    #properties_dabest = dabest.load(data = summary_data_CTR, x="time_interval", y="labelled_arterial_ECs",
    #                          idx=('P8_P9', 'P5_P6', 'P5_P7', 'P5_P8', 'P5_P9'))

    # Produce a Cumming estimation plot.
    #test = properties_dabest.mean_diff.plot(swarm_ylim=(0, 10), contrast_ylim=(-3, 2));
    #test = properties_dabest.mean_diff.plot(swarm_ylim=swarm_ylim, custom_palette = time_condition_palette) #, contrast_ylim=(-4, 2));
    test = properties_dabest.mean_diff.plot(ax=ax)
    test.savefig(output_folder / "sorting.pdf")
    test.savefig(output_folder / "sorting.png")


# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
from pathlib import Path

import numpy as np
import dabest
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append("../")
from src.io import read_parameters
from griottes import generate_delaunay_graph, plot_2D
from src.computation import build_velocity_dataset
import networkx as nx



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

subfolder = "time_graph_analysis"
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
min_vel_lim = -20.0 # um/h
max_vel_lim = 20.0 # um/h

# extend in microns of the coordinate system used for plotting, 
# range is [-max_x, max_x] in x direction and [-max_y, max_y] in y direction
max_x = 600 # um
max_y = 600 # um

obs_time_length_frames = observation_time[1] - observation_time[0]

print("Observation time (frames):", observation_time)

plot_tracks = False
plot_arrows = True

distance_threshold = parameters.get("distance_threshold", 50) # um, default value if not specified in parameters


# In[5]:


num_subsample = 5


# In[6]:


graphs = {}
observation_period_dfs = {}
# loop over all experiments 
for experimentID in key_file["experimentID"].unique():
    print("Processing experiment:", experimentID)

    # can only be one entry
    key_exp = key_file[key_file["experimentID"] == experimentID]

    row = key_exp.iloc[0]
    treatment = row["treatment"]

    tracking_file = "tracking_data_%s_%s_%s.csv" % (treatment, row["color"], row["experimentID"])
    data = pd.read_csv(str(data_folder.joinpath(tracking_file)), low_memory=False)

    # filter data for observation period
    observation_period_df = data[data["FRAME"] <= observation_time[1]]
    observation_period_df = observation_period_df[observation_period_df["FRAME"] >= observation_time[0]]
    observation_period_dfs[experimentID] = observation_period_df

    # rename columns for G_delaunay graph generation
    observation_period_df_exp = observation_period_df.rename(columns={"POSITION_X": "x", "POSITION_Y": "y"})
    # add "label" column
    observation_period_df_exp["label"] = observation_period_df_exp["TRACK_ID"]

    t_graphs = []

    # build G_delaunay graph per point in time
    for t in range(observation_time[0], observation_time[1]):
        observation_period_df_exp_id_t = observation_period_df_exp[observation_period_df_exp["FRAME"] == t].reset_index(drop=True)

        #print(observation_period_df_exp_id_t[["FRAME", "TRACK_ID", "label"]].head())

        print("Generating graph at time point %d ..." % t)
        G_delaunay = generate_delaunay_graph(
            observation_period_df_exp_id_t[observation_period_df_exp_id_t.columns],
            descriptors=observation_period_df_exp_id_t.columns,
            distance=distance_threshold,
            image_is_2D=True
        )
        print("Graph generated!")

        # relabel nodes to TRACK_IDs
        mapping = {n: G_delaunay.nodes[n]["TRACK_ID"] for n in G_delaunay.nodes}
        G_delaunay = nx.relabel_nodes(G_delaunay, mapping)

        t_graphs.append(G_delaunay)

    # store graphs for this experiment (using experimentID for now, will map to condition later)
    graphs[experimentID] = t_graphs


# In[9]:


from src.node_trajectory_viz import plot_node_trajectory_3d

# Choose a node ID to track (must be a TRACK_ID that exists in your graphs)
node_to_track = 123  # Replace with your desired node ID

# Visualize node trajectory for each experiment (organized by condition)
for condition in key_file["condition"].unique():
    print(f"\n=== Visualizing condition: {condition} ===")
    
    # Get all experiments for this condition
    experiments_for_condition = key_file[key_file["condition"] == condition]
    
    for idx, (_, row) in enumerate(experiments_for_condition.iterrows()):
        experimentID = row["experimentID"]
        print(f"  Processing experiment {idx+1}/{len(experiments_for_condition)}: {experimentID}")
        
        if experimentID not in graphs:
            print(f"    Skipping - no graphs for {experimentID}")
            continue
        
        # Generate the visualization
        fig, ax = plot_node_trajectory_3d(
            node_id=node_to_track,
            graphs=graphs,                    # from previous cell
            experimentID=experimentID,        # which experiment to visualize
            observation_time=observation_time, # frame range from parameters
            figsize=(14, 10),
            node_color='red',
            neighbor_color='lightblue',
            neighbor_alpha=0.35,
            trajectory_linewidth=2.5,
            show_neighbors=False,
        )

        plt.show()


# In[8]:


neighbor_lifetimes_by_condition = {}

# iterate over conditions
for condition in key_file["condition"].unique():
    print(f"\nAnalyzing condition: {condition}")
    
    # Get all experiments for this condition
    experiments_for_condition = key_file[key_file["condition"] == condition]
    
    # Aggregate neighbor lifetimes across all experiments in this condition
    neighbor_lifetimes_all_tracks = {}
    
    for _, row in experiments_for_condition.iterrows():
        experimentID = row["experimentID"]
        print(f"  Processing experiment: {experimentID}")
        
        if experimentID not in observation_period_dfs:
            print(f"    Skipping - no data for {experimentID}")
            continue
        
        observation_period_df = observation_period_dfs[experimentID]
        
        if experimentID not in graphs:
            print(f"    Skipping - no graphs for {experimentID}")
            continue
        
        # get time graphs for this experiment
        t_graphs = graphs[experimentID]
        
        # iterate over sampled tracks
        trackIDs = observation_period_df["TRACK_ID"].unique()
        
        for track_id in trackIDs:
            neighbor_lifetimes = {}

            # first time point
            t_graphs0 = t_graphs[0]

            if track_id not in t_graphs0.nodes(data="TRACK_ID"):
                continue

            neighbors0 = list(t_graphs0.neighbors(track_id))

            for neighbor in neighbors0:
                neighbor_lifetimes[neighbor] = 1  # initialize lifetimes at time 0

            # iterate over rest of time points
            for t, G_delaunay in enumerate(t_graphs[1:], start=1):

                # iterate
                node_track = [n for n, d in G_delaunay.nodes(data=True) if d.get("TRACK_ID") == track_id]
                if len(node_track) > 1:
                    continue
                elif len(node_track) == 0:
                    break

                a = node_track[0]
                neighbors_t = G_delaunay[a]

                # increment lifetime count for each observed neighbor
                for neighbor in neighbors_t:
                    if neighbor in neighbor_lifetimes:
                        # increment only if consecutive
                        if neighbor_lifetimes[neighbor] == t:
                            neighbor_lifetimes[neighbor] += 1

            neighbor_lifetimes_all_tracks[track_id] = neighbor_lifetimes
    
    neighbor_lifetimes_by_condition[condition] = neighbor_lifetimes_all_tracks


# In[9]:


# randomly sample some tracks and plot histograms of neighbor lifetimes
for condition in key_file["condition"].unique():
    print(f"\nCondition: {condition}")

    neighbor_lifetimes_tracks = neighbor_lifetimes_by_condition[condition]

    trackIDs = list(neighbor_lifetimes_tracks.keys())

    if len(trackIDs) > num_subsample:
        sampled_trackIDs = np.random.choice(trackIDs, size=num_subsample, replace=False)
    else:
        sampled_trackIDs = trackIDs

    print(f"Sampled {len(sampled_trackIDs)} track IDs:", sampled_trackIDs)

    for track_id in sampled_trackIDs:
        neighbor_lifetimes = neighbor_lifetimes_tracks[track_id]

        lifetimes = list(neighbor_lifetimes.values())

        print(f"Edge lifetimes for track ID {track_id}:", lifetimes)

        plt.figure()
        plt.hist(lifetimes, bins=range(1, obs_time_length_frames + 2),
                 align='left', color='blue', alpha=0.7)
        plt.xlabel('Neighbor Lifetime (frames)')
        plt.ylabel('Frequency')
        plt.title(f'Neighbor Lifetime Histogram for Track ID {track_id} (Condition: {condition})')
        plt.xticks(range(1, obs_time_length_frames + 1))
        plt.grid(axis='y')
        plt.show()


# In[10]:


# summary stats of neighbor lifetimes across all tracks
conditions = key_file["condition"].unique()

# Prepare a single row of subplots
n_conditions = len(conditions)
fig, axes = plt.subplots(1, n_conditions, figsize=(6 * n_conditions, 5), sharey=True)

if n_conditions == 1:
    axes = [axes]  # ensure axes is iterable even for a single condition

for ax, condition in zip(axes, conditions):
    print(f"\nCondition: {condition}")

    neighbor_lifetimes_tracks = neighbor_lifetimes_by_condition[condition]

    all_lifetimes = []
    for track_id, neighbor_lifetimes in neighbor_lifetimes_tracks.items():
        lifetimes = list(neighbor_lifetimes.values())
        all_lifetimes.extend(lifetimes)

    if all_lifetimes:
        mean_lifetime = np.mean(all_lifetimes)
        median_lifetime = np.median(all_lifetimes)
        std_lifetime = np.std(all_lifetimes)

        print(f"Mean Neighbor Lifetime: {mean_lifetime:.2f} frames")
        print(f"Median Neighbor Lifetime: {median_lifetime:.2f} frames")
        print(f"Standard Deviation: {std_lifetime:.2f} frames")

        # plot histogram in its subplot
        ax.hist(all_lifetimes, bins=range(1, obs_time_length_frames + 2),
                align='left', color='green', alpha=0.7)
        ax.set_xlabel('Neighbor Lifetime (frames)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Condition: {condition}')
        ax.set_xticks(range(1, obs_time_length_frames + 1))
        ax.grid(axis='y')

    else:
        print("No neighbor lifetimes recorded for this condition.")
        ax.set_visible(False)

plt.tight_layout()
plt.show()


# In[11]:


# Compute neighbor retention/survival curve Sn(tau)
# Sn(τ) = ⟨|Ni(t0) ∩ Ni(t0+τ)| / |Ni(t0)|⟩ for all i,t0
# This quantifies what fraction of original neighbors are retained after time tau

def compute_neighbor_retention_curve(graphs_dict, key_file, observation_time):
    """
    Compute the neighbor retention/survival curve for each condition.
    
    For each time lag τ, computes the fraction of neighbors that persist.
    
    Parameters:
    -----------
    graphs_dict : dict
        Dictionary mapping experimentID to list of graphs (one per timepoint)
    key_file : pd.DataFrame
        Key file with experimentID and condition information
    observation_time : tuple
        (start_frame, end_frame)
    
    Returns:
    --------
    survival_curves : dict
        Dictionary mapping condition to dict with:
        - 'tau': array of time lags (in frames)
        - 'Sn': array of retention fractions for each tau
        - 'Sn_std': standard deviation of retention fractions
        - 'N_measurements': number of (track, t0) pairs averaged
    """
    start_frame, end_frame = observation_time
    max_tau = end_frame - start_frame - 1  # maximum possible tau
    
    survival_curves = {}
    
    for condition in key_file["condition"].unique():
        print(f"\nComputing neighbor retention curve for condition: {condition}")
        
        # Get all experiments for this condition
        experiments_for_condition = key_file[key_file["condition"] == condition]
        
        # Dictionary to store retention fractions for each tau
        retention_by_tau = {tau: [] for tau in range(1, max_tau + 1)}
        
        # Iterate over all experiments in this condition
        for _, row in experiments_for_condition.iterrows():
            experimentID = row["experimentID"]
            
            if experimentID not in graphs_dict:
                print(f"  Skipping {experimentID} - no graphs")
                continue
            
            t_graphs = graphs_dict[experimentID]
            
            # Iterate over all starting timepoints (t0)
            for t0_idx in range(len(t_graphs) - 1):  # -1 because we need at least one future timepoint
                G_t0 = t_graphs[t0_idx]
                
                # For each node in the graph at t0
                for node_id in G_t0.nodes():
                    neighbors_t0 = set(G_t0.neighbors(node_id))
                    
                    # Skip if node has no neighbors
                    if len(neighbors_t0) == 0:
                        continue
                    
                    # For each future timepoint (t0 + tau)
                    for tau in range(1, max_tau + 1):
                        t_idx = t0_idx + tau
                        if t_idx >= len(t_graphs):
                            break
                        
                        G_t = t_graphs[t_idx]
                        
                        # Check if node still exists at this future timepoint
                        if node_id not in G_t.nodes():
                            continue
                        
                        # Get neighbors at future timepoint
                        neighbors_t = set(G_t.neighbors(node_id))
                        
                        # Compute retention: intersection / original size
                        intersection_size = len(neighbors_t0 & neighbors_t)
                        retention_fraction = intersection_size / len(neighbors_t0)
                        
                        retention_by_tau[tau].append(retention_fraction)
        
        # Compute mean and std for each tau
        tau_values = []
        Sn_values = []
        Sn_std_values = []
        N_measurements_values = []
        
        for tau in sorted(retention_by_tau.keys()):
            if len(retention_by_tau[tau]) > 0:
                tau_values.append(tau)
                Sn_values.append(np.mean(retention_by_tau[tau]))
                Sn_std_values.append(np.std(retention_by_tau[tau]))
                N_measurements_values.append(len(retention_by_tau[tau]))
        
        survival_curves[condition] = {
            'tau': np.array(tau_values),
            'Sn': np.array(Sn_values),
            'Sn_std': np.array(Sn_std_values),
            'N_measurements': np.array(N_measurements_values)
        }
        
        print(f"  Computed retention curve with {len(tau_values)} time points")
        if len(N_measurements_values) > 0:
            print(f"  Average measurements per time point: {np.mean(N_measurements_values):.0f}")
    
    return survival_curves


# Compute the survival curves
survival_curves = compute_neighbor_retention_curve(graphs, key_file, observation_time)


# In[12]:


# Plot neighbor retention curves for all conditions
fig, ax = plt.subplots(figsize=(10, 6))

for condition in key_file["condition"].unique():
    curve_data = survival_curves[condition]
    tau = curve_data['tau']
    Sn = curve_data['Sn']
    Sn_std = curve_data['Sn_std']
    
    # Plot with error bars
    ax.errorbar(tau, Sn, yerr=Sn_std, marker='o', label=condition, linewidth=2, capsize=3, alpha=0.7)

ax.set_xlabel('Time lag τ (frames)', fontsize=12, fontweight='bold')
ax.set_ylabel('Neighbor Retention Sₙ(τ)', fontsize=12, fontweight='bold')
ax.set_title('Neighbor Retention/Survival Curves by Condition', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()


# In[13]:


# Fit exponential decay to retention curves and extract half-life
from scipy.optimize import curve_fit

def exponential_decay(tau, S0, k):
    """
    Exponential decay model: S(τ) = S0 * exp(-k*τ)
    """
    return S0 * np.exp(-k * tau)

retention_analysis = {}

for condition in key_file["condition"].unique():
    print(f"\n--- Fitting decay model for condition: {condition} ---")
    
    curve_data = survival_curves[condition]
    tau = curve_data['tau']
    Sn = curve_data['Sn']
    Sn_std = curve_data['Sn_std']
    
    # Fit exponential decay
    try:
        # Use weights inverse to variance for better fitting
        weights = 1.0 / (Sn_std + 1e-6)
        
        popt, pcov = curve_fit(
            exponential_decay, 
            tau, 
            Sn,
            p0=[1.0, 0.1],
            sigma=Sn_std,
            absolute_sigma=True,
            maxfev=5000
        )
        
        S0, k = popt
        S0_err, k_err = np.sqrt(np.diag(pcov))
        
        # Calculate half-life: t_1/2 = ln(2) / k
        if k > 0:
            half_life = np.log(2) / k
            half_life_err = (np.log(2) / (k**2)) * k_err
        else:
            half_life = np.inf
            half_life_err = np.inf
        
        retention_analysis[condition] = {
            'S0': S0,
            'S0_err': S0_err,
            'k': k,
            'k_err': k_err,
            'half_life': half_life,
            'half_life_err': half_life_err,
            'fit_success': True
        }
        
        print(f"  S0 (initial retention): {S0:.3f} ± {S0_err:.3f}")
        print(f"  Decay rate k: {k:.4f} ± {k_err:.4f} frame⁻¹")
        print(f"  Half-life t_1/2: {half_life:.2f} ± {half_life_err:.2f} frames")
        
    except Exception as e:
        print(f"  Warning: Fitting failed - {e}")
        retention_analysis[condition] = {'fit_success': False}


# In[14]:


# Plot retention curves with exponential fits
fig, axes = plt.subplots(1, len(key_file["condition"].unique()), 
                          figsize=(6 * len(key_file["condition"].unique()), 5))

if len(key_file["condition"].unique()) == 1:
    axes = [axes]

for ax, condition in zip(axes, key_file["condition"].unique()):
    curve_data = survival_curves[condition]
    tau = curve_data['tau']
    Sn = curve_data['Sn']
    Sn_std = curve_data['Sn_std']
    
    # Plot data with error bars
    ax.errorbar(tau, Sn, yerr=Sn_std, marker='o', markersize=6, label='Data',
                linewidth=2, capsize=4, color='blue', alpha=0.7)
    
    # Plot fit if successful
    if retention_analysis[condition]['fit_success']:
        S0 = retention_analysis[condition]['S0']
        k = retention_analysis[condition]['k']
        half_life = retention_analysis[condition]['half_life']
        
        # Generate smooth fit curve
        tau_smooth = np.linspace(tau[0], tau[-1], 100)
        Sn_fit = exponential_decay(tau_smooth, S0, k)
        
        ax.plot(tau_smooth, Sn_fit, 'r-', linewidth=2.5, label=f'Fit (t₁/₂={half_life:.1f} f)')
    
    ax.set_xlabel('Time lag τ (frames)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Neighbor Retention Sₙ(τ)', fontsize=11, fontweight='bold')
    ax.set_title(f'Condition: {condition}', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()


# In[15]:


# Summary table of retention metrics
print("\n" + "="*70)
print("NEIGHBOR RETENTION ANALYSIS SUMMARY")
print("="*70)

summary_data = []
for condition in key_file["condition"].unique():
    # Get all experiments for this condition
    experiments_for_condition = key_file[key_file["condition"] == condition]
    n_experiments = len(experiments_for_condition)
    
    analysis = retention_analysis[condition]
    
    summary_row = {
        'Condition': condition,
        'N Experiments': n_experiments,
    }
    
    if analysis.get('fit_success'):
        summary_row['S0 (initial)'] = f"{analysis['S0']:.3f} ± {analysis['S0_err']:.3f}"
        summary_row['k (decay rate)'] = f"{analysis['k']:.4f} ± {analysis['k_err']:.4f}"
        summary_row['Half-life (frames)'] = f"{analysis['half_life']:.2f} ± {analysis['half_life_err']:.2f}"
    else:
        summary_row['S0 (initial)'] = 'N/A'
        summary_row['k (decay rate)'] = 'N/A'
        summary_row['Half-life (frames)'] = 'N/A'
    
    summary_data.append(summary_row)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))
print("="*70)


# In[ ]:


# neighborhood consistency as a heatmap (tracks × time) to identify locally unstable areas


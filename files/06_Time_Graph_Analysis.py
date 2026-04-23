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

sys.path.append("../")
from src.io import read_parameters
from src.graph_analysis import (
    build_time_graphs,
    compute_neighbor_lifetimes,
    compute_neighbor_retention_curve,
    exponential_decay,
    compute_relative_neighbor_displacement,
    compute_velocity_alignment_curves,
    compute_anisotropic_separation_and_msd,
)
from src.node_trajectory_viz import plot_node_trajectory_3d

from scipy.optimize import curve_fit


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

# plot parameters

observation_time = parameters["observation_time"]

# limits for min velocity and max velocity in um/h used for the color map
min_vel_lim = parameters["velocity_colormap_limits"][0]
max_vel_lim = parameters["velocity_colormap_limits"][1]

# extend in microns of the coordinate system used for plotting, 
# range is [-max_x, max_x] in x direction and [-max_y, max_y] in y direction
max_x = 600 # um
max_y = 600 # um

obs_time_length_frames = observation_time[1] - observation_time[0]

print("Observation time (frames):", observation_time)

plot_tracks = False
plot_arrows = True

distance_threshold = parameters["distance_threshold"]


# In[5]:

num_subsample = parameters["num_subsample"]


# In[6]:

# Build Delaunay graphs per time point for each experiment
graphs, observation_period_dfs = build_time_graphs(
    key_file, data_folder, observation_time, distance_threshold
)


# In[7]:

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
            graphs=graphs,
            experimentID=experimentID,
            observation_time=observation_time,
            figsize=(14, 10),
            node_color='red',
            neighbor_color='lightblue',
            neighbor_alpha=0.35,
            trajectory_linewidth=2.5,
            show_neighbors=False,
        )

        plt.savefig(output_folder / subfolder / f"node_trajectory_3d_{experimentID}.pdf")
        plt.savefig(output_folder / subfolder / f"node_trajectory_3d_{experimentID}.png")
        plt.show()


# Compute neighbor lifetimes per condition
neighbor_lifetimes_by_condition = compute_neighbor_lifetimes(
    graphs, observation_period_dfs, key_file
)


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
        plt.savefig(output_folder / subfolder / f"neighbor_lifetime_hist_{condition}_track{track_id}.pdf")
        plt.savefig(output_folder / subfolder / f"neighbor_lifetime_hist_{condition}_track{track_id}.png")
        plt.show()


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
plt.savefig(output_folder / subfolder / "neighbor_lifetime_summary.pdf")
plt.savefig(output_folder / subfolder / "neighbor_lifetime_summary.png")
plt.show()


# In[11]:

# Compute the neighbor retention / survival curves
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
plt.savefig(output_folder / subfolder / "neighbor_retention_curves.pdf")
plt.savefig(output_folder / subfolder / "neighbor_retention_curves.png")
plt.show()


# In[13]:

# Fit exponential decay to retention curves and extract half-life

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
plt.savefig(output_folder / subfolder / "retention_curves_with_fits.pdf")
plt.savefig(output_folder / subfolder / "retention_curves_with_fits.png")
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


# In[16]:

# Compute cage-relative neighbor displacement curves
displacement_curves = compute_relative_neighbor_displacement(graphs, key_file, observation_time)


# In[17]:

# Plot cage-relative displacement ⟨δr(τ)⟩ for all conditions
fig, ax = plt.subplots(figsize=(10, 6))

for condition in key_file["condition"].unique():
    cd = displacement_curves[condition]
    tau = cd["tau"]
    mean = cd["delta_r_mean"]
    sem = cd["delta_r_sem"]

    ax.plot(tau, mean, marker="o", linewidth=2, label=condition)
    ax.fill_between(tau, mean - sem, mean + sem, alpha=0.2)

ax.set_xlabel("Time lag τ (frames)", fontsize=12, fontweight="bold")
ax.set_ylabel("⟨δr(τ)⟩  (µm)", fontsize=12, fontweight="bold")
ax.set_title("Cage-Relative Neighbor Displacement", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(output_folder / subfolder / "cage_relative_displacement.pdf")
plt.savefig(output_folder / subfolder / "cage_relative_displacement.png")
plt.show()


# In[ ]:

# neighborhood consistency as a heatmap (tracks × time) to identify locally unstable areas


# In[18]:


# Per-condition subplots: mean ± std (shaded) with pair counts annotated
conditions = key_file["condition"].unique()
n_cond = len(conditions)

fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), sharey=True)
if n_cond == 1:
    axes = [axes]

for ax, condition in zip(axes, conditions):
    cd = displacement_curves[condition]
    tau = cd["tau"]
    mean = cd["delta_r_mean"]
    std = cd["delta_r_std"]

    ax.plot(tau, mean, "o-", color="teal", linewidth=2)
    ax.fill_between(tau, mean - std, mean + std, color="teal", alpha=0.15, label="± 1 std")
    ax.set_xlabel("Time lag τ (frames)", fontsize=11, fontweight="bold")
    ax.set_ylabel("⟨δr(τ)⟩  (µm)", fontsize=11, fontweight="bold")
    ax.set_title(f"Condition: {condition}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(output_folder / subfolder / "cage_relative_displacement_per_condition.pdf")
plt.savefig(output_folder / subfolder / "cage_relative_displacement_per_condition.png")
plt.show()


# In[19]:


# Summary table of cage-relative displacement metrics
print("\n" + "=" * 80)
print("CAGE-RELATIVE NEIGHBOR DISPLACEMENT SUMMARY")
print("=" * 80)

summary_rows = []
for condition in key_file["condition"].unique():
    cd = displacement_curves[condition]
    if len(cd["tau"]) == 0:
        continue

    # δr at τ=1 and at the last available τ
    dr_first = cd["delta_r_mean"][0]
    dr_last = cd["delta_r_mean"][-1]
    tau_last = cd["tau"][-1]

    # Slope of linear fit (simple metric for rate of separation)
    if len(cd["tau"]) > 1:
        coeffs = np.polyfit(cd["tau"], cd["delta_r_mean"], 1)
        slope = coeffs[0]
    else:
        slope = np.nan

    summary_rows.append({
        "Condition": condition,
        "⟨δr⟩ at τ=1 (µm)": f"{dr_first:.2f}",
        f"⟨δr⟩ at τ={tau_last} (µm)": f"{dr_last:.2f}",
        "Linear slope (µm/frame)": f"{slope:.3f}",
        "Avg pairs per τ": f"{np.mean(cd['N_pairs']):.0f}",
    })

summary_disp_df = pd.DataFrame(summary_rows)
print(summary_disp_df.to_string(index=False))
print("=" * 80)


# In[20]:

# Compute velocity alignment curves (normalized and unnormalized)
alignment_curves_norm = compute_velocity_alignment_curves(
    graphs, key_file, observation_time, use_unnormalized=False
)

alignment_curves_unnorm = compute_velocity_alignment_curves(
    graphs, key_file, observation_time, use_unnormalized=True
)


# In[21]:


# Plot normalized velocity alignment  C_align(τ)  for all conditions
fig, ax = plt.subplots(figsize=(10, 6))

for condition in key_file["condition"].unique():
    cd = alignment_curves_norm[condition]
    tau = cd["tau"]
    mean = cd["C_mean"]
    sem = cd["C_sem"]

    ax.plot(tau, mean, marker="o", linewidth=2, label=condition)
    ax.fill_between(tau, mean - sem, mean + sem, alpha=0.2)

ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_xlabel("Time lag τ (frames)", fontsize=12, fontweight="bold")
ax.set_ylabel(r"$C_{\mathrm{align}}(\tau)$  (unit-vector dot product)", fontsize=12, fontweight="bold")
ax.set_title("Neighbor Velocity Alignment – Normalized", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(output_folder / subfolder / "velocity_alignment_normalized.pdf")
plt.savefig(output_folder / subfolder / "velocity_alignment_normalized.png")
plt.show()



# In[22]:


# Plot unnormalized (speed-weighted) velocity alignment for all conditions
fig, ax = plt.subplots(figsize=(10, 6))

for condition in key_file["condition"].unique():
    cd = alignment_curves_unnorm[condition]
    tau = cd["tau"]
    mean = cd["C_mean"]
    sem = cd["C_sem"]

    ax.plot(tau, mean, marker="o", linewidth=2, label=condition)
    ax.fill_between(tau, mean - sem, mean + sem, alpha=0.2)

ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_xlabel("Time lag τ (frames)", fontsize=12, fontweight="bold")
ax.set_ylabel(r"$\langle \mathbf{v}_i(t) \cdot \mathbf{v}_j(t+\tau) \rangle$  (µm²/frame²)",
              fontsize=12, fontweight="bold")
ax.set_title("Neighbor Velocity Alignment – Unnormalized (speed-weighted)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(output_folder / subfolder / "velocity_alignment_unnormalized.pdf")
plt.savefig(output_folder / subfolder / "velocity_alignment_unnormalized.png")
plt.show()


# In[23]:


# Per-condition subplots: normalized alignment  ±  1 std
conditions = key_file["condition"].unique()
n_cond = len(conditions)

fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), sharey=True)
if n_cond == 1:
    axes = [axes]

for ax, condition in zip(axes, conditions):
    cd = alignment_curves_norm[condition]
    tau = cd["tau"]
    mean = cd["C_mean"]
    std = cd["C_std"]

    ax.plot(tau, mean, "o-", color="darkorange", linewidth=2)
    ax.fill_between(tau, mean - std, mean + std, color="darkorange", alpha=0.15, label="± 1 std")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time lag τ (frames)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$C_{\mathrm{align}}(\tau)$", fontsize=11, fontweight="bold")
    ax.set_title(f"Condition: {condition}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(output_folder / subfolder / "velocity_alignment_normalized_per_condition.pdf")
plt.savefig(output_folder / subfolder / "velocity_alignment_normalized_per_condition.png")
plt.show()


# In[24]:

# Fit exponential decay to alignment curves and extract decorrelation time

alignment_fit_results = {}

for condition in key_file["condition"].unique():
    print(f"\n--- Fitting alignment decay for condition: {condition} ---")

    cd = alignment_curves_norm[condition]
    tau = cd["tau"]
    C = cd["C_mean"]
    C_std = cd["C_std"]

    # Use only τ >= 0 where C > 0 for exponential fit
    mask = C > 0
    if mask.sum() < 3:
        print("  Not enough positive points for fitting.")
        alignment_fit_results[condition] = {"fit_success": False}
        continue

    try:
        popt, pcov = curve_fit(
            exponential_decay,
            tau[mask], C[mask],
            p0=[C[0], 0.1],
            sigma=C_std[mask] + 1e-8,
            absolute_sigma=True,
            maxfev=5000,
        )
        C0, k = popt
        C0_err, k_err = np.sqrt(np.diag(pcov))

        # Decorrelation time (1/e time): τ_d = 1/k
        if k > 0:
            tau_d = 1.0 / k
            tau_d_err = k_err / (k ** 2)
        else:
            tau_d = np.inf
            tau_d_err = np.inf

        alignment_fit_results[condition] = {
            "C0": C0, "C0_err": C0_err,
            "k": k, "k_err": k_err,
            "tau_d": tau_d, "tau_d_err": tau_d_err,
            "fit_success": True,
        }

        print(f"  C0 (initial alignment): {C0:.4f} ± {C0_err:.4f}")
        print(f"  Decay rate k: {k:.4f} ± {k_err:.4f} frame⁻¹")
        print(f"  Decorrelation time τ_d = 1/k: {tau_d:.2f} ± {tau_d_err:.2f} frames")

    except Exception as e:
        print(f"  Fitting failed – {e}")
        alignment_fit_results[condition] = {"fit_success": False}


# In[25]:



# Plot alignment curves with exponential fits overlaid
conditions = key_file["condition"].unique()
n_cond = len(conditions)

fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), sharey=True)
if n_cond == 1:
    axes = [axes]

for ax, condition in zip(axes, conditions):
    cd = alignment_curves_norm[condition]
    tau = cd["tau"]
    C = cd["C_mean"]
    C_std = cd["C_std"]

    ax.errorbar(tau, C, yerr=C_std, marker="o", markersize=6, label="Data",
                linewidth=2, capsize=4, color="blue", alpha=0.7)

    res = alignment_fit_results[condition]
    if res.get("fit_success"):
        tau_smooth = np.linspace(tau[0], tau[-1], 200)
        C_fit = exponential_decay(tau_smooth, res["C0"], res["k"])
        ax.plot(tau_smooth, C_fit, "r-", linewidth=2.5,
                label=f"Fit (τ_d={res['tau_d']:.1f} f)")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time lag τ (frames)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$C_{\mathrm{align}}(\tau)$", fontsize=11, fontweight="bold")
    ax.set_title(f"Condition: {condition}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(output_folder / subfolder / "velocity_alignment_with_fits.pdf")
plt.savefig(output_folder / subfolder / "velocity_alignment_with_fits.png")
plt.show()


# In[26]:


# Summary table of velocity alignment metrics
print("\n" + "=" * 80)
print("NEIGHBOR VELOCITY ALIGNMENT SUMMARY")
print("=" * 80)

summary_rows_align = []
for condition in key_file["condition"].unique():
    cd_norm = alignment_curves_norm[condition]
    cd_unnorm = alignment_curves_unnorm[condition]

    row_data = {"Condition": condition}

    # Same-time alignment A(τ=0)
    if len(cd_norm["tau"]) > 0 and cd_norm["tau"][0] == 0:
        row_data["A(τ=0) norm"] = f"{cd_norm['C_mean'][0]:.4f} ± {cd_norm['C_std'][0]:.4f}"
    else:
        row_data["A(τ=0) norm"] = "N/A"

    if len(cd_unnorm["tau"]) > 0 and cd_unnorm["tau"][0] == 0:
        row_data["A(τ=0) unnorm (µm²/f²)"] = (
            f"{cd_unnorm['C_mean'][0]:.2f} ± {cd_unnorm['C_std'][0]:.2f}"
        )
    else:
        row_data["A(τ=0) unnorm (µm²/f²)"] = "N/A"

    # Decorrelation time from fit
    res = alignment_fit_results[condition]
    if res.get("fit_success"):
        row_data["τ_d (frames)"] = f"{res['tau_d']:.2f} ± {res['tau_d_err']:.2f}"
        row_data["C0"] = f"{res['C0']:.4f} ± {res['C0_err']:.4f}"
    else:
        row_data["τ_d (frames)"] = "N/A"
        row_data["C0"] = "N/A"

    row_data["Avg pairs per τ"] = f"{np.mean(cd_norm['N_pairs']):.0f}"

    summary_rows_align.append(row_data)

summary_align_df = pd.DataFrame(summary_rows_align)
print(summary_align_df.to_string(index=False))
print("=" * 80)



# In[27]:

# Compute flow-decomposed anisotropic separation and MSD
pair_sep_curves, msd_aniso_curves = compute_anisotropic_separation_and_msd(
    graphs, key_file, observation_time
)


# In[28]:


# ---- Plot 1: Pairwise separation ⟨δx²⟩ and ⟨δy²⟩ overlay ----
fig, ax = plt.subplots(figsize=(10, 6))

for condition in key_file["condition"].unique():
    cd = pair_sep_curves[condition]
    tau = cd["tau"]

    ax.plot(tau, cd["dx2_mean"], marker="o", linewidth=2,
            label=f"{condition} – ∥ flow (x)")
    ax.fill_between(tau, cd["dx2_mean"] - cd["dx2_sem"],
                     cd["dx2_mean"] + cd["dx2_sem"], alpha=0.15)

    ax.plot(tau, cd["dy2_mean"], marker="s", linewidth=2, linestyle="--",
            label=f"{condition} – ⊥ flow (y)")
    ax.fill_between(tau, cd["dy2_mean"] - cd["dy2_sem"],
                     cd["dy2_mean"] + cd["dy2_sem"], alpha=0.15)

ax.set_xlabel("Time lag τ (frames)", fontsize=12, fontweight="bold")
ax.set_ylabel(r"$\langle \delta x^2 \rangle$, $\langle \delta y^2 \rangle$  (µm²)",
              fontsize=12, fontweight="bold")
ax.set_title("Pairwise Neighbor Separation – Flow-Decomposed", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(output_folder / subfolder / "pairwise_separation_flow_decomposed.pdf")
plt.savefig(output_folder / subfolder / "pairwise_separation_flow_decomposed.png")
plt.show()


# In[29]:


# ---- Plot 2: Per-condition subplots of pairwise separation ----
conditions = key_file["condition"].unique()
n_cond = len(conditions)

fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), sharey=True)
if n_cond == 1:
    axes = [axes]

for ax, condition in zip(axes, conditions):
    cd = pair_sep_curves[condition]
    tau = cd["tau"]

    ax.plot(tau, cd["dx2_mean"], "o-", color="royalblue", linewidth=2, label="∥ flow (x)")
    ax.fill_between(tau, cd["dx2_mean"] - cd["dx2_std"],
                     cd["dx2_mean"] + cd["dx2_std"], color="royalblue", alpha=0.12, label="± 1 std")

    ax.plot(tau, cd["dy2_mean"], "s--", color="crimson", linewidth=2, label="⊥ flow (y)")
    ax.fill_between(tau, cd["dy2_mean"] - cd["dy2_std"],
                     cd["dy2_mean"] + cd["dy2_std"], color="crimson", alpha=0.12)

    ax.set_xlabel("Time lag τ (frames)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\langle \delta^2 \rangle$  (µm²)", fontsize=11, fontweight="bold")
    ax.set_title(f"Pair Separation – {condition}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(output_folder / subfolder / "pairwise_separation_per_condition.pdf")
plt.savefig(output_folder / subfolder / "pairwise_separation_per_condition.png")
plt.show()


# In[30]:


# ---- Plot 3: Anisotropic self-MSD overlay ----
fig, ax = plt.subplots(figsize=(10, 6))

for condition in key_file["condition"].unique():
    cd = msd_aniso_curves[condition]
    tau = cd["tau"]

    ax.plot(tau, cd["msd_par_mean"], marker="o", linewidth=2,
            label=f"{condition} – MSD∥ (x)")
    ax.fill_between(tau, cd["msd_par_mean"] - cd["msd_par_sem"],
                     cd["msd_par_mean"] + cd["msd_par_sem"], alpha=0.15)

    ax.plot(tau, cd["msd_perp_mean"], marker="s", linewidth=2, linestyle="--",
            label=f"{condition} – MSD⊥ (y)")
    ax.fill_between(tau, cd["msd_perp_mean"] - cd["msd_perp_sem"],
                     cd["msd_perp_mean"] + cd["msd_perp_sem"], alpha=0.15)

ax.set_xlabel("Time lag τ (frames)", fontsize=12, fontweight="bold")
ax.set_ylabel(r"MSD  (µm²)", fontsize=12, fontweight="bold")
ax.set_title("Anisotropic Self-MSD – Flow-Decomposed", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(output_folder / subfolder / "anisotropic_self_msd.pdf")
plt.savefig(output_folder / subfolder / "anisotropic_self_msd.png")
plt.show()


# In[31]:


# ---- Plot 4: Per-condition subplots of anisotropic self-MSD ----
conditions = key_file["condition"].unique()
n_cond = len(conditions)

fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), sharey=True)
if n_cond == 1:
    axes = [axes]

for ax, condition in zip(axes, conditions):
    cd = msd_aniso_curves[condition]
    tau = cd["tau"]

    ax.plot(tau, cd["msd_par_mean"], "o-", color="royalblue", linewidth=2, label="MSD∥ (x)")
    ax.fill_between(tau, cd["msd_par_mean"] - cd["msd_par_std"],
                     cd["msd_par_mean"] + cd["msd_par_std"], color="royalblue", alpha=0.12,
                     label="± 1 std")

    ax.plot(tau, cd["msd_perp_mean"], "s--", color="crimson", linewidth=2, label="MSD⊥ (y)")
    ax.fill_between(tau, cd["msd_perp_mean"] - cd["msd_perp_std"],
                     cd["msd_perp_mean"] + cd["msd_perp_std"], color="crimson", alpha=0.12)

    ax.set_xlabel("Time lag τ (frames)", fontsize=11, fontweight="bold")
    ax.set_ylabel("MSD  (µm²)", fontsize=11, fontweight="bold")
    ax.set_title(f"Self-MSD – {condition}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(output_folder / subfolder / "anisotropic_self_msd_per_condition.pdf")
plt.savefig(output_folder / subfolder / "anisotropic_self_msd_per_condition.png")
plt.show()


# In[32]:



# ---- Plot 5: Anisotropy ratio MSD∥ / MSD⊥ ----
fig, ax = plt.subplots(figsize=(10, 5))

for condition in key_file["condition"].unique():
    cd = msd_aniso_curves[condition]
    tau = cd["tau"]

    # Avoid division by zero
    ratio = np.where(cd["msd_perp_mean"] > 1e-12,
                     cd["msd_par_mean"] / cd["msd_perp_mean"],
                     np.nan)

    ax.plot(tau, ratio, marker="o", linewidth=2, label=condition)

ax.axhline(1, color="grey", linewidth=1, linestyle="--", label="Isotropic (ratio = 1)")
ax.set_xlabel("Time lag τ (frames)", fontsize=12, fontweight="bold")
ax.set_ylabel("MSD∥ / MSD⊥", fontsize=12, fontweight="bold")
ax.set_title("Self-MSD Anisotropy Ratio", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(output_folder / subfolder / "anisotropy_ratio.pdf")
plt.savefig(output_folder / subfolder / "anisotropy_ratio.png")
plt.show()


# In[33]:


# ---- Summary table ----
print("\n" + "=" * 90)
print("FLOW-DECOMPOSED SEPARATION & MSD SUMMARY")
print("=" * 90)

summary_rows_aniso = []
for condition in key_file["condition"].unique():
    cd_p = pair_sep_curves[condition]
    cd_m = msd_aniso_curves[condition]

    row_data = {"Condition": condition}

    # Pair separation at last available τ
    if len(cd_p["tau"]) > 0:
        last = -1
        tau_last = cd_p["tau"][last]
        row_data[f"⟨δx²⟩ τ={tau_last}"] = f"{cd_p['dx2_mean'][last]:.2f}"
        row_data[f"⟨δy²⟩ τ={tau_last}"] = f"{cd_p['dy2_mean'][last]:.2f}"

        pair_ratio = (cd_p["dx2_mean"][last] / cd_p["dy2_mean"][last]
                      if cd_p["dy2_mean"][last] > 1e-12 else np.nan)
        row_data["Pair ∥/⊥ ratio"] = f"{pair_ratio:.2f}"
    else:
        row_data[f"⟨δx²⟩ last τ"] = "N/A"
        row_data[f"⟨δy²⟩ last τ"] = "N/A"
        row_data["Pair ∥/⊥ ratio"] = "N/A"

    # Self-MSD at last available τ
    if len(cd_m["tau"]) > 0:
        last = -1
        tau_last_m = cd_m["tau"][last]
        row_data[f"MSD∥ τ={tau_last_m}"] = f"{cd_m['msd_par_mean'][last]:.2f}"
        row_data[f"MSD⊥ τ={tau_last_m}"] = f"{cd_m['msd_perp_mean'][last]:.2f}"

        msd_ratio = (cd_m["msd_par_mean"][last] / cd_m["msd_perp_mean"][last]
                     if cd_m["msd_perp_mean"][last] > 1e-12 else np.nan)
        row_data["MSD ∥/⊥ ratio"] = f"{msd_ratio:.2f}"
    else:
        row_data[f"MSD∥ last τ"] = "N/A"
        row_data[f"MSD⊥ last τ"] = "N/A"
        row_data["MSD ∥/⊥ ratio"] = "N/A"

    row_data["Avg pairs/τ"] = (f"{np.mean(cd_p['N_pairs']):.0f}"
                                if len(cd_p["N_pairs"]) else "N/A")
    row_data["Avg cells/τ"] = (f"{np.mean(cd_m['N_cells']):.0f}"
                                if len(cd_m["N_cells"]) else "N/A")

    summary_rows_aniso.append(row_data)

summary_aniso_df = pd.DataFrame(summary_rows_aniso)
print(summary_aniso_df.to_string(index=False))
print("=" * 90)



# In[ ]:


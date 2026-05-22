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
from src.computation import compute_instantaneous_neighbor_velocity_alignment
from src.graph_analysis import (
    build_time_graphs,
    compute_neighbor_lifetimes,
    compute_neighbor_retention_curve,
    exponential_decay,
    compute_relative_neighbor_displacement,
    compute_velocity_alignment_curves,
    compute_anisotropic_separation_and_msd,
    cross_experiment_stats, )
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

n_jobs = parameters.get("n_jobs", None)
print("Parallel graph construction n_jobs:", n_jobs if n_jobs is not None else "all cores")

# --- Analysis module flags (can be toggled in the parameters YAML) ---
run_node_trajectory_viz    = parameters.get("run_node_trajectory_viz",    True)
run_neighbor_lifetimes     = parameters.get("run_neighbor_lifetimes",     True)
run_neighbor_retention     = parameters.get("run_neighbor_retention",     True)
run_cage_relative_displacement = parameters.get("run_cage_relative_displacement", True)
run_velocity_alignment     = parameters.get("run_velocity_alignment",     True)
run_anisotropic_msd        = parameters.get("run_anisotropic_msd",        True)

print("Analysis modules enabled:")
for name, flag in [
    ("node_trajectory_viz",        run_node_trajectory_viz),
    ("neighbor_lifetimes",         run_neighbor_lifetimes),
    ("neighbor_retention",         run_neighbor_retention),
    ("cage_relative_displacement", run_cage_relative_displacement),
    ("velocity_alignment",         run_velocity_alignment),
    ("anisotropic_msd",            run_anisotropic_msd),
]:
    print(f"  {name}: {'ON' if flag else 'OFF'}")


# In[5]:

num_subsample = parameters["num_subsample"]

# ----------------------------------------------------------------------
# Helpers for trustworthiness-aware plotting
# ----------------------------------------------------------------------
# Per-experiment samples used inside a single replicate to compute ⟨·⟩(τ)
# are highly non-iid (they share trajectories and overlapping time
# windows), so the within-replicate ``std/sqrt(N)`` SEM is optimistic.
# Biological replicates, in contrast, are independent realisations.  The
# plotting below therefore prefers:
#   (1) mean ± SEM across experiments (proper independent-replicate bar),
#   (2) an overlay of N_pairs / N_cells on a twin axis so the reader
#       sees where the sliding-window statistics thin out,
#   (3) shading of the τ > τ_max/2 region, which is known to be
#       unreliable for sliding-window time-lag estimators regardless of
#       the error bar used.


def _low_stat_tau_threshold(tau_array):
    """Return τ above which sliding-window statistics are considered
    unreliable (τ > half the maximal available lag)."""
    if len(tau_array) == 0:
        return None
    return tau_array[-1] / 2.0


def plot_curve_with_trust(
    ax,
    condition_curves,
    value_key,
    n_key,
    color=None,
    label=None,
    ylabel="",
    show_n_axis=True,
    shade_low_stat=True,
):
    """Plot one condition's curve with cross-experiment SEM and N overlay.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    condition_curves : dict
        One condition's entry of the condition-level dict returned by any
        of the graph_analysis functions.  Must contain "tau",
        ``value_key``, ``n_key`` and "per_experiment".
    value_key : str
        Key for the aggregate curve inside ``condition_curves`` and inside
        each per-experiment dict (e.g. "Sn", "delta_r_mean", "C_mean",
        "dx2_mean", "msd_par_mean").
    n_key : str
        Key for the sample count in ``condition_curves`` (e.g. "N_pairs",
        "N_cells", "N_measurements").
    show_n_axis : bool
        If True, overlay sample count on a twin axis (greyed out).
    shade_low_stat : bool
        If True, shade the region τ > τ_max/2 to flag low-statistics.
    """
    tau = condition_curves["tau"]
    if len(tau) == 0:
        return None

    # --- cross-replicate curve ---
    cx = cross_experiment_stats(
        condition_curves["per_experiment"], value_key=value_key
    )
    if len(cx["tau"]) > 0:
        ax.plot(cx["tau"], cx["mean"], marker="o", linewidth=2,
                color=color, label=label)
        if np.any(cx["sem"] > 0):
            ax.fill_between(cx["tau"], cx["mean"] - cx["sem"],
                             cx["mean"] + cx["sem"], alpha=0.25, color=color,
                             linewidth=0)

    # --- low-statistics shading ---
    if shade_low_stat:
        cutoff = _low_stat_tau_threshold(tau)
        if cutoff is not None and cutoff < tau[-1]:
            ax.axvspan(cutoff, tau[-1], color="grey", alpha=0.08, zorder=0)

    # --- N overlay on twin axis ---
    if show_n_axis:
        ax2 = ax.twinx()
        ax2.bar(tau, condition_curves[n_key], width=0.8, alpha=0.18,
                color="grey", zorder=0)
        ax2.set_ylabel(f"{n_key} (sliding-window samples)",
                        fontsize=9, color="grey")
        ax2.tick_params(axis="y", colors="grey", labelsize=8)
        ax2.spines["right"].set_color("grey")
        ax2.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)

    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    return ax



# In[6]:

# Build Delaunay graphs per time point for each experiment
graphs, observation_period_dfs = build_time_graphs(
    key_file, data_folder, observation_time, distance_threshold, n_jobs=n_jobs
)


# In[7]:

# Choose a node ID to track (must be a TRACK_ID that exists in your graphs)
node_to_track = 123  # Replace with your desired node ID

# Visualize node trajectory for each experiment (organized by condition)
if run_node_trajectory_viz:
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
else:
    print("Skipping node trajectory visualisation (run_node_trajectory_viz=False)")


# Compute neighbor lifetimes per condition
if run_neighbor_lifetimes:
    neighbor_lifetimes_by_condition = compute_neighbor_lifetimes(
        graphs, observation_period_dfs, key_file, n_jobs=n_jobs
    )


# In[9]:

# randomly sample some tracks and plot histograms of neighbor lifetimes
if run_neighbor_lifetimes:
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
else:
    print("Skipping neighbor lifetimes (run_neighbor_lifetimes=False)")


# In[10]:

# ---------------------------------------------------------------------------
# Instantaneous neighbor velocity alignment over time
# ---------------------------------------------------------------------------

if run_velocity_alignment:
    instantaneous_alignment_norm = compute_instantaneous_neighbor_velocity_alignment(
        graphs, key_file, observation_time,
        velocity_window=3, use_unnormalized=False, n_jobs=n_jobs
    )

    instantaneous_alignment_unnorm = compute_instantaneous_neighbor_velocity_alignment(
        graphs, key_file, observation_time,
        velocity_window=3, use_unnormalized=True, n_jobs=n_jobs
    )
else:
    print("Skipping instantaneous velocity alignment (run_velocity_alignment=False)")


# In[10b]:

if run_velocity_alignment:
    # Helper: cross-replicate stats for frame-indexed per_experiment dicts
    def _cross_replicate_stats_frame(per_experiment):
        all_frames = set()
        for ed in per_experiment.values():
            all_frames.update(ed["frame"].tolist())
        frames_sorted = np.array(sorted(all_frames))
        means, sems, ns = [], [], []
        out_frames = []
        for f in frames_sorted:
            vals = [ed["C_mean"][np.where(ed["frame"] == f)[0][0]]
                    for ed in per_experiment.values()
                    if len(np.where(ed["frame"] == f)[0]) == 1]
            if vals:
                out_frames.append(f)
                vals = np.asarray(vals, dtype=float)
                means.append(np.mean(vals))
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
                ns.append(len(vals))
        return {
            "frame": np.array(out_frames),
            "mean": np.array(means),
            "sem": np.array(sems),
            "n_experiments": np.array(ns),
        }

    # Plot 1: Normalized — all conditions overlaid
    fig, ax = plt.subplots(figsize=(10, 6))
    for condition in key_file["condition"].unique():
        cd = instantaneous_alignment_norm[condition]
        cx = _cross_replicate_stats_frame(cd["per_experiment"])
        if len(cx["frame"]) == 0:
            continue
        line, = ax.plot(cx["frame"], cx["mean"], linewidth=2,
                        label=f"{condition}  (n={int(np.max(cx['n_experiments']))} exp)")
        if np.any(cx["sem"] > 0):
            ax.fill_between(cx["frame"], cx["mean"] - cx["sem"], cx["mean"] + cx["sem"],
                            alpha=0.25, color=line.get_color(), linewidth=0)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Frame", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$C_{\mathrm{align}}(t)$  (unit-vector dot product) — replicate mean ± SEM",
                  fontsize=12, fontweight="bold")
    ax.set_title("Instantaneous Neighbor Velocity Alignment – Normalized",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_folder / subfolder / "instantaneous_velocity_alignment_normalized.pdf")
    plt.savefig(output_folder / subfolder / "instantaneous_velocity_alignment_normalized.png")
    plt.show()

    # Plot 2: Unnormalized (speed-weighted) — all conditions overlaid
    fig, ax = plt.subplots(figsize=(10, 6))
    for condition in key_file["condition"].unique():
        cd = instantaneous_alignment_unnorm[condition]
        cx = _cross_replicate_stats_frame(cd["per_experiment"])
        if len(cx["frame"]) == 0:
            continue
        line, = ax.plot(cx["frame"], cx["mean"], linewidth=2,
                        label=f"{condition}  (n={int(np.max(cx['n_experiments']))} exp)")
        if np.any(cx["sem"] > 0):
            ax.fill_between(cx["frame"], cx["mean"] - cx["sem"], cx["mean"] + cx["sem"],
                            alpha=0.25, color=line.get_color(), linewidth=0)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Frame", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$\langle \mathbf{v}_i(t) \cdot \mathbf{v}_j(t) \rangle$  (µm²/frame²) — replicate mean ± SEM",
                  fontsize=12, fontweight="bold")
    ax.set_title("Instantaneous Neighbor Velocity Alignment – Unnormalized (speed-weighted)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_folder / subfolder / "instantaneous_velocity_alignment_unnormalized.pdf")
    plt.savefig(output_folder / subfolder / "instantaneous_velocity_alignment_unnormalized.png")
    plt.show()

    # Plot 3: Per-condition subplots with N_pairs overlay (normalized)
    conditions = list(key_file["condition"].unique())
    n_cond = len(conditions)
    fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]
    for ax, condition in zip(axes, conditions):
        cd = instantaneous_alignment_norm[condition]
        cx = _cross_replicate_stats_frame(cd["per_experiment"])
        if len(cx["frame"]) == 0:
            ax.set_visible(False)
            continue
        color = "darkorange"
        ax.plot(cx["frame"], cx["mean"], color=color, linewidth=2)
        if np.any(cx["sem"] > 0):
            ax.fill_between(cx["frame"], cx["mean"] - cx["sem"], cx["mean"] + cx["sem"],
                            alpha=0.25, color=color, linewidth=0)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax2 = ax.twinx()
        ax2.bar(cd["frame"], cd["N_pairs"], color="grey", alpha=0.18, width=1.0)
        ax2.set_ylabel("N pairs", fontsize=9, color="grey")
        ax2.tick_params(axis="y", labelcolor="grey", labelsize=8)
        ax.set_xlabel("Frame", fontsize=11, fontweight="bold")
        ax.set_ylabel(r"$C_{\mathrm{align}}(t)$", fontsize=11, fontweight="bold")
        ax.set_title(f"Condition: {condition}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_folder / subfolder / "instantaneous_velocity_alignment_per_condition.pdf")
    plt.savefig(output_folder / subfolder / "instantaneous_velocity_alignment_per_condition.png")
    plt.show()


# In[11]:

# Compute the neighbor retention / survival curves
if run_neighbor_retention:
    survival_curves = compute_neighbor_retention_curve(graphs, key_file, observation_time, n_jobs=n_jobs)


# In[12]:

if run_neighbor_retention:
    # Plot neighbor retention curves for all conditions
    # Uses cross-experiment SEM (independent replicates) instead of within-replicate
    # std/sqrt(N), which is optimistic due to correlated sliding-window samples.
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in key_file["condition"].unique():
        curve_data = survival_curves[condition]
        cx = cross_experiment_stats(curve_data["per_experiment"], value_key="Sn")
        if len(cx["tau"]) == 0:
            continue
        line, = ax.plot(cx["tau"], cx["mean"], marker="o", linewidth=2,
                        label=f"{condition}  (n={int(np.max(cx['n_experiments']))} exp)")
        if np.any(cx["sem"] > 0):
            ax.fill_between(cx["tau"], cx["mean"] - cx["sem"], cx["mean"] + cx["sem"],
                            alpha=0.25, color=line.get_color(), linewidth=0)

    # shade low-statistics region (τ > τ_max/2) – sliding-window estimator unreliable
    tau_max_global = max(
        (survival_curves[c]["tau"][-1] for c in key_file["condition"].unique()
         if len(survival_curves[c]["tau"]) > 0),
        default=None,
    )
    if tau_max_global is not None:
        ax.axvspan(tau_max_global / 2.0, tau_max_global, color="grey", alpha=0.08,
                   label="τ > τ_max/2 (low stat.)")


    ax.set_xlabel('Time lag τ (frames)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Neighbor Retention Sₙ(τ) — replicate mean ± SEM', fontsize=12, fontweight='bold')
    ax.set_title('Neighbor Retention/Survival Curves by Condition', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_folder / subfolder / "neighbor_retention_curves.pdf")
    plt.savefig(output_folder / subfolder / "neighbor_retention_curves.png")
    plt.show()




    # In[12b]:

    # Per-condition retention with sample-count overlay
    conditions = list(key_file["condition"].unique())
    n_cond = len(conditions)
    fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]
    for ax, condition in zip(axes, conditions):
        plot_curve_with_trust(
            ax, survival_curves[condition],
            value_key="Sn", n_key="N_measurements",
            color="steelblue",
            ylabel="Neighbor Retention Sₙ(τ)",
        )
        ax.set_xlabel("Time lag τ (frames)", fontsize=11, fontweight="bold")
        ax.set_title(f"Condition: {condition}", fontsize=12, fontweight="bold")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_folder / subfolder / "neighbor_retention_curves_trust.pdf")
    plt.savefig(output_folder / subfolder / "neighbor_retention_curves_trust.png")
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
else:
    print("Skipping neighbor retention (run_neighbor_retention=False)")


# In[16]:

# Compute cage-relative neighbor displacement curves
if run_cage_relative_displacement:
    displacement_curves = compute_relative_neighbor_displacement(graphs, key_file, observation_time, n_jobs=n_jobs)


# In[17]:

if run_cage_relative_displacement:
    # Plot cage-relative displacement ⟨δr(τ)⟩ for all conditions
    # Band = SEM across experiments (independent replicates); shaded τ-region flags
    # sliding-window low-statistics (τ > τ_max/2).
    fig, ax = plt.subplots(figsize=(10, 6))

    tau_max_global = None
    for condition in key_file["condition"].unique():
        cd = displacement_curves[condition]
        cx = cross_experiment_stats(cd["per_experiment"], value_key="delta_r_mean")
        if len(cx["tau"]) == 0:
            continue
        line, = ax.plot(cx["tau"], cx["mean"], marker="o", linewidth=2,
                        label=f"{condition}  (n={int(np.max(cx['n_experiments']))} exp)")
        if np.any(cx["sem"] > 0):
            ax.fill_between(cx["tau"], cx["mean"] - cx["sem"], cx["mean"] + cx["sem"],
                            alpha=0.25, color=line.get_color(), linewidth=0)
        if len(cd["tau"]) and (tau_max_global is None or cd["tau"][-1] > tau_max_global):
            tau_max_global = cd["tau"][-1]

    if tau_max_global is not None:
        ax.axvspan(tau_max_global / 2.0, tau_max_global, color="grey", alpha=0.08,
                   label="τ > τ_max/2 (low stat.)")


    ax.set_xlabel("Time lag τ (frames)", fontsize=12, fontweight="bold")
    ax.set_ylabel("⟨δr(τ)⟩  (µm) — replicate mean ± SEM", fontsize=12, fontweight="bold")
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


    # Per-condition subplots with pair-count overlay and low-statistics shading
    conditions = key_file["condition"].unique()
    n_cond = len(conditions)

    fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for ax, condition in zip(axes, conditions):
        plot_curve_with_trust(
            ax, displacement_curves[condition],
            value_key="delta_r_mean", n_key="N_pairs",
            color="teal",
            ylabel="⟨δr(τ)⟩  (µm)",
        )
        ax.set_xlabel("Time lag τ (frames)", fontsize=11, fontweight="bold")
        ax.set_title(f"Condition: {condition}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

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
else:
    print("Skipping cage-relative displacement (run_cage_relative_displacement=False)")


# In[20]:

# Compute velocity alignment curves (normalized and unnormalized)
if run_velocity_alignment:
    alignment_curves_norm = compute_velocity_alignment_curves(
        graphs, key_file, observation_time, use_unnormalized=False, n_jobs=n_jobs
    )

    alignment_curves_unnorm = compute_velocity_alignment_curves(
        graphs, key_file, observation_time, use_unnormalized=True, n_jobs=n_jobs
    )


# In[21]:

if run_velocity_alignment:
    # Plot normalized velocity alignment  C_align(τ)  for all conditions
    # Band = SEM across experiments (independent replicates).
    fig, ax = plt.subplots(figsize=(10, 6))

    tau_max_global = None
    for condition in key_file["condition"].unique():
        cd = alignment_curves_norm[condition]
        cx = cross_experiment_stats(cd["per_experiment"], value_key="C_mean")
        if len(cx["tau"]) == 0:
            continue
        line, = ax.plot(cx["tau"], cx["mean"], marker="o", linewidth=2,
                        label=f"{condition}  (n={int(np.max(cx['n_experiments']))} exp)")
        if np.any(cx["sem"] > 0):
            ax.fill_between(cx["tau"], cx["mean"] - cx["sem"], cx["mean"] + cx["sem"],
                            alpha=0.25, color=line.get_color(), linewidth=0)
        if len(cd["tau"]) and (tau_max_global is None or cd["tau"][-1] > tau_max_global):
            tau_max_global = cd["tau"][-1]

    if tau_max_global is not None:
        ax.axvspan(tau_max_global / 2.0, tau_max_global, color="grey", alpha=0.08,
                   label="τ > τ_max/2 (low stat.)")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time lag τ (frames)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$C_{\mathrm{align}}(\tau)$  (unit-vector dot product) — replicate mean ± SEM",
                  fontsize=12, fontweight="bold")
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
        cx = cross_experiment_stats(cd["per_experiment"], value_key="C_mean")
        if len(cx["tau"]) == 0:
            continue
        line, = ax.plot(cx["tau"], cx["mean"], marker="o", linewidth=2,
                        label=f"{condition}  (n={int(np.max(cx['n_experiments']))} exp)")
        if np.any(cx["sem"] > 0):
            ax.fill_between(cx["tau"], cx["mean"] - cx["sem"], cx["mean"] + cx["sem"],
                            alpha=0.25, color=line.get_color(), linewidth=0)

    if tau_max_global is not None:
        ax.axvspan(tau_max_global / 2.0, tau_max_global, color="grey", alpha=0.08,
                   label="τ > τ_max/2 (low stat.)")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time lag τ (frames)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$\langle \mathbf{v}_i(t) \cdot \mathbf{v}_j(t+\tau) \rangle$  (µm²/frame²) — replicate mean ± SEM",
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

    # Per-condition subplots: normalized alignment with replicate SEM + pair-count overlay
    conditions = key_file["condition"].unique()
    n_cond = len(conditions)

    fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for ax, condition in zip(axes, conditions):
        plot_curve_with_trust(
            ax, alignment_curves_norm[condition],
            value_key="C_mean", n_key="N_pairs",
            color="darkorange",
            ylabel=r"$C_{\mathrm{align}}(\tau)$",
        )
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time lag τ (frames)", fontsize=11, fontweight="bold")
        ax.set_title(f"Condition: {condition}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

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
else:
    print("Skipping velocity alignment (run_velocity_alignment=False)")



# In[27]:

# Compute flow-decomposed anisotropic separation and MSD
if run_anisotropic_msd:
    pair_sep_curves, msd_aniso_curves = compute_anisotropic_separation_and_msd(
        graphs, key_file, observation_time, n_jobs=n_jobs
    )


# In[28]:

if run_anisotropic_msd:
    # ---- Plot 1: Pairwise separation ⟨δx²⟩ and ⟨δy²⟩ overlay ----
    fig, ax = plt.subplots(figsize=(10, 6))

    tau_max_global_ps = None
    for condition in key_file["condition"].unique():
        cd = pair_sep_curves[condition]
        cx_x = cross_experiment_stats(cd["per_experiment"], value_key="dx2_mean")
        cx_y = cross_experiment_stats(cd["per_experiment"], value_key="dy2_mean")
        if len(cx_x["tau"]):
            lx, = ax.plot(cx_x["tau"], cx_x["mean"], marker="o", linewidth=2,
                          label=f"{condition} – ∥ flow (x)")
            if np.any(cx_x["sem"] > 0):
                ax.fill_between(cx_x["tau"], cx_x["mean"] - cx_x["sem"],
                                 cx_x["mean"] + cx_x["sem"], alpha=0.20, color=lx.get_color(),
                                 linewidth=0)
        if len(cx_y["tau"]):
            ly, = ax.plot(cx_y["tau"], cx_y["mean"], marker="s", linewidth=2, linestyle="--",
                          label=f"{condition} – ⊥ flow (y)")
            if np.any(cx_y["sem"] > 0):
                ax.fill_between(cx_y["tau"], cx_y["mean"] - cx_y["sem"],
                                 cx_y["mean"] + cx_y["sem"], alpha=0.20, color=ly.get_color(),
                                 linewidth=0)
        if len(cd["tau"]) and (tau_max_global_ps is None or cd["tau"][-1] > tau_max_global_ps):
            tau_max_global_ps = cd["tau"][-1]

    if tau_max_global_ps is not None:
        ax.axvspan(tau_max_global_ps / 2.0, tau_max_global_ps, color="grey", alpha=0.08,
                   label="τ > τ_max/2 (low stat.)")

    ax.set_xlabel("Time lag τ (frames)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"$\langle \delta x^2 \rangle$, $\langle \delta y^2 \rangle$  (µm²) — replicate mean ± SEM",
                  fontsize=12, fontweight="bold")
    ax.set_title("Pairwise Neighbor Separation – Flow-Decomposed", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_folder / subfolder / "pairwise_separation_flow_decomposed.pdf")
    plt.savefig(output_folder / subfolder / "pairwise_separation_flow_decomposed.png")
    plt.show()


    # In[29]:

    # ---- Plot 2: Per-condition subplots of pairwise separation with trust overlays ----
    conditions = key_file["condition"].unique()
    n_cond = len(conditions)

    fig, axes = plt.subplots(1, n_cond, figsize=(6 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for ax, condition in zip(axes, conditions):
        cd = pair_sep_curves[condition]
        cx_x = cross_experiment_stats(cd["per_experiment"], value_key="dx2_mean")
        cx_y = cross_experiment_stats(cd["per_experiment"], value_key="dy2_mean")

        if len(cx_x["tau"]):
            ax.plot(cx_x["tau"], cx_x["mean"], "o-", color="royalblue", linewidth=2,
                    label="∥ flow (x)")
            if np.any(cx_x["sem"] > 0):
                ax.fill_between(cx_x["tau"], cx_x["mean"] - cx_x["sem"],
                                 cx_x["mean"] + cx_x["sem"], color="royalblue", alpha=0.20,
                                 linewidth=0, label="± SEM (replicates)")
        if len(cx_y["tau"]):
            ax.plot(cx_y["tau"], cx_y["mean"], "s--", color="crimson", linewidth=2,
                    label="⊥ flow (y)")
            if np.any(cx_y["sem"] > 0):
                ax.fill_between(cx_y["tau"], cx_y["mean"] - cx_y["sem"],
                                 cx_y["mean"] + cx_y["sem"], color="crimson", alpha=0.20,
                                 linewidth=0)

        cutoff = _low_stat_tau_threshold(cd["tau"])
        if cutoff is not None and cutoff < cd["tau"][-1]:
            ax.axvspan(cutoff, cd["tau"][-1], color="grey", alpha=0.08, zorder=0)

        # N overlay
        ax2 = ax.twinx()
        ax2.bar(cd["tau"], cd["N_pairs"], width=0.8, alpha=0.18, color="grey", zorder=0)
        ax2.set_ylabel("N_pairs (sliding-window samples)", fontsize=9, color="grey")
        ax2.tick_params(axis="y", colors="grey", labelsize=8)
        ax2.spines["right"].set_color("grey")
        ax2.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)

        ax.set_xlabel("Time lag τ (frames)", fontsize=11, fontweight="bold")
        ax.set_ylabel(r"$\langle \delta^2 \rangle$  (µm²)", fontsize=11, fontweight="bold")
        ax.set_title(f"Pair Separation – {condition}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_folder / subfolder / "pairwise_separation_per_condition.pdf")
    plt.savefig(output_folder / subfolder / "pairwise_separation_per_condition.png")
    plt.show()


    # In[30]:

    # ---- Plot 3: Anisotropic self-MSD overlay (replicate-SEM error bands) ----
    fig, ax = plt.subplots(figsize=(10, 6))

    tau_max_global_ms = None
    for condition in key_file["condition"].unique():
        cd = msd_aniso_curves[condition]
        cx_par = cross_experiment_stats(cd["per_experiment"], value_key="msd_par_mean")
        cx_perp = cross_experiment_stats(cd["per_experiment"], value_key="msd_perp_mean")
        if len(cx_par["tau"]):
            lp, = ax.plot(cx_par["tau"], cx_par["mean"], marker="o", linewidth=2,
                          label=f"{condition} – MSD∥ (x)")
            if np.any(cx_par["sem"] > 0):
                ax.fill_between(cx_par["tau"], cx_par["mean"] - cx_par["sem"],
                                 cx_par["mean"] + cx_par["sem"], alpha=0.20,
                                 color=lp.get_color(), linewidth=0)
        if len(cx_perp["tau"]):
            lr, = ax.plot(cx_perp["tau"], cx_perp["mean"], marker="s", linewidth=2,
                          linestyle="--", label=f"{condition} – MSD⊥ (y)")
            if np.any(cx_perp["sem"] > 0):
                ax.fill_between(cx_perp["tau"], cx_perp["mean"] - cx_perp["sem"],
                                 cx_perp["mean"] + cx_perp["sem"], alpha=0.20,
                                 color=lr.get_color(), linewidth=0)
        if len(cd["tau"]) and (tau_max_global_ms is None or cd["tau"][-1] > tau_max_global_ms):
            tau_max_global_ms = cd["tau"][-1]

    if tau_max_global_ms is not None:
        ax.axvspan(tau_max_global_ms / 2.0, tau_max_global_ms, color="grey", alpha=0.08,
                   label="τ > τ_max/2 (low stat.)")

    ax.set_xlabel("Time lag τ (frames)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"MSD  (µm²) — replicate mean ± SEM", fontsize=12, fontweight="bold")
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
        cx_par = cross_experiment_stats(cd["per_experiment"], value_key="msd_par_mean")
        cx_perp = cross_experiment_stats(cd["per_experiment"], value_key="msd_perp_mean")

        if len(cx_par["tau"]):
            ax.plot(cx_par["tau"], cx_par["mean"], "o-", color="royalblue", linewidth=2,
                    label="MSD∥ (x)")
            if np.any(cx_par["sem"] > 0):
                ax.fill_between(cx_par["tau"], cx_par["mean"] - cx_par["sem"],
                                 cx_par["mean"] + cx_par["sem"], color="royalblue",
                                 alpha=0.20, linewidth=0, label="± SEM (replicates)")
        if len(cx_perp["tau"]):
            ax.plot(cx_perp["tau"], cx_perp["mean"], "s--", color="crimson", linewidth=2,
                    label="MSD⊥ (y)")
            if np.any(cx_perp["sem"] > 0):
                ax.fill_between(cx_perp["tau"], cx_perp["mean"] - cx_perp["sem"],
                                 cx_perp["mean"] + cx_perp["sem"], color="crimson",
                                 alpha=0.20, linewidth=0)

        cutoff = _low_stat_tau_threshold(cd["tau"])
        if cutoff is not None and cutoff < cd["tau"][-1]:
            ax.axvspan(cutoff, cd["tau"][-1], color="grey", alpha=0.08, zorder=0)

        # N_cells overlay
        ax2 = ax.twinx()
        ax2.bar(cd["tau"], cd["N_cells"], width=0.8, alpha=0.18, color="grey", zorder=0)
        ax2.set_ylabel("N_cells (sliding-window samples)", fontsize=9, color="grey")
        ax2.tick_params(axis="y", colors="grey", labelsize=8)
        ax2.spines["right"].set_color("grey")
        ax2.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)

        ax.set_xlabel("Time lag τ (frames)", fontsize=11, fontweight="bold")
        ax.set_ylabel("MSD  (µm²)", fontsize=11, fontweight="bold")
        ax.set_title(f"Self-MSD – {condition}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="upper left")

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
else:
    print("Skipping anisotropic MSD (run_anisotropic_msd=False)")

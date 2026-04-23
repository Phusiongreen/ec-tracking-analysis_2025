"""
Time-resolved graph analysis for endothelial cell tracking.

Functions for building Delaunay neighbour graphs at each frame,
computing neighbor lifetimes / retention curves, cage-relative
displacement, velocity alignment, and anisotropic MSD.
"""

import numpy as np
import pandas as pd
import networkx as nx
from griottes import generate_delaunay_graph


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_time_graphs(key_file, data_folder, observation_time, distance_threshold):
    """Build Delaunay graphs per frame for every experiment in *key_file*.

    Parameters
    ----------
    key_file : pd.DataFrame
        Must contain columns ``experimentID``, ``treatment``, ``color``.
    data_folder : pathlib.Path
        Folder containing per-experiment tracking CSVs (e.g. ``tracking_data/``).
    observation_time : tuple (start_frame, end_frame)
    distance_threshold : float
        Maximum edge length for the Delaunay graph.

    Returns
    -------
    graphs : dict
        ``{experimentID: [G_t0, G_t1, …]}`` — one graph per frame.
    observation_period_dfs : dict
        ``{experimentID: DataFrame}`` — filtered tracking data for the
        observation window.
    """
    graphs = {}
    observation_period_dfs = {}

    for experimentID in key_file["experimentID"].unique():
        print("Processing experiment:", experimentID)

        key_exp = key_file[key_file["experimentID"] == experimentID]
        row = key_exp.iloc[0]
        treatment = row["treatment"]

        tracking_file = "tracking_data_%s_%s_%s.csv" % (
            treatment, row["color"], row["experimentID"]
        )
        data = pd.read_csv(data_folder / tracking_file, low_memory=False)

        # filter for observation period
        observation_period_df = data[data["FRAME"] <= observation_time[1]]
        observation_period_df = observation_period_df[
            observation_period_df["FRAME"] >= observation_time[0]
        ]
        observation_period_dfs[experimentID] = observation_period_df

        # rename columns for griottes
        obs_df_renamed = observation_period_df.rename(
            columns={"POSITION_X": "x", "POSITION_Y": "y"}
        )
        obs_df_renamed["label"] = obs_df_renamed["TRACK_ID"]

        t_graphs = []
        for t in range(observation_time[0], observation_time[1]):
            frame_df = obs_df_renamed[obs_df_renamed["FRAME"] == t].reset_index(drop=True)

            print("Generating graph at time point %d ..." % t)
            G_delaunay = generate_delaunay_graph(
                frame_df[frame_df.columns],
                descriptors=frame_df.columns,
                distance=distance_threshold,
                image_is_2D=True,
            )
            print("Graph generated!")

            # relabel nodes to TRACK_IDs
            mapping = {n: G_delaunay.nodes[n]["TRACK_ID"] for n in G_delaunay.nodes}
            G_delaunay = nx.relabel_nodes(G_delaunay, mapping)

            t_graphs.append(G_delaunay)

        graphs[experimentID] = t_graphs

    return graphs, observation_period_dfs


# ---------------------------------------------------------------------------
# Neighbor lifetimes
# ---------------------------------------------------------------------------

def compute_neighbor_lifetimes(graphs, observation_period_dfs, key_file):
    """Compute consecutive-neighbor lifetimes for every condition.

    Parameters
    ----------
    graphs : dict
        ``{experimentID: [G_t0, G_t1, …]}``.
    observation_period_dfs : dict
        ``{experimentID: DataFrame}`` with at least a ``TRACK_ID`` column.
    key_file : pd.DataFrame
        Must contain ``condition`` and ``experimentID``.

    Returns
    -------
    neighbor_lifetimes_by_condition : dict
        ``{condition: {track_id: {neighbor_id: lifetime_frames}}}``
    """
    neighbor_lifetimes_by_condition = {}

    for condition in key_file["condition"].unique():
        print(f"\nAnalyzing condition: {condition}")

        experiments_for_condition = key_file[key_file["condition"] == condition]
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

            t_graphs = graphs[experimentID]
            trackIDs = observation_period_df["TRACK_ID"].unique()

            for track_id in trackIDs:
                neighbor_lifetimes = {}

                t_graphs0 = t_graphs[0]
                if track_id not in t_graphs0.nodes(data="TRACK_ID"):
                    continue

                neighbors0 = list(t_graphs0.neighbors(track_id))
                for neighbor in neighbors0:
                    neighbor_lifetimes[neighbor] = 1

                for t, G_delaunay in enumerate(t_graphs[1:], start=1):
                    node_track = [
                        n
                        for n, d in G_delaunay.nodes(data=True)
                        if d.get("TRACK_ID") == track_id
                    ]
                    if len(node_track) > 1:
                        continue
                    elif len(node_track) == 0:
                        break

                    a = node_track[0]
                    neighbors_t = G_delaunay[a]

                    for neighbor in neighbors_t:
                        if neighbor in neighbor_lifetimes:
                            if neighbor_lifetimes[neighbor] == t:
                                neighbor_lifetimes[neighbor] += 1

                neighbor_lifetimes_all_tracks[track_id] = neighbor_lifetimes

        neighbor_lifetimes_by_condition[condition] = neighbor_lifetimes_all_tracks

    return neighbor_lifetimes_by_condition


# ---------------------------------------------------------------------------
# Neighbor retention / survival curve
# ---------------------------------------------------------------------------

def compute_neighbor_retention_curve(graphs_dict, key_file, observation_time):
    """Compute the neighbor retention (survival) curve Sₙ(τ) per condition.

    For each time lag τ, computes the fraction of original neighbours still
    present:  Sₙ(τ) = ⟨|Nᵢ(t₀) ∩ Nᵢ(t₀+τ)| / |Nᵢ(t₀)|⟩

    Returns
    -------
    survival_curves : dict
        ``{condition: {'tau', 'Sn', 'Sn_std', 'N_measurements'}}``
    """
    start_frame, end_frame = observation_time
    max_tau = end_frame - start_frame - 1

    survival_curves = {}

    for condition in key_file["condition"].unique():
        print(f"\nComputing neighbor retention curve for condition: {condition}")

        experiments_for_condition = key_file[key_file["condition"] == condition]
        retention_by_tau = {tau: [] for tau in range(1, max_tau + 1)}

        for _, row in experiments_for_condition.iterrows():
            experimentID = row["experimentID"]

            if experimentID not in graphs_dict:
                print(f"  Skipping {experimentID} - no graphs")
                continue

            t_graphs = graphs_dict[experimentID]

            for t0_idx in range(len(t_graphs) - 1):
                G_t0 = t_graphs[t0_idx]

                for node_id in G_t0.nodes():
                    neighbors_t0 = set(G_t0.neighbors(node_id))
                    if len(neighbors_t0) == 0:
                        continue

                    for tau in range(1, max_tau + 1):
                        t_idx = t0_idx + tau
                        if t_idx >= len(t_graphs):
                            break

                        G_t = t_graphs[t_idx]
                        if node_id not in G_t.nodes():
                            continue

                        neighbors_t = set(G_t.neighbors(node_id))
                        intersection_size = len(neighbors_t0 & neighbors_t)
                        retention_fraction = intersection_size / len(neighbors_t0)
                        retention_by_tau[tau].append(retention_fraction)

        # aggregate
        tau_values, Sn_values, Sn_std_values, N_measurements_values = [], [], [], []
        for tau in sorted(retention_by_tau.keys()):
            if len(retention_by_tau[tau]) > 0:
                tau_values.append(tau)
                Sn_values.append(np.mean(retention_by_tau[tau]))
                Sn_std_values.append(np.std(retention_by_tau[tau]))
                N_measurements_values.append(len(retention_by_tau[tau]))

        survival_curves[condition] = {
            "tau": np.array(tau_values),
            "Sn": np.array(Sn_values),
            "Sn_std": np.array(Sn_std_values),
            "N_measurements": np.array(N_measurements_values),
        }

        print(f"  Computed retention curve with {len(tau_values)} time points")
        if len(N_measurements_values) > 0:
            print(
                f"  Average measurements per time point: "
                f"{np.mean(N_measurements_values):.0f}"
            )

    return survival_curves


# ---------------------------------------------------------------------------
# Exponential decay model (used for retention & alignment fitting)
# ---------------------------------------------------------------------------

def exponential_decay(tau, S0, k):
    """Exponential decay: S(τ) = S0 · exp(−k·τ)."""
    return S0 * np.exp(-k * tau)


# ---------------------------------------------------------------------------
# Cage-relative neighbor displacement
# ---------------------------------------------------------------------------

def compute_relative_neighbor_displacement(graphs_dict, key_file, observation_time):
    """Compute cage-relative displacement ⟨δr(τ)⟩ per condition.

    For each initial neighbor pair (i, j) at time t₀, tracks how the
    relative separation vector changes over time lag τ.

    Returns
    -------
    displacement_curves : dict
        ``{condition: {'tau', 'delta_r_mean', 'delta_r_std',
        'delta_r_sem', 'N_pairs'}}``
    """
    start_frame, end_frame = observation_time
    max_tau = end_frame - start_frame - 1

    displacement_curves = {}

    for condition in key_file["condition"].unique():
        print(f"\nComputing relative neighbor displacement for condition: {condition}")

        experiments_for_condition = key_file[key_file["condition"] == condition]
        delta_r_by_tau = {tau: [] for tau in range(1, max_tau + 1)}

        for _, row in experiments_for_condition.iterrows():
            experimentID = row["experimentID"]

            if experimentID not in graphs_dict:
                print(f"  Skipping {experimentID} – no graphs")
                continue

            t_graphs = graphs_dict[experimentID]

            for t0_idx in range(len(t_graphs) - 1):
                G_t0 = t_graphs[t0_idx]

                for i, j in G_t0.edges():
                    ri_t0 = np.array([G_t0.nodes[i]["x"], G_t0.nodes[i]["y"]])
                    rj_t0 = np.array([G_t0.nodes[j]["x"], G_t0.nodes[j]["y"]])
                    d_ij_t0 = ri_t0 - rj_t0

                    for tau in range(1, max_tau + 1):
                        t_idx = t0_idx + tau
                        if t_idx >= len(t_graphs):
                            break

                        G_t = t_graphs[t_idx]
                        if i not in G_t.nodes() or j not in G_t.nodes():
                            continue

                        ri_t = np.array([G_t.nodes[i]["x"], G_t.nodes[i]["y"]])
                        rj_t = np.array([G_t.nodes[j]["x"], G_t.nodes[j]["y"]])
                        d_ij_t = ri_t - rj_t

                        delta_r = np.linalg.norm(d_ij_t - d_ij_t0)
                        delta_r_by_tau[tau].append(delta_r)

        # aggregate
        tau_values, mean_values, std_values, sem_values, n_pairs_values = (
            [], [], [], [], [],
        )
        for tau in sorted(delta_r_by_tau.keys()):
            vals = delta_r_by_tau[tau]
            if len(vals) > 0:
                tau_values.append(tau)
                mean_values.append(np.mean(vals))
                std_values.append(np.std(vals))
                sem_values.append(np.std(vals) / np.sqrt(len(vals)))
                n_pairs_values.append(len(vals))

        displacement_curves[condition] = {
            "tau": np.array(tau_values),
            "delta_r_mean": np.array(mean_values),
            "delta_r_std": np.array(std_values),
            "delta_r_sem": np.array(sem_values),
            "N_pairs": np.array(n_pairs_values),
        }

        print(
            f"  {len(tau_values)} τ-points, "
            f"avg {np.mean(n_pairs_values):.0f} pairs per τ"
            if len(n_pairs_values)
            else ""
        )

    return displacement_curves


# ---------------------------------------------------------------------------
# Velocity alignment
# ---------------------------------------------------------------------------

def _node_velocity(graphs_list, node_id, t_idx):
    """Return velocity vector of *node_id* at frame index *t_idx*.

    Velocity is the displacement to the next frame: v(t) = r(t+1) − r(t).
    Returns ``None`` if the node is missing at t or t+1.
    """
    if t_idx + 1 >= len(graphs_list):
        return None
    G0 = graphs_list[t_idx]
    G1 = graphs_list[t_idx + 1]
    if node_id not in G0.nodes() or node_id not in G1.nodes():
        return None
    r0 = np.array([G0.nodes[node_id]["x"], G0.nodes[node_id]["y"]])
    r1 = np.array([G1.nodes[node_id]["x"], G1.nodes[node_id]["y"]])
    return r1 - r0


def _unit(v):
    """Return the unit vector.  Returns ``None`` for zero-length vectors."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


def compute_velocity_alignment_curves(
    graphs_dict, key_file, observation_time, use_unnormalized=False
):
    """Compute neighbor velocity alignment C_align(τ) for each condition.

    Parameters
    ----------
    graphs_dict : dict
        ``experimentID -> list[nx.Graph]`` (one per frame).
    key_file : pd.DataFrame
        Must contain ``experimentID`` and ``condition`` columns.
    observation_time : tuple (start_frame, end_frame)
    use_unnormalized : bool
        If ``True``, use raw dot-products (speed matters).
        If ``False`` (default), use unit-vector dot-products (pure alignment).

    Returns
    -------
    alignment_curves : dict
        ``{condition: {'tau', 'C_mean', 'C_std', 'C_sem', 'N_pairs'}}``
    """
    start_frame, end_frame = observation_time
    max_tau = end_frame - start_frame - 2  # need one extra frame for v

    alignment_curves = {}

    for condition in key_file["condition"].unique():
        print(f"\nComputing velocity alignment for condition: {condition}")

        experiments = key_file[key_file["condition"] == condition]
        dot_by_tau = {tau: [] for tau in range(0, max_tau + 1)}

        for _, row in experiments.iterrows():
            experimentID = row["experimentID"]
            if experimentID not in graphs_dict:
                print(f"  Skipping {experimentID} – no graphs")
                continue

            t_graphs = graphs_dict[experimentID]

            # pre-compute velocities
            vel_cache = {}
            for t_idx in range(len(t_graphs) - 1):
                G = t_graphs[t_idx]
                for nid in G.nodes():
                    vel_cache[(nid, t_idx)] = _node_velocity(t_graphs, nid, t_idx)

            for t0_idx in range(len(t_graphs) - 1):
                G_t0 = t_graphs[t0_idx]

                for i, j in G_t0.edges():
                    vi = vel_cache.get((i, t0_idx))
                    if vi is None:
                        continue

                    if not use_unnormalized:
                        vi_use = _unit(vi)
                        if vi_use is None:
                            continue
                    else:
                        vi_use = vi

                    for tau in range(0, max_tau + 1):
                        tj_idx = t0_idx + tau
                        if tj_idx >= len(t_graphs) - 1:
                            break

                        vj = vel_cache.get((j, tj_idx))
                        if vj is None:
                            continue

                        if not use_unnormalized:
                            vj_use = _unit(vj)
                            if vj_use is None:
                                continue
                        else:
                            vj_use = vj

                        dot_by_tau[tau].append(np.dot(vi_use, vj_use))

        # aggregate
        tau_vals, mean_vals, std_vals, sem_vals, n_vals = [], [], [], [], []
        for tau in sorted(dot_by_tau.keys()):
            vals = dot_by_tau[tau]
            if len(vals) > 0:
                tau_vals.append(tau)
                mean_vals.append(np.mean(vals))
                std_vals.append(np.std(vals))
                sem_vals.append(np.std(vals) / np.sqrt(len(vals)))
                n_vals.append(len(vals))

        alignment_curves[condition] = {
            "tau": np.array(tau_vals),
            "C_mean": np.array(mean_vals),
            "C_std": np.array(std_vals),
            "C_sem": np.array(sem_vals),
            "N_pairs": np.array(n_vals),
        }

        print(
            f"  {len(tau_vals)} τ-points, "
            f"avg {np.mean(n_vals):.0f} pairs per τ"
            if n_vals
            else ""
        )

    return alignment_curves


# ---------------------------------------------------------------------------
# Anisotropic pairwise separation & self-MSD
# ---------------------------------------------------------------------------

def compute_anisotropic_separation_and_msd(graphs_dict, key_file, observation_time):
    """Compute flow-decomposed pairwise separation and single-cell MSD.

    Returns
    -------
    pair_sep : dict
        ``{condition: {'tau', 'dx2_mean', …, 'dy2_mean', …, 'N_pairs'}}``
    msd_aniso : dict
        ``{condition: {'tau', 'msd_par_mean', …, 'msd_perp_mean', …, 'N_cells'}}``
    """
    start_frame, end_frame = observation_time
    max_tau = end_frame - start_frame - 1

    pair_sep = {}
    msd_aniso = {}

    for condition in key_file["condition"].unique():
        print(f"\nComputing anisotropic metrics for condition: {condition}")

        experiments = key_file[key_file["condition"] == condition]

        dx2_by_tau = {tau: [] for tau in range(1, max_tau + 1)}
        dy2_by_tau = {tau: [] for tau in range(1, max_tau + 1)}
        msd_par_by_tau = {tau: [] for tau in range(1, max_tau + 1)}
        msd_perp_by_tau = {tau: [] for tau in range(1, max_tau + 1)}

        for _, row in experiments.iterrows():
            experimentID = row["experimentID"]
            if experimentID not in graphs_dict:
                print(f"  Skipping {experimentID} – no graphs")
                continue

            t_graphs = graphs_dict[experimentID]

            # --- Pair separation components ---
            for t0_idx in range(len(t_graphs) - 1):
                G_t0 = t_graphs[t0_idx]

                for i, j in G_t0.edges():
                    xi_t0 = G_t0.nodes[i]["x"]
                    yi_t0 = G_t0.nodes[i]["y"]
                    xj_t0 = G_t0.nodes[j]["x"]
                    yj_t0 = G_t0.nodes[j]["y"]

                    sep_x_t0 = xi_t0 - xj_t0
                    sep_y_t0 = yi_t0 - yj_t0

                    for tau in range(1, max_tau + 1):
                        t_idx = t0_idx + tau
                        if t_idx >= len(t_graphs):
                            break
                        G_t = t_graphs[t_idx]

                        if i not in G_t.nodes() or j not in G_t.nodes():
                            continue

                        xi_t = G_t.nodes[i]["x"]
                        yi_t = G_t.nodes[i]["y"]
                        xj_t = G_t.nodes[j]["x"]
                        yj_t = G_t.nodes[j]["y"]

                        dx = (xi_t - xj_t) - sep_x_t0
                        dy = (yi_t - yj_t) - sep_y_t0

                        dx2_by_tau[tau].append(dx ** 2)
                        dy2_by_tau[tau].append(dy ** 2)

            # --- Self-MSD components ---
            for t0_idx in range(len(t_graphs)):
                G_t0 = t_graphs[t0_idx]

                for nid in G_t0.nodes():
                    x0 = G_t0.nodes[nid]["x"]
                    y0 = G_t0.nodes[nid]["y"]

                    for tau in range(1, max_tau + 1):
                        t_idx = t0_idx + tau
                        if t_idx >= len(t_graphs):
                            break
                        G_t = t_graphs[t_idx]

                        if nid not in G_t.nodes():
                            continue

                        dx_self = G_t.nodes[nid]["x"] - x0
                        dy_self = G_t.nodes[nid]["y"] - y0

                        msd_par_by_tau[tau].append(dx_self ** 2)
                        msd_perp_by_tau[tau].append(dy_self ** 2)

        # --- Aggregate pair separation ---
        tau_vals_p, dx2_m, dx2_s, dx2_se, dy2_m, dy2_s, dy2_se, n_pairs = (
            [], [], [], [], [], [], [], [],
        )
        for tau in sorted(dx2_by_tau.keys()):
            vx = dx2_by_tau[tau]
            vy = dy2_by_tau[tau]
            if len(vx) > 0:
                tau_vals_p.append(tau)
                dx2_m.append(np.mean(vx))
                dx2_s.append(np.std(vx))
                dx2_se.append(np.std(vx) / np.sqrt(len(vx)))
                dy2_m.append(np.mean(vy))
                dy2_s.append(np.std(vy))
                dy2_se.append(np.std(vy) / np.sqrt(len(vy)))
                n_pairs.append(len(vx))

        pair_sep[condition] = {
            "tau": np.array(tau_vals_p),
            "dx2_mean": np.array(dx2_m),
            "dx2_std": np.array(dx2_s),
            "dx2_sem": np.array(dx2_se),
            "dy2_mean": np.array(dy2_m),
            "dy2_std": np.array(dy2_s),
            "dy2_sem": np.array(dy2_se),
            "N_pairs": np.array(n_pairs),
        }

        # --- Aggregate self-MSD ---
        tau_vals_m, mp_m, mp_s, mp_se, mr_m, mr_s, mr_se, n_cells = (
            [], [], [], [], [], [], [], [],
        )
        for tau in sorted(msd_par_by_tau.keys()):
            vp = msd_par_by_tau[tau]
            vr = msd_perp_by_tau[tau]
            if len(vp) > 0:
                tau_vals_m.append(tau)
                mp_m.append(np.mean(vp))
                mp_s.append(np.std(vp))
                mp_se.append(np.std(vp) / np.sqrt(len(vp)))
                mr_m.append(np.mean(vr))
                mr_s.append(np.std(vr))
                mr_se.append(np.std(vr) / np.sqrt(len(vr)))
                n_cells.append(len(vp))

        msd_aniso[condition] = {
            "tau": np.array(tau_vals_m),
            "msd_par_mean": np.array(mp_m),
            "msd_par_std": np.array(mp_s),
            "msd_par_sem": np.array(mp_se),
            "msd_perp_mean": np.array(mr_m),
            "msd_perp_std": np.array(mr_s),
            "msd_perp_sem": np.array(mr_se),
            "N_cells": np.array(n_cells),
        }

        print(
            f"  Pair sep: {len(tau_vals_p)} τ-points, "
            f"avg {np.mean(n_pairs):.0f} pairs/τ"
            if n_pairs
            else ""
        )
        print(
            f"  Self-MSD: {len(tau_vals_m)} τ-points, "
            f"avg {np.mean(n_cells):.0f} cells/τ"
            if n_cells
            else ""
        )

    return pair_sep, msd_aniso


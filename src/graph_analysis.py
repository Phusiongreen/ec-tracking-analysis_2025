"""
Time-resolved graph analysis for endothelial cell tracking.

Functions for building Delaunay neighbour graphs at each frame,
computing neighbor lifetimes / retention curves, cage-relative
displacement, velocity alignment, and anisotropic MSD.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from griottes import generate_delaunay_graph


# ---------------------------------------------------------------------------
# Module-level worker (must be at module scope for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _build_graph_at_frame(args):
    """Build and relabel a Delaunay graph for a single frame.

    Parameters
    ----------
    args : tuple
        ``(t, frame_df, distance_threshold)``

    Returns
    -------
    tuple
        ``(t, G_delaunay)`` — frame index and the relabelled graph.
    """
    t, frame_df, distance_threshold = args
    G_delaunay = generate_delaunay_graph(
        frame_df[frame_df.columns],
        descriptors=frame_df.columns,
        distance=distance_threshold,
        image_is_2D=True,
    )
    mapping = {n: G_delaunay.nodes[n]["TRACK_ID"] for n in G_delaunay.nodes}
    G_delaunay = nx.relabel_nodes(G_delaunay, mapping)
    return t, G_delaunay


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_time_graphs(key_file, data_folder, observation_time, distance_threshold, n_jobs=None):
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
    n_jobs : int or None
        Number of worker processes for parallel graph construction.
        ``None`` uses ``os.cpu_count()``.

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



        # Build one task tuple per frame
        frames = range(observation_time[0], observation_time[1])
        tasks = [
            (
                t,
                obs_df_renamed[obs_df_renamed["FRAME"] == t].reset_index(drop=True),
                distance_threshold,
            )
            for t in frames
        ]

        print(
            f"  Building {len(tasks)} frame graphs in parallel "
            f"(n_jobs={n_jobs or os.cpu_count()}) …"
        )
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_build_graph_at_frame, tasks))

        # results come back in submission order (map preserves order)
        t_graphs = [G for _, G in results]
        print(f"  All {len(t_graphs)} graphs built.")

        graphs[experimentID] = t_graphs

    return graphs, observation_period_dfs


# ---------------------------------------------------------------------------
# Neighbor lifetimes
# ---------------------------------------------------------------------------

def compute_neighbor_lifetimes(graphs, observation_period_dfs, key_file, n_jobs=None):
    """Compute consecutive-neighbor lifetimes for every condition.

    Parameters
    ----------
    graphs : dict
        ``{experimentID: [G_t0, G_t1, …]}``.
    observation_period_dfs : dict
        ``{experimentID: DataFrame}`` with at least a ``TRACK_ID`` column.
    key_file : pd.DataFrame
        Must contain ``condition`` and ``experimentID``.
    n_jobs : int or None
        Worker processes. ``None`` uses all CPU cores.

    Returns
    -------
    neighbor_lifetimes_by_condition : dict
        ``{condition: {track_id: {neighbor_id: lifetime_frames}}}``
    """
    # Build one task per unique experimentID that has data
    tasks, seen = [], set()
    for _, row in key_file.iterrows():
        eid = row["experimentID"]
        if eid in seen:
            continue
        seen.add(eid)
        if eid not in graphs or eid not in observation_period_dfs:
            print(f"  Skipping {eid} – missing graphs or tracking data")
            continue
        track_ids = observation_period_dfs[eid]["TRACK_ID"].unique().tolist()
        tasks.append((eid, graphs[eid], track_ids))

    print(
        f"\nComputing neighbor lifetimes for {len(tasks)} experiments "
        f"(n_jobs={n_jobs or os.cpu_count()}) …"
    )
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        results = list(pool.map(_lifetimes_worker, tasks))

    # map experimentID → condition (first match wins)
    exp_to_condition = (
        key_file.drop_duplicates("experimentID")
        .set_index("experimentID")["condition"]
        .to_dict()
    )
    neighbor_lifetimes_by_condition = {c: {} for c in key_file["condition"].unique()}
    for eid, exp_lifetimes in results:
        cond = exp_to_condition.get(eid)
        if cond is not None:
            neighbor_lifetimes_by_condition[cond].update(exp_lifetimes)

    return neighbor_lifetimes_by_condition


# ---------------------------------------------------------------------------
# Neighbor retention / survival curve
# ---------------------------------------------------------------------------

def compute_neighbor_retention_curve(graphs_dict, key_file, observation_time, n_jobs=None):
    """Compute the neighbor retention (survival) curve Sₙ(τ) per condition.

    For each time lag τ, computes the fraction of original neighbours still
    present:  Sₙ(τ) = ⟨|Nᵢ(t₀) ∩ Nᵢ(t₀+τ)| / |Nᵢ(t₀)|⟩

    Parameters
    ----------
    n_jobs : int or None
        Worker processes. ``None`` uses all CPU cores.

    Returns
    -------
    survival_curves : dict
        ``{condition: {'tau', 'Sn', 'Sn_std', 'N_measurements', 'per_experiment'}}``
    """
    start_frame, end_frame = observation_time
    max_tau = end_frame - start_frame - 1

    tasks, seen = [], set()
    for _, row in key_file.iterrows():
        eid = row["experimentID"]
        if eid in seen or eid not in graphs_dict:
            continue
        seen.add(eid)
        tasks.append((eid, graphs_dict[eid], max_tau))

    print(
        f"\nComputing neighbor retention for {len(tasks)} experiments "
        f"(n_jobs={n_jobs or os.cpu_count()}) …"
    )
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        raw = list(pool.map(_retention_worker, tasks))

    exp_data = {eid: by_tau for eid, by_tau in raw}

    survival_curves = {}
    for condition in key_file["condition"].unique():
        print(f"\nAggregating condition: {condition}")
        retention_by_tau = {tau: [] for tau in range(1, max_tau + 1)}
        per_experiment = {}

        for _, row in key_file[key_file["condition"] == condition].iterrows():
            eid = row["experimentID"]
            if eid not in exp_data:
                print(f"  Skipping {eid} – no graphs")
                continue
            exp_by_tau = exp_data[eid]
            for tau, vals in exp_by_tau.items():
                retention_by_tau[tau].extend(vals)

            exp_tau, exp_mean, exp_n = [], [], []
            for tau in sorted(exp_by_tau):
                if exp_by_tau[tau]:
                    exp_tau.append(tau)
                    exp_mean.append(np.mean(exp_by_tau[tau]))
                    exp_n.append(len(exp_by_tau[tau]))
            per_experiment[eid] = {
                "tau": np.array(exp_tau),
                "Sn": np.array(exp_mean),
                "N_measurements": np.array(exp_n),
            }

        tau_values, Sn_values, Sn_std_values, N_values = [], [], [], []
        for tau in sorted(retention_by_tau):
            if retention_by_tau[tau]:
                tau_values.append(tau)
                Sn_values.append(np.mean(retention_by_tau[tau]))
                Sn_std_values.append(np.std(retention_by_tau[tau]))
                N_values.append(len(retention_by_tau[tau]))

        survival_curves[condition] = {
            "tau": np.array(tau_values),
            "Sn": np.array(Sn_values),
            "Sn_std": np.array(Sn_std_values),
            "N_measurements": np.array(N_values),
            "per_experiment": per_experiment,
        }

        print(f"  Computed retention curve with {len(tau_values)} time points")
        if N_values:
            print(
                f"  Average measurements per time point: "
                f"{np.mean(N_values):.0f}"
            )

    return survival_curves


# ---------------------------------------------------------------------------
# Exponential decay model (used for retention & alignment fitting)
# ---------------------------------------------------------------------------

def exponential_decay(tau, S0, k):
    """Exponential decay: S(τ) = S0 · exp(−k·τ)."""
    return S0 * np.exp(-k * tau)


# ---------------------------------------------------------------------------
# Per-experiment parallel workers  (module-scope required for pickling)
# ---------------------------------------------------------------------------

def _lifetimes_worker(args):
    """Compute neighbor lifetimes for all tracks in one experiment.

    Parameters: ``(experimentID, t_graphs, trackIDs)``
    Returns: ``(experimentID, {track_id: {neighbor_id: lifetime_frames}})``
    """
    experimentID, t_graphs, trackIDs = args
    result = {}
    if not t_graphs:
        return experimentID, result
    G0 = t_graphs[0]
    for track_id in trackIDs:
        if track_id not in G0:
            continue
        neighbor_lifetimes = {nb: 1 for nb in G0.neighbors(track_id)}
        for t, G in enumerate(t_graphs[1:], start=1):
            if track_id not in G:
                break
            for nb in G[track_id]:
                if neighbor_lifetimes.get(nb) == t:
                    neighbor_lifetimes[nb] += 1
        result[track_id] = neighbor_lifetimes
    return experimentID, result


def _retention_worker(args):
    """Retention fractions for all (node, t0, τ) triples in one experiment.

    Parameters: ``(experimentID, t_graphs, max_tau)``
    Returns: ``(experimentID, {tau: [fractions]})``
    """
    experimentID, t_graphs, max_tau = args
    exp_by_tau = {tau: [] for tau in range(1, max_tau + 1)}
    for t0_idx in range(len(t_graphs) - 1):
        G_t0 = t_graphs[t0_idx]
        for node_id in G_t0.nodes():
            nb0 = set(G_t0.neighbors(node_id))
            if not nb0:
                continue
            for tau in range(1, max_tau + 1):
                t_idx = t0_idx + tau
                if t_idx >= len(t_graphs):
                    break
                G_t = t_graphs[t_idx]
                if node_id not in G_t:
                    continue
                nb_t = set(G_t.neighbors(node_id))
                exp_by_tau[tau].append(len(nb0 & nb_t) / len(nb0))
    return experimentID, exp_by_tau


def _displacement_worker(args):
    """Cage-relative displacement for all edge-pairs in one experiment.

    Parameters: ``(experimentID, t_graphs, max_tau)``
    Returns: ``(experimentID, {tau: [delta_r values]})``
    """
    experimentID, t_graphs, max_tau = args
    exp_by_tau = {tau: [] for tau in range(1, max_tau + 1)}
    for t0_idx in range(len(t_graphs) - 1):
        G_t0 = t_graphs[t0_idx]
        for i, j in G_t0.edges():
            d_t0 = np.array([
                G_t0.nodes[i]["x"] - G_t0.nodes[j]["x"],
                G_t0.nodes[i]["y"] - G_t0.nodes[j]["y"],
            ])
            for tau in range(1, max_tau + 1):
                t_idx = t0_idx + tau
                if t_idx >= len(t_graphs):
                    break
                G_t = t_graphs[t_idx]
                if i not in G_t or j not in G_t:
                    continue
                d_t = np.array([
                    G_t.nodes[i]["x"] - G_t.nodes[j]["x"],
                    G_t.nodes[i]["y"] - G_t.nodes[j]["y"],
                ])
                exp_by_tau[tau].append(np.linalg.norm(d_t - d_t0))
    return experimentID, exp_by_tau


def _alignment_worker(args):
    """Velocity alignment dot-products for all edges in one experiment.

    Parameters: ``(experimentID, t_graphs, max_tau, use_unnormalized)``
    Returns: ``(experimentID, {tau: [dot-product values]})``
    """
    experimentID, t_graphs, max_tau, use_unnormalized = args
    exp_by_tau = {tau: [] for tau in range(0, max_tau + 1)}
    # pre-compute velocities
    vel_cache = {}
    for t_idx in range(len(t_graphs) - 1):
        for nid in t_graphs[t_idx].nodes():
            vel_cache[(nid, t_idx)] = _node_velocity(t_graphs, nid, t_idx)
    for t0_idx in range(len(t_graphs) - 1):
        G_t0 = t_graphs[t0_idx]
        for i, j in G_t0.edges():
            vi = vel_cache.get((i, t0_idx))
            if vi is None:
                continue
            vi_use = vi if use_unnormalized else _unit(vi)
            if vi_use is None:
                continue
            for tau in range(0, max_tau + 1):
                tj_idx = t0_idx + tau
                if tj_idx >= len(t_graphs) - 1:
                    break
                vj = vel_cache.get((j, tj_idx))
                if vj is None:
                    continue
                vj_use = vj if use_unnormalized else _unit(vj)
                if vj_use is None:
                    continue
                exp_by_tau[tau].append(np.dot(vi_use, vj_use))
    return experimentID, exp_by_tau


def _anisotropic_worker(args):
    """Flow-decomposed pair separation and self-MSD for one experiment.

    Parameters: ``(experimentID, t_graphs, max_tau)``
    Returns: ``(experimentID, exp_dx2, exp_dy2, exp_mp, exp_mr)``
        Each dict maps ``tau -> [values]``.
    """
    experimentID, t_graphs, max_tau = args
    exp_dx2 = {tau: [] for tau in range(1, max_tau + 1)}
    exp_dy2 = {tau: [] for tau in range(1, max_tau + 1)}
    exp_mp  = {tau: [] for tau in range(1, max_tau + 1)}
    exp_mr  = {tau: [] for tau in range(1, max_tau + 1)}

    # --- Pair separation ---
    for t0_idx in range(len(t_graphs) - 1):
        G_t0 = t_graphs[t0_idx]
        for i, j in G_t0.edges():
            sx0 = G_t0.nodes[i]["x"] - G_t0.nodes[j]["x"]
            sy0 = G_t0.nodes[i]["y"] - G_t0.nodes[j]["y"]
            for tau in range(1, max_tau + 1):
                t_idx = t0_idx + tau
                if t_idx >= len(t_graphs):
                    break
                G_t = t_graphs[t_idx]
                if i not in G_t or j not in G_t:
                    continue
                dx = (G_t.nodes[i]["x"] - G_t.nodes[j]["x"]) - sx0
                dy = (G_t.nodes[i]["y"] - G_t.nodes[j]["y"]) - sy0
                exp_dx2[tau].append(dx ** 2)
                exp_dy2[tau].append(dy ** 2)

    # --- Self-MSD ---
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
                if nid not in G_t:
                    continue
                exp_mp[tau].append((G_t.nodes[nid]["x"] - x0) ** 2)
                exp_mr[tau].append((G_t.nodes[nid]["y"] - y0) ** 2)

    return experimentID, exp_dx2, exp_dy2, exp_mp, exp_mr


# ---------------------------------------------------------------------------
# Cage-relative neighbor displacement
# ---------------------------------------------------------------------------

def compute_relative_neighbor_displacement(graphs_dict, key_file, observation_time, n_jobs=None):
    """Compute cage-relative displacement ⟨δr(τ)⟩ per condition.

    For each initial neighbor pair (i, j) at time t₀, tracks how the
    relative separation vector changes over time lag τ.

    Parameters
    ----------
    n_jobs : int or None
        Worker processes. ``None`` uses all CPU cores.

    Returns
    -------
    displacement_curves : dict
        ``{condition: {'tau', 'delta_r_mean', 'delta_r_std',
        'delta_r_sem', 'N_pairs', 'per_experiment'}}``
    """
    start_frame, end_frame = observation_time
    max_tau = end_frame - start_frame - 1

    tasks, seen = [], set()
    for _, row in key_file.iterrows():
        eid = row["experimentID"]
        if eid in seen or eid not in graphs_dict:
            continue
        seen.add(eid)
        tasks.append((eid, graphs_dict[eid], max_tau))

    print(
        f"\nComputing cage-relative displacement for {len(tasks)} experiments "
        f"(n_jobs={n_jobs or os.cpu_count()}) …"
    )
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        raw = list(pool.map(_displacement_worker, tasks))

    exp_data = {eid: by_tau for eid, by_tau in raw}

    displacement_curves = {}
    for condition in key_file["condition"].unique():
        print(f"\nAggregating condition: {condition}")
        delta_r_by_tau = {tau: [] for tau in range(1, max_tau + 1)}
        per_experiment = {}

        for _, row in key_file[key_file["condition"] == condition].iterrows():
            eid = row["experimentID"]
            if eid not in exp_data:
                print(f"  Skipping {eid} – no graphs")
                continue
            exp_by_tau = exp_data[eid]
            for tau, vals in exp_by_tau.items():
                delta_r_by_tau[tau].extend(vals)

            exp_tau, exp_mean, exp_n = [], [], []
            for tau in sorted(exp_by_tau):
                if exp_by_tau[tau]:
                    exp_tau.append(tau)
                    exp_mean.append(np.mean(exp_by_tau[tau]))
                    exp_n.append(len(exp_by_tau[tau]))
            per_experiment[eid] = {
                "tau": np.array(exp_tau),
                "delta_r_mean": np.array(exp_mean),
                "N_pairs": np.array(exp_n),
            }

        tau_values, mean_values, std_values, sem_values, n_pairs_values = (
            [], [], [], [], [],
        )
        for tau in sorted(delta_r_by_tau):
            vals = delta_r_by_tau[tau]
            if vals:
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
            "per_experiment": per_experiment,
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
    graphs_dict, key_file, observation_time, use_unnormalized=False, n_jobs=None
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
    n_jobs : int or None
        Worker processes. ``None`` uses all CPU cores.

    Returns
    -------
    alignment_curves : dict
        ``{condition: {'tau', 'C_mean', 'C_std', 'C_sem', 'N_pairs', 'per_experiment'}}``
    """
    start_frame, end_frame = observation_time
    max_tau = end_frame - start_frame - 2  # need one extra frame for v

    tasks, seen = [], set()
    for _, row in key_file.iterrows():
        eid = row["experimentID"]
        if eid in seen or eid not in graphs_dict:
            continue
        seen.add(eid)
        tasks.append((eid, graphs_dict[eid], max_tau, use_unnormalized))

    print(
        f"\nComputing velocity alignment for {len(tasks)} experiments "
        f"(n_jobs={n_jobs or os.cpu_count()}) …"
    )
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        raw = list(pool.map(_alignment_worker, tasks))

    exp_data = {eid: by_tau for eid, by_tau in raw}

    alignment_curves = {}
    for condition in key_file["condition"].unique():
        print(f"\nAggregating condition: {condition}")
        dot_by_tau = {tau: [] for tau in range(0, max_tau + 1)}
        per_experiment = {}

        for _, row in key_file[key_file["condition"] == condition].iterrows():
            eid = row["experimentID"]
            if eid not in exp_data:
                print(f"  Skipping {eid} – no graphs")
                continue
            exp_by_tau = exp_data[eid]
            for tau, vals in exp_by_tau.items():
                dot_by_tau[tau].extend(vals)

            exp_tau, exp_mean, exp_n = [], [], []
            for tau in sorted(exp_by_tau):
                if exp_by_tau[tau]:
                    exp_tau.append(tau)
                    exp_mean.append(np.mean(exp_by_tau[tau]))
                    exp_n.append(len(exp_by_tau[tau]))
            per_experiment[eid] = {
                "tau": np.array(exp_tau),
                "C_mean": np.array(exp_mean),
                "N_pairs": np.array(exp_n),
            }

        tau_vals, mean_vals, std_vals, sem_vals, n_vals = [], [], [], [], []
        for tau in sorted(dot_by_tau):
            vals = dot_by_tau[tau]
            if vals:
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
            "per_experiment": per_experiment,
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

def compute_anisotropic_separation_and_msd(graphs_dict, key_file, observation_time, n_jobs=None):
    """Compute flow-decomposed pairwise separation and single-cell MSD.

    Parameters
    ----------
    n_jobs : int or None
        Worker processes. ``None`` uses all CPU cores.

    Returns
    -------
    pair_sep : dict
        ``{condition: {'tau', 'dx2_mean', …, 'dy2_mean', …, 'N_pairs', 'per_experiment'}}``
    msd_aniso : dict
        ``{condition: {'tau', 'msd_par_mean', …, 'msd_perp_mean', …, 'N_cells', 'per_experiment'}}``
    """
    start_frame, end_frame = observation_time
    max_tau = end_frame - start_frame - 1

    tasks, seen = [], set()
    for _, row in key_file.iterrows():
        eid = row["experimentID"]
        if eid in seen or eid not in graphs_dict:
            continue
        seen.add(eid)
        tasks.append((eid, graphs_dict[eid], max_tau))

    print(
        f"\nComputing anisotropic metrics for {len(tasks)} experiments "
        f"(n_jobs={n_jobs or os.cpu_count()}) …"
    )
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        raw = list(pool.map(_anisotropic_worker, tasks))

    exp_data = {eid: (dx2, dy2, mp, mr) for eid, dx2, dy2, mp, mr in raw}

    pair_sep = {}
    msd_aniso = {}

    for condition in key_file["condition"].unique():
        print(f"\nAggregating condition: {condition}")

        dx2_by_tau  = {tau: [] for tau in range(1, max_tau + 1)}
        dy2_by_tau  = {tau: [] for tau in range(1, max_tau + 1)}
        msd_par_by_tau  = {tau: [] for tau in range(1, max_tau + 1)}
        msd_perp_by_tau = {tau: [] for tau in range(1, max_tau + 1)}
        per_experiment_pair = {}
        per_experiment_msd  = {}

        for _, row in key_file[key_file["condition"] == condition].iterrows():
            eid = row["experimentID"]
            if eid not in exp_data:
                print(f"  Skipping {eid} – no graphs")
                continue
            exp_dx2, exp_dy2, exp_mp, exp_mr = exp_data[eid]

            for tau in exp_dx2:
                dx2_by_tau[tau].extend(exp_dx2[tau])
                dy2_by_tau[tau].extend(exp_dy2[tau])
                msd_par_by_tau[tau].extend(exp_mp[tau])
                msd_perp_by_tau[tau].extend(exp_mr[tau])

            p_tau, p_dx2, p_dy2, p_n = [], [], [], []
            for tau in sorted(exp_dx2):
                if exp_dx2[tau]:
                    p_tau.append(tau)
                    p_dx2.append(np.mean(exp_dx2[tau]))
                    p_dy2.append(np.mean(exp_dy2[tau]))
                    p_n.append(len(exp_dx2[tau]))
            per_experiment_pair[eid] = {
                "tau": np.array(p_tau),
                "dx2_mean": np.array(p_dx2),
                "dy2_mean": np.array(p_dy2),
                "N_pairs": np.array(p_n),
            }

            m_tau, m_mp, m_mr, m_n = [], [], [], []
            for tau in sorted(exp_mp):
                if exp_mp[tau]:
                    m_tau.append(tau)
                    m_mp.append(np.mean(exp_mp[tau]))
                    m_mr.append(np.mean(exp_mr[tau]))
                    m_n.append(len(exp_mp[tau]))
            per_experiment_msd[eid] = {
                "tau": np.array(m_tau),
                "msd_par_mean": np.array(m_mp),
                "msd_perp_mean": np.array(m_mr),
                "N_cells": np.array(m_n),
            }

        # --- Aggregate pair separation ---
        tau_vals_p, dx2_m, dx2_s, dx2_se, dy2_m, dy2_s, dy2_se, n_pairs = (
            [], [], [], [], [], [], [], [],
        )
        for tau in sorted(dx2_by_tau):
            vx = dx2_by_tau[tau]
            vy = dy2_by_tau[tau]
            if vx:
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
            "per_experiment": per_experiment_pair,
        }

        # --- Aggregate self-MSD ---
        tau_vals_m, mp_m, mp_s, mp_se, mr_m, mr_s, mr_se, n_cells = (
            [], [], [], [], [], [], [], [],
        )
        for tau in sorted(msd_par_by_tau):
            vp = msd_par_by_tau[tau]
            vr = msd_perp_by_tau[tau]
            if vp:
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
            "per_experiment": per_experiment_msd,
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

# ---------------------------------------------------------------------------
# Cross-experiment (replicate-level) aggregation for honest error bars
# ---------------------------------------------------------------------------

def cross_experiment_stats(per_experiment_dict, value_key):
    """Aggregate per-experiment curves across replicates into mean ± SEM.

    Within one experiment, samples used to compute the lag-τ mean are highly
    correlated (overlapping time windows along the same trajectories), so the
    within-experiment ``std / sqrt(N)`` badly under-estimates the uncertainty
    on the curve.  Different experiments (biological replicates) are, however,
    statistically independent realisations.  This helper computes mean and
    SEM across replicates on the common τ grid, which is the frequentist-
    correct error bar for between-condition comparisons.

    Parameters
    ----------
    per_experiment_dict : dict
        ``{experimentID: {"tau": array, value_key: array, ...}}``
        as returned under the ``"per_experiment"`` field of the condition-
        level dictionaries from the other functions in this module.
    value_key : str
        Key of the per-experiment curve to aggregate (e.g. ``"Sn"``,
        ``"delta_r_mean"``, ``"C_mean"``, ``"dx2_mean"``, ``"msd_par_mean"``).

    Returns
    -------
    dict with keys ``tau``, ``mean``, ``sem``, ``std``, ``n_experiments``.
        Each τ entry uses only experiments that actually have a value at
        that τ (so ``n_experiments`` can vary with τ if tracking lengths
        differ).
    """
    # collect tau union
    all_taus = set()
    for exp_data in per_experiment_dict.values():
        all_taus.update(exp_data["tau"].tolist())
    taus_sorted = np.array(sorted(all_taus))

    means, sems, stds, ns = [], [], [], []
    out_taus = []
    for tau in taus_sorted:
        vals = []
        for exp_data in per_experiment_dict.values():
            idx = np.where(exp_data["tau"] == tau)[0]
            if len(idx) == 1:
                vals.append(exp_data[value_key][idx[0]])
        if len(vals) >= 1:
            out_taus.append(tau)
            vals = np.asarray(vals, dtype=float)
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            sems.append(
                np.std(vals, ddof=1) / np.sqrt(len(vals))
                if len(vals) > 1 else 0.0
            )
            ns.append(len(vals))

    return {
        "tau": np.array(out_taus),
        "mean": np.array(means),
        "sem": np.array(sems),
        "std": np.array(stds),
        "n_experiments": np.array(ns),
    }


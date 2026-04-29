from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import sys
sys.path.append("../ec-tracking-analysis_2025/")
from src.computation import gaps_for_track
from src.io import create_path_recursively


# ---------------------------------------------------------------------------
# Axis-limit helpers — keep trajectory / velocity / migration plots from
# being silently clipped by hard-coded literal limits.  All helpers accept
# either a numeric value (used directly) or the string "auto" (data-driven,
# rounded up to the next *step* units so the border is never tighter than
# the longest track / highest speed).
# ---------------------------------------------------------------------------

def is_auto(v) -> bool:
    """Return True if v is the string 'auto' (case-insensitive)."""
    return isinstance(v, str) and v.strip().lower() == "auto"


def round_up_step(x: float, step: float = 100.0) -> float:
    """Round *x* up to the next multiple of *step*, minimum = *step*."""
    return max(step, float(np.ceil(x / step) * step))


def resolve_abs_max(cfg, *arrays, default: float = 2000.0, step: float = 100.0) -> float:
    """Resolve an absolute upper limit from a YAML-style *cfg* value.

    cfg : number | "auto"
        Number → used directly.  "auto" → fit data extent.
    *arrays : 1-D array-likes
        Data arrays to inspect when cfg == "auto".  NaNs are ignored.
    """
    if is_auto(cfg):
        arrs = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
        arrs = [a for a in arrs if a.size]
        if not arrs:
            return float(default)
        m = float(np.nanmax(np.concatenate(arrs)))
        return round_up_step(m, step)
    return float(cfg)


def resolve_halfrange(cfg, *arrays, default: float = 1000.0, step: float = 100.0) -> float:
    """Resolve a symmetric half-range ``[-v, +v]`` from a YAML-style *cfg*.

    cfg : number | "auto"
        Number → used directly.  "auto" → use max(|data|).
    """
    if is_auto(cfg):
        arrs = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
        arrs = [a for a in arrs if a.size]
        if not arrs:
            return float(default)
        m = float(np.nanmax(np.abs(np.concatenate(arrs))))
        return round_up_step(m, step)
    return float(cfg)


def resolve_interval(cfg, *arrays, default=(0.0, 1.0), step: float = 1.0):
    """Resolve a (low, high) interval from a YAML-style *cfg*.

    cfg : (low, high) | "auto"
        Two-tuple/list → used directly.  "auto" → ``(min(data), max(data))``,
        each bound rounded out to the next multiple of *step*.
    """
    if is_auto(cfg):
        arrs = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
        arrs = [a for a in arrs if a.size]
        if not arrs:
            return (float(default[0]), float(default[1]))
        allv = np.concatenate(arrs)
        lo = float(np.nanmin(allv))
        hi = float(np.nanmax(allv))
        # round outward to the next *step*
        lo = float(np.floor(lo / step) * step)
        hi = float(np.ceil(hi / step) * step)
        if hi <= lo:
            hi = lo + step
        return (lo, hi)
    lo, hi = cfg
    return (float(lo), float(hi))


sns.set_theme(
    context="paper",              # 'paper' or 'talk'
    style="whitegrid",
    palette="deep",
    font_scale=1.2
)
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "axes.titleweight": "semibold",
    "axes.labelweight": "regular",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.frameon": False,
})


def _save(fig, outdir: Path, fname: str):
    fig.savefig(outdir.joinpath(fname), bbox_inches="tight")
    plt.close(fig)

thousands = FuncFormatter(lambda x, pos: f"{int(x):,}")
def _polish(ax, xlabel=None, ylabel=None, title=None, subtitle=None, y_thousands=False):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title if not subtitle else f"{title}\n{subtitle}", loc="left", pad=10)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, left=False, bottom=False)
    if y_thousands:
        ax.yaxis.set_major_formatter(thousands)
    ax.figure.tight_layout()


def plot_quality_control(parameters, key_file, tracking_data_path):
    out_base = Path(parameters["output_folder"]).joinpath("quality_control")
    create_path_recursively(out_base)


    for _, row in key_file.iterrows():
        tracking_file = f"tracking_data_{row['treatment']}_{row['color']}_{row['experimentID']}.csv"
        print("Plot quality control for file", row["filename"])

        try:
            data = pd.read_csv(tracking_data_path / tracking_file, low_memory=False)
        except FileNotFoundError:
            print(f"  ✗ Missing file: {tracking_file}")
            continue

        # Basic sanity
        needed = {"FRAME", "TRACK_ID"}
        if not needed.issubset(data.columns):
            print(f"  ✗ Missing columns in {tracking_file}: {needed - set(data.columns)}")
            continue

        # -------- 1) Scatter: Frames vs Tracks (sample if huge) --------
        df_scatter = data[["FRAME", "TRACK_ID"]].copy()
        n = len(df_scatter)
        if n > 400_000:
            df_scatter = df_scatter.sample(400_000, random_state=0)  # visual density control

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(
            df_scatter["FRAME"], df_scatter["TRACK_ID"],
            s=4, alpha=0.2, rasterized=True
        )
        _polish(
            ax,
            xlabel="Frame",
            ylabel="Track ID",
            title=f"Experiment {row['experimentID']} — Frame vs. Track ID",
            subtitle=f"{row['treatment']} / {row['color']}  •  points: {len(df_scatter):,}"
        )
        plt.show()
        _save(fig, out_base, f"qc1_scatter_{row['treatment']}_{row['color']}_{row['experimentID']}.png")

        # -------- 2) Line: #tracks with data per frame (+ rolling avg) --------
        tracks_per_frame = (data[["FRAME", "TRACK_ID"]]
                            .groupby("FRAME", as_index=False)
                            .agg(n_tracks=("TRACK_ID", "count"))
                            .sort_values("FRAME"))
        tracks_per_frame["rolling"] = tracks_per_frame["n_tracks"].rolling(5, center=True, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(tracks_per_frame["FRAME"], tracks_per_frame["n_tracks"], linewidth=1, alpha=0.7, label="# tracks")
        ax.plot(tracks_per_frame["FRAME"], tracks_per_frame["rolling"], linewidth=2, label="rolling(5)")
        ax.legend(loc="upper right")
        _polish(
            ax,
            xlabel="Frame",
            ylabel="Tracks present",
            title="Tracks present over time",
            subtitle=tracking_file,
            y_thousands=True
        )
        plt.show()
        _save(fig, out_base, f"qc2_tracks_over_time_{row['treatment']}_{row['color']}_{row['experimentID']}.png")

        # -------- 3) Histogram: Track lengths (frames) with stats lines --------
        track_length = (data.groupby("TRACK_ID", as_index=False)["FRAME"]
                        .count()
                        .rename(columns={"FRAME": "length_frames"}))

        L = track_length["length_frames"]
        q50, q90 = int(L.median()), int(L.quantile(0.9))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(L, bins=40, alpha=0.9)
        ax.axvline(q50, linestyle="--")
        ax.axvline(q90, linestyle=":")
        ax.text(q50, ax.get_ylim()[1] * 0.95, f" median={q50}", va="top")
        ax.text(q90, ax.get_ylim()[1] * 0.90, f" p90={q90}", va="top")
        _polish(
            ax,
            xlabel="Track length (frames)",
            ylabel="Number of tracks",
            title="Distribution of track lengths",
            subtitle=f"{len(track_length):,} tracks"
        )
        plt.show()
        _save(fig, out_base, f"qc3_track_length_{row['treatment']}_{row['color']}_{row['experimentID']}.png")

        # -------- 4) Gaps: histogram of gap lengths --------
        gaps = (data.groupby("TRACK_ID")["FRAME"]
                .apply(gaps_for_track)
                .reset_index(level=0, names=["TRACK_ID"])
                .reset_index(drop=True))
        if not gaps.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(gaps["gap_length"], bins=np.arange(1, gaps["gap_length"].max() + 2) - 0.5)
            _polish(
                ax,
                xlabel="Gap length (frames)",
                ylabel="Number of gaps",
                title="Gap length distribution",
                subtitle=f"total gaps: {len(gaps):,}"
            )
            _save(fig, out_base, f"qc4_gap_length_{row['treatment']}_{row['color']}_{row['experimentID']}.png")

            # -------- 5) Gaps over time: scatter (alpha+rasterized) --------
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(gaps["gap_start"], gaps["gap_length"], s=8, alpha=0.3, rasterized=True)
            _polish(
                ax,
                xlabel="Gap start (frame)",
                ylabel="Gap length (frames)",
                title="Gap positions over time",
                subtitle=tracking_file
            )
            plt.show()
            _save(fig, out_base, f"qc5_gap_time_{row['treatment']}_{row['color']}_{row['experimentID']}.png")
        else:
            print("  ✓ No gaps detected.")
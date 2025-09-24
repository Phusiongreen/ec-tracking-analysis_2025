from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import sys
sys.path.append("../")
from src.computation import gaps_for_track
from src.io import create_path_recursively


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
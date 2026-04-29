#!/usr/bin/env python
# coding: utf-8

# In[2]:

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("../")
from src.io import read_parameters


# In[5]:

parameter_file = "/home/jpa/PycharmProjects/ec-tracking-analysis_2025/data/collectivity/parameters_collectivity.yml"
parameters = read_parameters(parameter_file)

# This is the folder that contains only CSV files (no subfolders).
raw_folder = Path(parameters["raw"])

# This is where the script will save the time‐corrected CSV files.
output_folder = Path(parameters["output_folder"])

if not os.path.exists(output_folder.joinpath("time_correction")):
    os.makedirs(output_folder.joinpath("time_correction"), exist_ok=True)


# In[7]:

# --- Data types for columns (optional, keeps memory usage lower) ---
# NOTE: POSITION_X/Y must be at least float32.  float16 snaps coordinates
# onto the half-precision grid (1-µm steps for X∈[1024,2048], 2-µm steps for
# X∈[2048,4096]); this silently pushes cells onto discrete lanes near the
# edges of the field of view and breaks every downstream Delaunay / velocity /
# MSD analysis.
column_dtypes = {
    "TRACK_ID": "int32",
    "FRAME": "int16",
    "POSITION_X": "float32",
    "POSITION_Y": "float32",
    "POSITION_T": "float32"
}

# --- Edge-node exclusion ---
# TrackMate's spot detector has a known bias: spots are over-detected in the
# outermost pixel rows/columns of the field of view (intensity normalization
# and sub-pixel centroid artefacts).  These edge cells pile up onto a thin
# line at the image border and pollute the Delaunay graph.  Rows whose
# POSITION_X / POSITION_Y lie within `edge_exclusion_um` micrometres of the
# per-file min or max are dropped.  Set to 0 (default) to disable filtering.
edge_exclusion_um = float(parameters.get("edge_exclusion_um", 0.0))
print(f"edge_exclusion_um = {edge_exclusion_um}")

# --- Gather all CSVs in the resources folder (no subfolders) ---
csv_files = glob.glob(str(raw_folder / "*.csv"))

print(f"Found {len(csv_files)} CSV files in {raw_folder}.")

# --- Process each CSV ---
for filename in csv_files:
    print(f"Processing: {filename}")

    # 1) Read the CSV into a DataFrame
    #    If you have already removed any extra text/header rows manually, you do not need to drop them again and should remove this part
    data = pd.read_csv(filename, low_memory=False)
    header_rows = parameters["trackmate_header_rows"]
    if header_rows > 0:
        data = pd.read_csv(filename, low_memory=False).drop(list(range(header_rows)))
    # 2) Optionally cast to defined data types
    data = data.astype(column_dtypes)

    # 3) Sort by FRAME so the time sequence is in order
    data = data.sort_values(by="FRAME")

    # 4) Recompute the time column based on seconds per frame (derived from frames_per_hour)
    #    This will override the original 'POSITION_T' from TrackMate
    seconds_per_frame = 3600.0 / parameters["frames_per_hour"]
    data["POSITION_T"] = data["FRAME"] * seconds_per_frame

    # 5) Drop rows within `edge_exclusion_um` of the field-of-view borders,
    #    then — to avoid introducing mid-track gaps when a cell briefly
    #    dips into the edge ring and comes back — keep only the LONGEST
    #    contiguous-FRAME segment per TRACK_ID.  Tracks that lie almost
    #    entirely inside the valid zone are therefore preserved, only the
    #    edge-dip tails/heads are truncated; tracks that live predominantly
    #    in the edge ring effectively vanish (their longest inside-segment
    #    is short or empty and will later be dropped by `min_track_length`
    #    in 01_Preprocess_TackMate_Data.py).
    #    The FOV extent is taken as the per-file min/max of POSITION_X/Y.
    if edge_exclusion_um > 0 and len(data) > 0:
        x_lo, x_hi = float(data["POSITION_X"].min()), float(data["POSITION_X"].max())
        y_lo, y_hi = float(data["POSITION_Y"].min()), float(data["POSITION_Y"].max())
        inside = (
            (data["POSITION_X"] >= x_lo + edge_exclusion_um)
            & (data["POSITION_X"] <= x_hi - edge_exclusion_um)
            & (data["POSITION_Y"] >= y_lo + edge_exclusion_um)
            & (data["POSITION_Y"] <= y_hi - edge_exclusion_um)
        )
        n_before = len(data)
        n_tracks_before = data["TRACK_ID"].nunique()

        data = data.loc[inside].copy()
        data = data.sort_values(["TRACK_ID", "FRAME"])

        def _longest_contiguous_segment(group: pd.DataFrame) -> pd.DataFrame:
            """Return the slice corresponding to the longest consecutive-FRAME run."""
            frames = group["FRAME"].to_numpy()
            if frames.size == 0:
                return group.iloc[0:0]
            # a break occurs wherever consecutive frames are not exactly +1 apart
            breaks = np.where(np.diff(frames) != 1)[0] + 1
            segments = np.split(np.arange(frames.size), breaks)
            longest = max(segments, key=len)
            return group.iloc[longest]

        data = (
            data.groupby("TRACK_ID", group_keys=False, sort=False)
                .apply(_longest_contiguous_segment)
                .reset_index(drop=True)
        )
        n_after = len(data)
        n_tracks_after = data["TRACK_ID"].nunique()
        print(
            f"   FOV extent  x: [{x_lo:.1f}, {x_hi:.1f}]  "
            f"y: [{y_lo:.1f}, {y_hi:.1f}]  (µm)"
        )
        print(
            f"   dropped {n_before - n_after} / {n_before} rows "
            f"({100 * (n_before - n_after) / max(n_before, 1):.1f}%): "
            f"edge rows + track sub-segments (kept longest gap-free run per track)"
        )
        print(
            f"   tracks retained: {n_tracks_after} / {n_tracks_before} "
            f"(fully-edge tracks collapse to 0-length and are dropped here)"
        )

    # 6) Print a small diagnostic
    print("Frames found:", data["FRAME"].unique()[:5], "...")  # show first 5 unique frames
    print("Newly assigned times (POSITION_T):", data["POSITION_T"].unique()[:5], "...")

    # 7) Save the corrected data to the output folder
    #    Use the original file name but place in the "time_corrected" directory
    file_basename = os.path.basename(filename)
    new_basename = "timecorrected_" + file_basename
    output_path = output_folder / "time_correction" / new_basename
    data.to_csv(output_path, index=False)

    print(f"Saved time‐corrected CSV -> {output_path}\n")


# In[ ]:




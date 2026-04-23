#!/usr/bin/env python
# coding: utf-8

# In[2]:

import glob
import os
import sys
from pathlib import Path

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
column_dtypes = {
    "TRACK_ID": "int16",
    "FRAME": "int16",
    "POSITION_X": "float16",
    "POSITION_Y": "float16",
    "POSITION_T": "float32"
}

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

    # 5) Print a small diagnostic
    print("Frames found:", data["FRAME"].unique()[:5], "...")  # show first 5 unique frames
    print("Newly assigned times (POSITION_T):", data["POSITION_T"].unique()[:5], "...")

    # 6) Save the corrected data to the output folder
    #    Use the original file name but place in the "time_corrected" directory
    file_basename = os.path.basename(filename)
    new_basename = "timecorrected_" + file_basename
    output_path = output_folder / "time_correction" / new_basename
    data.to_csv(output_path, index=False)

    print(f"Saved time‐corrected CSV -> {output_path}\n")


# In[ ]:




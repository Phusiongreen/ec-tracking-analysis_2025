import os
from pathlib import Path
from typing import Union

import pandas as pd
import yaml

"""
Global variable to save program call time.
"""


def get_tracking_data_files(parameters):
    """Return list of tracking-data CSV paths from the raw folder.

    By default all *.csv files in the ``raw`` folder are returned.
    If ``tracking_data_include`` is non-empty, only those filenames are used.
    If ``tracking_data_exclude`` is non-empty, those filenames are skipped.
    """
    raw_folder = Path(parameters["raw"])
    include = parameters.get("tracking_data_include", [])
    exclude = parameters.get("tracking_data_exclude", [])

    if include:
        files = [raw_folder / f for f in include]
    else:
        files = sorted(raw_folder.glob("*.csv"))
        if exclude:
            exclude_set = set(exclude)
            files = [f for f in files if f.name not in exclude_set]

    return files


def resolve_observation_time(parameters):
    """Resolve None values in observation_time by scanning the tracking data.

    If either element of observation_time is None, scan all tracking-data CSVs
    in the raw folder to determine the value automatically:
      - None at index 0 → replaced with 0 (global minimum frame)
      - None at index 1 → replaced with min(max(FRAME)) across all files

    Using min-of-max ensures every file has data at every frame in the range.
    """
    obs = parameters.get("observation_time")
    if obs is None:
        return

    if obs[0] is not None and obs[1] is not None:
        return  # nothing to resolve

    files = get_tracking_data_files(parameters)

    max_frames = []
    for fpath in files:
        if fpath.exists():
            df = pd.read_csv(str(fpath), usecols=["FRAME"])
            max_frames.append(int(df["FRAME"].max()))

    if not max_frames:
        print("WARNING: Could not auto-detect observation_time — no tracking data files found.")
        return

    if obs[0] is None:
        obs[0] = 0
    if obs[1] is None:
        obs[1] = min(max_frames)

    parameters["observation_time"] = obs
    print(f"Auto-resolved observation_time: {obs}  (min of max FRAME across {len(max_frames)} files)")


def read_parameters(parameter_file):
    """Reads in default parameters and replaces user defined parameters."""
    current_path = Path(os.path.dirname(os.path.realpath(__file__)))

    param_base_file = Path(current_path).parent.joinpath("resources", "parameters.yml")

    with open(param_base_file, 'r') as yml_f:
        parameters = yaml.safe_load(yml_f)

    with open(parameter_file) as file:
        parameters_local = yaml.safe_load(file)

    # overwrite global parameters with local setting
    for key in parameters_local:
        parameters[key] = parameters_local[key]

    # auto-resolve observation_time if any element is None (null in YAML)
    resolve_observation_time(parameters)

    return parameters

def create_path_recursively(path: Union[str, Path]) -> bool:
    """Create a path. Creates missing parent folders.

    Args:
        path:
            Path to be created.

    Returns:
        True if successful.

    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    return True

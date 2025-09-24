import pandas as pd
import numpy as np
from pathlib import Path


def prepare_tracking_data(parameters, key_file, subfolder="tracking_data"):
    """
    This function reads the tracking data from the Trackmate generated csv files and prepares it for single trajectory analysis, but its generated spreadsheets are
     also used for plotting collective migration plots for some reason.
    """

    # Ensure the required columns are present
    required_columns = {"filename", "condition", "treatment", "color", "experimentID"}
    missing = required_columns - set(key_file.columns)
    if missing:
        raise KeyError(f"Missing columns in key_file: {missing}")

    output_folder = Path(parameters["output_folder"])

    tracking_data_df = pd.DataFrame()

    column_dtypes = {'TRACK_ID': 'int16',
                     'FRAME': 'int16',
                     'POSITION_X': 'float16',
                     'POSITION_Y': 'float16',
                     'POSITION_T': 'float32'}

    gap_analysis = parameters["gap_analysis"]

    print(list(column_dtypes))

    for index, row in key_file.iterrows():
        print("Processing file: ", row["filename"])
        # data = pd.read_csv(base_folder + row["filename"], low_memory=False).drop([0, 1, 2]) #this is now done at 00_correct_time_points_from_trackmate.ipynb
        data = pd.read_csv(str(output_folder.joinpath("time_correction", row["filename"])), low_memory=False)

        # df copy to work on
        data_ = data[list(column_dtypes)]
        data_.insert(0, "filename", row["filename"])
        data_.insert(0, "condition", row["condition"])
        data_ = data_.astype(column_dtypes)
        data_ = data_.sort_values(by="FRAME")

        # loop through the track ID
        for track_id in data_["TRACK_ID"].unique():

            # extract the single track data
            single_track_df = data_[data_["TRACK_ID"] == track_id]

            # length
            track_length = len(single_track_df.index)

            start_frame = single_track_df["FRAME"].min()
            end_frame = single_track_df["FRAME"].max()

            # check if there is a gap in the track
            if gap_analysis:
                if track_length < end_frame - start_frame + 1:
                    print("Track: ", track_id, "with length ", track_length, " has a gap")
                    print(np.array(single_track_df["FRAME"]))

            start_x = np.array(single_track_df["POSITION_X"])[0]
            start_y = np.array(single_track_df["POSITION_Y"])[0]

            # set the start position
            data_.loc[data_.TRACK_ID == track_id, "START_X"] = start_x
            data_.loc[data_.TRACK_ID == track_id, "START_Y"] = start_y

            # threshold for track length
            if track_length < parameters["min_track_length"]:
                data_ = data_[data_["TRACK_ID"] != track_id]

        ### for trajectory plots
        data_["ORIGIN_X"] = data_["POSITION_X"] - data_["START_X"]
        data_["ORIGIN_Y"] = data_["POSITION_Y"] - data_["START_Y"]
        data_["ORIGIN_L"] = np.sqrt(data_["ORIGIN_X"] ** 2 + data_["ORIGIN_Y"] ** 2)

        outpath = Path(output_folder).joinpath(
            subfolder,
            "tracking_data_%s_%s_%s.csv" % (row["treatment"], row["color"], row["experimentID"])
        )
        print("Saving tracking data to: ", str(outpath))
        data_.to_csv(str(outpath), index=False)

        print("##################")
        if len(tracking_data_df.index) > 10:
            # tracking_data_df = tracking_data_df.append(data_)
            tracking_data_df = pd.concat([tracking_data_df, data_], ignore_index=True)
        else:
            tracking_data_df = data_.copy()

        del data_
        del data

    return tracking_data_df


def build_velocity_dataset(
        parameters: dict,
        key_file: pd.DataFrame,
        data_folder: Path,
        observation_time: tuple,  # (start_frame, end_frame)
        obs_time_length_frames: int,
) -> pd.DataFrame:
    velocity_df = pd.DataFrame(
        columns=["EXPERIMENT_ID", "CONDITION", "label", "TRACK_ID", "VEL", "VEL_X", "VEL_Y", "DELTA_X", "DELTA_Y",
                 "START_X", "START_Y", "END_X", "END_Y"])
    vel_index = 0

    for condition in key_file["condition"].unique():
        key_select = key_file[key_file["condition"] == condition]
        for experimentID in key_select["experimentID"].unique():

            key_exp = key_select[key_select["experimentID"] == experimentID]

            for index, row in key_exp.iterrows():

                treatment = row["treatment"]

                tracking_file = "tracking_data_%s_%s_%s.csv" % (treatment, row["color"], row["experimentID"])

                data = pd.read_csv(str(data_folder.joinpath(tracking_file)), low_memory=False)

                print(data["POSITION_X"].min(), data["POSITION_X"].max())

                observation_period_df = data[data["FRAME"] <= observation_time[1]]
                observation_period_df = observation_period_df[observation_period_df["FRAME"] >= observation_time[0]]

                # Get the first frame of each track
                tracks_start = observation_period_df[observation_period_df["FRAME"] == observation_time[0]]

                # determine the min end point of the tracks
                # in case "allow_tracks_shorter_than_observation_time" in the parameters file is set to True
                # -> the min required track length is the min of 'obs_time_length_frames' and 'min_track_length'
                # in case "allow_tracks_shorter_than_observation_time" in the parameters file is set to False
                # -> the end point of each trajectory is the end of the observation time

                frame_end_point = observation_time[1]
                if parameters["allow_tracks_shorter_than_observation_time"] == True:
                    if obs_time_length_frames > parameters["min_track_length"]:
                        frame_end_point = observation_time[0] + parameters["min_track_length"]

                # Assign tracks_end using the determined frame
                tracks_end = observation_period_df[observation_period_df["FRAME"] == frame_end_point]

                track_ids_start = np.array(tracks_start["TRACK_ID"].unique())
                track_ids_end = np.array(tracks_end["TRACK_ID"].unique())

                unique_common_track_ids = np.intersect1d(track_ids_start, track_ids_end)

                trackID_list = np.unique(unique_common_track_ids)

                num_tracks = len(trackID_list)

                print("Available tracks: %s" % num_tracks)

                for trackID in trackID_list:
                    single_track_df = observation_period_df[observation_period_df["TRACK_ID"] == trackID]
                    start_frame = single_track_df["FRAME"].min()
                    end_frame = single_track_df["FRAME"].max()
                    delta_frame = end_frame - start_frame

                    delta_hour = delta_frame / parameters["frames_per_hour"]

                    row_start = single_track_df[single_track_df["FRAME"] == start_frame]
                    row_end = single_track_df[single_track_df["FRAME"] == end_frame]

                    start_x = np.array(row_start["POSITION_X"])[0]
                    start_y = np.array(row_start["POSITION_Y"])[0]
                    end_x = np.array(row_end["POSITION_X"])[0]
                    end_y = np.array(row_end["POSITION_Y"])[0]
                    delta_x = end_x - start_x
                    delta_y = end_y - start_y
                    delta = np.sqrt(delta_x ** 2 + delta_y ** 2)

                    velocity_df.at[vel_index, "EXPERIMENT_ID"] = experimentID
                    velocity_df.at[vel_index, "CONDITION"] = condition
                    velocity_df.at[vel_index, "treatment"] = row["treatment"]
                    velocity_df.at[vel_index, "filename"] = row["filename"]
                    velocity_df.at[vel_index, "TRACK_ID"] = trackID
                    velocity_df.at[vel_index, "label"] = str(trackID) + "_" + row["treatment"]
                    velocity_df.at[vel_index, "color"] = row["color"]
                    velocity_df.at[vel_index, "VEL"] = delta / delta_hour
                    velocity_df.at[vel_index, "VEL_X"] = delta_x / delta_hour
                    velocity_df.at[vel_index, "VEL_Y"] = delta_y / delta_hour
                    velocity_df.at[vel_index, "DELTA_X"] = delta_x
                    velocity_df.at[vel_index, "DELTA_Y"] = delta_y
                    velocity_df.at[vel_index, "START_X"] = start_x
                    velocity_df.at[vel_index, "START_Y"] = start_y
                    velocity_df.at[vel_index, "END_X"] = end_x
                    velocity_df.at[vel_index, "END_Y"] = end_y
                    velocity_df.at[vel_index, "ORIENTATION_DEG"] = np.arctan2(delta_y, delta_x) * 180 / np.pi + 180.0
                    velocity_df.at[vel_index, "ORIENTATION_RAD"] = np.arctan2(delta_y, delta_x) + np.pi

                    vel_index += 1

        velocity_condition_df = velocity_df[velocity_df["CONDITION"] == condition]
        velocity_condition_df.to_csv(str(output_folder.joinpath(subfolder, "velocity_field_%s.csv" % condition)),
                                     index=False)

        velocity_df.to_csv(str(output_folder.joinpath(subfolder, "velocity_field.csv")), index=False)

def _filter_tracks(tracking_data, parameters):
    observation_time = parameters["observation_time"]

    obs_time_length_frames = observation_time[1] - observation_time[0]

    # filter data for observation time
    observation_period_df = tracking_data[tracking_data["FRAME"] <= observation_time[1]]
    observation_period_df = observation_period_df[observation_period_df["FRAME"] >= observation_time[0]]

    # Get the first frame of each track
    tracks_start = observation_period_df[observation_period_df["FRAME"] == observation_time[0]]

    # set the end point of the tracks - must be inside the observation time
    frame_end_point = observation_time[1]
    if parameters["allow_tracks_shorter_than_observation_time"] == True:
        # set minimum required track length
        if obs_time_length_frames > parameters["min_track_length"]:
            frame_end_point = observation_time[0] + parameters["min_track_length"]

    # filter data for end point of each track
    tracks_end = observation_period_df[observation_period_df["FRAME"] == frame_end_point]

    # get track IDs that are present at the start and end of the observation period
    track_ids_start = np.array(tracks_start["TRACK_ID"].unique())
    track_ids_end = np.array(tracks_end["TRACK_ID"].unique())

    # find track IDs that are present at both the start and end of the observation period
    unique_common_track_ids = np.intersect1d(track_ids_start, track_ids_end)

    # list of unique track IDs to be plotted
    trackID_list = np.unique(unique_common_track_ids)

    return observation_period_df, trackID_list


def _calc_rel_vel(observation_period_df, track_id, min_vel_lim, max_vel_lim, max_vel, min_vel, parameters):
    single_track_df = observation_period_df[observation_period_df["TRACK_ID"] == track_id].copy()

    start_frame = single_track_df["FRAME"].min()
    end_frame = single_track_df["FRAME"].max()
    delta_frame = end_frame - start_frame

    delta_hour = delta_frame / parameters["frames_per_hour"]

    row_start = single_track_df[single_track_df["FRAME"] == start_frame]
    row_end = single_track_df[single_track_df["FRAME"] == end_frame]

    start_x = np.array(row_start["POSITION_X"])[0]
    start_y = np.array(row_start["POSITION_Y"])[0]
    end_x = np.array(row_end["POSITION_X"])[0]
    end_y = np.array(row_end["POSITION_Y"])[0]
    delta_x = end_x - start_x
    delta_y = end_y - start_y

    single_track_df["X_from_origin"] = single_track_df["POSITION_X"] - start_x
    single_track_df["Y_from_origin"] = single_track_df["POSITION_Y"] - start_y

    # relative velocity parallel to flow
    rel_vel = normalize_speed(delta_x / delta_hour, min_vel_lim, max_vel_lim)

    if delta_x / delta_hour > max_vel:
        max_vel = delta_x / delta_hour
    if delta_x / delta_hour < min_vel:
        min_vel = delta_x / delta_hour

    # return updated max and min velocity parallel to flow
    return rel_vel, max_vel, min_vel, single_track_df, delta_x, delta_y


def compute_speeds(parameters, key_file, subfolder="tracking_data"):
    """ This function computes the migration speed of each track in the tracking data."""

    # read parameter
    interval = parameters["time_lag"]
    decimal_places = parameters["decimal_places"]
    output_folder = parameters["output_folder"]
    # New feature flag (defaults to True if missing)
    variable_initial_lag = bool(parameters.get("variable_initial_lag", True))

    tracking_data_path = Path(output_folder).joinpath(subfolder)

    # iterate over the key_file to process each reported tracking file
    for index, row in key_file.iterrows():
        migration_speed_df = pd.DataFrame()

        # check if the tracking file exists
        tracking_file = "tracking_data_%s_%s_%s.csv" % (
            row["treatment"],
            row["color"],
            row["experimentID"]
        )

        print("Compute speeds for file ", row["filename"])

        # read the tracking data
        tracks_df_ = pd.read_csv(tracking_data_path.joinpath(tracking_file), low_memory=False)
        # filter the columns to keep only the relevant ones
        tracks_df = tracks_df_[["TRACK_ID", "POSITION_X", "POSITION_Y", "POSITION_T", "FRAME", "ORIGIN_X", "ORIGIN_Y"]]

        status = 0
        num_tracks = len(tracks_df["TRACK_ID"].unique())
        print("Number of tracks to analyze: ", num_tracks)

        # iterate over each track ID to compute the speed
        for track_id in tracks_df["TRACK_ID"].unique():
            # ensure we can safely clean up optional objects
            dist = None

            # extract the single track data
            single_track_df = tracks_df[tracks_df["TRACK_ID"] == track_id]

            # sort by frame and reset positional index
            single_track_df = single_track_df.sort_values(by="FRAME").reset_index(drop=True)

            # build numpy arrays
            x = single_track_df["POSITION_X"].to_numpy(dtype=float)
            y = single_track_df["POSITION_Y"].to_numpy(dtype=float)
            t_sec = single_track_df["POSITION_T"].to_numpy(dtype=float)
            t_h = t_sec / 3600.0

            if variable_initial_lag:
                # expanding window for early frames, capped at interval
                n = x.shape[0]
                idx = np.arange(n, dtype=int)
                lag = np.minimum(idx, int(interval))  # 0,1,2,...,interval,interval,...
                valid_rows = lag > 0
                i = np.where(valid_rows)[0]
                # look-back indices for each valid row
                p = i - lag[i]

                # allocate and fill displacement/time arrays
                dx = np.full(n, np.nan, dtype=float)
                dy = np.full(n, np.nan, dtype=float)
                dt_h = np.full(n, np.nan, dtype=float)

                dx[i] = x[i] - x[p]
                dy[i] = y[i] - y[p]
                dt_h[i] = t_h[i] - t_h[p]

                valid = dt_h > 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    step = np.where(valid, np.sqrt(dx**2 + dy**2), np.nan)
                    vel = np.where(valid, step / dt_h, np.nan)
                    vel_x = np.where(valid, dx / dt_h, np.nan)
                    vel_y = np.where(valid, dy / dt_h, np.nan)
                    phi_deg = np.where(valid, np.degrees(np.arctan2(dy, -dx)), np.nan)

                # assign rounded outputs
                single_track_df["step_size"] = np.round(step, decimal_places)
                single_track_df["step_size_x"] = np.round(np.where(valid, dx, np.nan), decimal_places)
                single_track_df["step_size_y"] = np.round(np.where(valid, dy, np.nan), decimal_places)

                single_track_df["vel_mu_per_h"] = np.round(vel, decimal_places)
                single_track_df["vel_x_mu_per_h"] = np.round(vel_x, decimal_places)
                single_track_df["vel_y_mu_per_h"] = np.round(vel_y, decimal_places)

                single_track_df["phi"] = np.round(phi_deg, decimal_places)
                single_track_df["lag_used"] = lag

            else:
                # original fixed-lag behavior, but without fillna(0)
                dist = single_track_df.diff(int(interval))
                dt_h = dist["POSITION_T"].to_numpy(dtype=float) / 3600.0
                dx = dist["POSITION_X"].to_numpy(dtype=float)
                dy = dist["POSITION_Y"].to_numpy(dtype=float)

                valid = dt_h > 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    step = np.where(valid, np.sqrt(dx**2 + dy**2), np.nan)
                    vel = np.where(valid, step / dt_h, np.nan)
                    vel_x = np.where(valid, dx / dt_h, np.nan)
                    vel_y = np.where(valid, dy / dt_h, np.nan)
                    phi_deg = np.where(valid, np.degrees(np.arctan2(dy, -dx)), np.nan)

                single_track_df["step_size"] = np.round(step, decimal_places)
                single_track_df["step_size_x"] = np.round(np.where(valid, dx, np.nan), decimal_places)
                single_track_df["step_size_y"] = np.round(np.where(valid, dy, np.nan), decimal_places)

                single_track_df["vel_mu_per_h"] = np.round(vel, decimal_places)
                single_track_df["vel_x_mu_per_h"] = np.round(vel_x, decimal_places)
                single_track_df["vel_y_mu_per_h"] = np.round(vel_y, decimal_places)

                single_track_df["phi"] = np.round(phi_deg, decimal_places)
                single_track_df["lag_used"] = int(interval)

            # metadata
            single_track_df["filename"] = tracking_file
            single_track_df["condition"] = tracks_df_["condition"].iloc[0]

            # time columns: absolute hours and relative (start at 0)
            single_track_df["time_in_h"] = np.round(t_h, decimal_places)
            single_track_df["time_from_start_h"] = np.round(t_h - t_h[0], decimal_places)

            # save current track data to the migration speed dataframe
            if len(migration_speed_df.index) > 1:
                migration_speed_df = pd.concat([migration_speed_df, single_track_df], ignore_index=True)
            else:
                migration_speed_df = single_track_df.copy()

            # print progress
            status += 1
            if status % 500 == 0:
                print("%s out of %s tracks analyzed." % (status, num_tracks))

            # clean up to save memory
            del single_track_df
            if dist is not None:
                del dist

        # store the migration speed dataframe to a csv file
        migration_speed_filepath = Path(output_folder).joinpath(
            "speed_data",
            "migration_speed_df_%s_%s_%s.csv" % (
                row["treatment"],
                row["color"],
                row["experimentID"]
            )
        )
        migration_speed_df.to_csv(str(migration_speed_filepath), index=False)

        # remove the migration speed dataframe from memory for the next file
        del migration_speed_df

    return

def gaps_for_track(frames: pd.Series) -> pd.DataFrame:
    # sorted unique frames for this track
    arr = np.sort(frames.unique())
    if arr.size < 2:
        return pd.DataFrame(columns=['gap_start', 'gap_end', 'gap_length'])

    prev = arr[:-1]
    curr = arr[1:]
    gap_mask = (curr - prev) > 1

    if not np.any(gap_mask):
        return pd.DataFrame(columns=['gap_start', 'gap_end', 'gap_length'])

    starts = prev[gap_mask] + 1
    ends = curr[gap_mask] - 1
    lengths = ends - starts + 1

    return pd.DataFrame({
        'gap_start': starts,
        'gap_end': ends,
        'gap_length': lengths
    })


def normalize_speed(vel_x, min_value, max_value):
    '''
    normalize the speed values to the range [0,1]
    '''
    rel_vel = (vel_x - min_value)/(max_value  - min_value)
    return rel_vel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def prepare_tracking_data(parameters, key_file, subfolder="tracking_data"):
    """
    This function reads the tracking data from the csv files and prepares it for further analysis.
    """

    base_folder = parameters["base_folder"]
    output_folder = parameters["output_folder"]

    tracking_data_df = pd.DataFrame()

    column_dtypes = {   'TRACK_ID': 'int16',
                        'FRAME': 'int16',
                        'POSITION_X': 'float16',
                        'POSITION_Y': 'float16',
                        'POSITION_T': 'float32'}

    print(list(column_dtypes))

    for index, row in key_file.iterrows():

        print("Processing file: ", row["filename"])
        #data = pd.read_csv(base_folder + row["filename"], low_memory=False).drop([0, 1, 2])
        data = pd.read_csv(base_folder + row["filename"], low_memory=False)

        data_ = data[list(column_dtypes)]

        data_.insert(0, "filename", row["filename"])
        data_.insert(0, "condition", row["condition"])

        data_ = data_.astype(column_dtypes)
        data_ = data_.sort_values(by="FRAME")

        for track_id in data_["TRACK_ID"].unique():
            single_track_df = data_[data_["TRACK_ID"] == track_id]
            track_length = len(single_track_df.index)
            start_frame = single_track_df["FRAME"].min()
            end_frame = single_track_df["FRAME"].max()
            ### uncomment to check for gaps
            # if track_length < end_frame - start_frame + 1:
            #    print("Track: ", track_id, "with length ", track_length, " has a gap")
            #    print(np.array(single_track_df["FRAME"]))
            # else:
            #    print("Track: ", track_id, "with length ", track_length," has no gap")
            #    print(np.array(single_track_df["FRAME"]))
            ###
            start_x = np.array(single_track_df["POSITION_X"])[0]
            start_y = np.array(single_track_df["POSITION_Y"])[0]
            data_.loc[data_.TRACK_ID == track_id, "START_X"] = start_x
            data_.loc[data_.TRACK_ID == track_id, "START_Y"] = start_y

            if track_length < parameters["min_track_length"]:
                data_ = data_[data_["TRACK_ID"] != track_id]

        ### for trajectory plots
        data_["ORIGIN_X"] = data_["POSITION_X"] - data_["START_X"]
        data_["ORIGIN_Y"] = data_["POSITION_Y"] - data_["START_Y"]
        data_["ORIGIN_L"] = np.sqrt(data_["ORIGIN_X"] ** 2 + data_["ORIGIN_Y"] ** 2)

        outpath = output_folder + subfolder + "/tracking_data_%s_%s_%s.csv" % (row["treatment"], row["color"], row["experimentID"])
        print("Saving tracking data to: ", outpath)
        data_.to_csv(outpath, index=False)

        print("##################")
        if len(tracking_data_df.index) > 10:
            # tracking_data_df = tracking_data_df.append(data_)
            tracking_data_df = pd.concat([tracking_data_df, data_], ignore_index=True)
        else:
            tracking_data_df = data_.copy()

        del data_
        del data

    return tracking_data_df

def plot_quality_control(parameters, key_file, subfolder = "tracking_data"):

    tracking_data_path = parameters["output_folder"] + subfolder + "/"

    for index, row in key_file.iterrows():

        tracking_file = "tracking_data_%s_%s_%s.csv" % (row["treatment"], row["color"], row["experimentID"])
        print("Plot quality control for file ", row["filename"])
        data = pd.read_csv(tracking_data_path + tracking_file, low_memory=False)

        # tracking_data_df_ = tracking_data_df[tracking_data_df["filename"] == filename]
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.scatterplot(data=data, x="FRAME", y="TRACK_ID")
        ax.set_title("Experiment ID %s" % row["experimentID"])

        fig.savefig(parameters["output_folder"] + "/quality_control/quality_control_%s_%s_%s.png" % (row["treatment"], row["color"], row["experimentID"]))



def compute_speeds(parameters, key_file, subfolder = "tracking_data"):

    interval = parameters["time_lag"]
    decimal_places = parameters["decimal_places"]
    output_folder = parameters["output_folder"]

    tracking_data_path = output_folder + subfolder + "/"

    for index, row in key_file.iterrows():

        migration_speed_df = pd.DataFrame()
        tracking_file = "tracking_data_%s_%s_%s.csv" % (row["treatment"], 
                                                        row["color"], row["experimentID"])
               
        print("Compute speeds for file ", row["filename"])
        
        tracks_df_ = pd.read_csv(tracking_data_path + tracking_file, low_memory=False)
        tracks_df = tracks_df_[["TRACK_ID", "POSITION_X", "POSITION_Y", 
                                "POSITION_T", "FRAME", "ORIGIN_X", "ORIGIN_Y"]]

        status = 0
        tracks_num = len(tracks_df["TRACK_ID"].unique())

        for track_id in tracks_df["TRACK_ID"].unique():
            single_track_df = tracks_df[tracks_df ["TRACK_ID"]==track_id]
            single_track_df = single_track_df.sort_values(by="FRAME")
            dist = single_track_df.diff(interval).fillna(0.)
            dist["time_in_h"] = dist["POSITION_T"]/3600.0

            single_track_df["step_size"] = np.round(np.sqrt(dist.POSITION_X**2 + dist.POSITION_Y**2),decimal_places)
            single_track_df["step_size_x"] = np.round(dist.POSITION_X,decimal_places)
            single_track_df["step_size_y"] =  np.round(dist.POSITION_Y,decimal_places)
            single_track_df["vel_mu_per_h"] = np.round(np.sqrt(dist.POSITION_X**2 + dist.POSITION_Y**2)/dist.time_in_h,decimal_places)
            single_track_df["vel_x_mu_per_h"] = np.round(dist.POSITION_X/dist.time_in_h,decimal_places)
            single_track_df["vel_y_mu_per_h"] =  np.round(dist.POSITION_Y/dist.time_in_h,decimal_places)
            
            single_track_df["phi"] =  np.round(np.arctan2(dist.POSITION_Y,-dist.POSITION_X)*180.0/np.pi,decimal_places)

            single_track_df["filename"] = tracking_file
            single_track_df["condition"] = tracks_df_["condition"].iloc[0]
            
            single_track_df["time_in_h"] = np.round(single_track_df["POSITION_T"]/3600.0,decimal_places)
            
            if len(migration_speed_df.index) > 1:
                migration_speed_df = pd.concat( [migration_speed_df, single_track_df], ignore_index=True)
            else:
                migration_speed_df = single_track_df.copy()
            
            status +=1 
            if status % 500 == 0:
                print("%s out of %s tracks analyzed." % (status,tracks_num)) 

            del single_track_df
            del dist

        migration_speed_filepath = output_folder + "speed_data/migration_speed_df_%s_%s_%s.csv" % (row["treatment"],
                                                                                         row["color"], 
                                                                                         row["experimentID"])
        migration_speed_df.to_csv(migration_speed_filepath, index=False)

        #data = pd.read_csv(tracking_data_path + tracking_file, low_memory=False)

        #data["DELTA_T"] = data["POSITION_T"] - data["POSITION_T"].shift(1)
        #data["DELTA_X"] = data["POSITION_X"] - data["POSITION_X"].shift(1)
        #data["DELTA_Y"] = data["POSITION_Y"] - data["POSITION_Y"].shift(1)

        #data["VEL_X"] = data["DELTA_X"] / data["DELTA_T"]
        #data["VEL_Y"] = data["DELTA_Y"] / data["DELTA_T"]

        #data["VEL"] = np.sqrt(data["VEL_X"] ** 2 + data["VEL_Y"] ** 2)

       # data.to_csv(tracking_data_path + tracking_file, index=False)

        del migration_speed_df

    return
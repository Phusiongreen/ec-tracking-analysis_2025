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

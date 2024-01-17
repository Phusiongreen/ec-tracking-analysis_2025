import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable



def plot_trajectories_per_file(parameters, key_file, subfolder = "tracking_data"):

    output_folder = parameters["output_folder"]
    tracking_data_path = output_folder + subfolder + "/"


    for index, row in key_file.iterrows():

        tracking_file = "tracking_data_%s_%s_%s.csv" % (row["treatment"], row["color"], row["experimentID"])
        print("Plotting trajectories from file: ", tracking_file)
        data = pd.read_csv(tracking_data_path + tracking_file, low_memory=False)

        phase_1_data_df = data[data["FRAME"] <= parameters["end_phase_1"]]

        fig, ax = plt.subplots(figsize=(9,9))

        for track_id in phase_1_data_df["TRACK_ID"].unique():
            single_track_df = data[data["TRACK_ID"] == track_id]
            ax.plot(single_track_df["POSITION_X"],single_track_df["POSITION_Y"])

        ax.set_xlim(-110,110)
        ax.set_ylim(-110,110)
        ax.axhline(0, color = "red", linestyle = "--")
        ax.axvline(0, color = "red", linestyle = "--")
        ax.set_title(row["filename"])
        ax.set_xlabel("$\Delta x$ in $\mu m$")
        ax.set_ylabel("$\Delta y$ in $\mu m$")
        ax.set_aspect(1)
        plt.tight_layout()
        plt.savefig(output_folder + "/trajectory_plots/trajectories_%s_%s_%s.png" % (row["treatment"], row["color"], row["experimentID"]))
        plt.savefig(output_folder + "/trajectory_plots/trajectories_%s_%s_%s.pdf" % (row["treatment"], row["color"], row["experimentID"]))
        plt.close()


def plot_trajectories_from_origin_per_file(parameters, key_file, subfolder = "tracking_data"):

    output_folder = parameters["output_folder"]
    tracking_data_path = output_folder + subfolder + "/"


    for index, row in key_file.iterrows():

        tracking_file = "tracking_data_%s_%s_%s.csv" % (row["treatment"], row["color"], row["experimentID"])
        print("Plotting trajectories (starting at origin) from file: ", tracking_file)
        data = pd.read_csv(tracking_data_path + tracking_file, low_memory=False)

        phase_1_data_df = data[data["FRAME"] <= parameters["end_phase_1"]]

        fig, ax = plt.subplots(figsize=(9,9))

        for track_id in phase_1_data_df["TRACK_ID"].unique():
            single_track_df = phase_1_data_df[phase_1_data_df["TRACK_ID"] == track_id]

            if len(single_track_df.index) < parameters["end_phase_1"]:
                continue

            end_x = np.array(single_track_df["ORIGIN_X"])[-1]
            end_y = np.array(single_track_df["ORIGIN_Y"])[-1]



            #rel_vel = get_color_from_speed(vel_x)

            ax.plot(single_track_df["ORIGIN_X"], single_track_df["ORIGIN_Y"])

            # ax.plot(single_track_df["ORIGIN_X"],single_track_df["ORIGIN_Y"], color = "#ff7f00")
            # ax.plot([end_x],[end_y], color = "black", marker = "o")

            ax.plot([end_x], [end_y], marker="o")

        ax.set_xlim(-110,110)
        ax.set_ylim(-110,110)
        ax.axhline(0, color = "red", linestyle = "--")
        ax.axvline(0, color = "red", linestyle = "--")
        ax.set_title(row["filename"])
        ax.set_xlabel("$\Delta x$ in $\mu m$")
        ax.set_ylabel("$\Delta y$ in $\mu m$")
        ax.set_aspect(1)
        plt.tight_layout()
        plt.savefig(output_folder + "/trajectory_plots/origin_trajectories_%s_%s_%s.png" % (row["treatment"], row["color"], row["experimentID"]))
        plt.savefig(output_folder + "/trajectory_plots/origin_trajectories_%s_%s_%s.pdf" % (row["treatment"], row["color"], row["experimentID"]))
        plt.close()


def plot_trajectories_from_origin_per_condition(parameters, key_file, subfolder = "tracking_data", number_of_tracks_per_condition = 1000):

    output_folder = parameters["output_folder"]
    tracking_data_path = output_folder + subfolder + "/"

    for condition in key_file["condition"].unique():
        key_file_condition = key_file[key_file["condition"] == condition]

        print("Plotting trajectories(starting at origin) for condition: ", condition)

        for treatment in key_file_condition["treatment"].unique():
            key_file_treatment = key_file_condition[key_file_condition["treatment"] == treatment]

            number_of_tracks_per_file = number_of_tracks_per_condition/len(key_file_treatment.index)

            for index, row in key_file_treatment.iterrows():

                tracking_file = "tracking_data_%s.csv" % row["experimentID"]
                print("Plotting trajectories (starting at origin) from file: ", tracking_file)
                data = pd.read_csv(tracking_data_path + tracking_file, low_memory=False)
                fig, ax = plt.subplots(figsize=(9,9))

                for index, row in key_file_treatment.iterrows():
                    tracking_file = "tracking_data_%s.csv" % row["experimentID"]

                    data = pd.read_csv(tracking_data_path + tracking_file, low_memory=False)

                    phase_1_data_df = data[data["FRAME"] <= parameters["end_phase_1"]]

                    for track_id in phase_1_data_df["TRACK_ID"].unique():
                        single_track_df = phase_1_data_df[phase_1_data_df["TRACK_ID"] == track_id]

                        if len(single_track_df.index) < parameters["end_phase_1"]:
                            continue

                    end_x = np.array(single_track_df["ORIGIN_X"])[-1]
                    end_y = np.array(single_track_df["ORIGIN_Y"])[-1]




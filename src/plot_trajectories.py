import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


cmap = matplotlib.colormaps["seismic"]
#max_value = 39.625
#min_value = -39.625
max_value = 40.0
min_value = -40.0

def get_color_from_speed(vel_x):
    '''
    
    '''
    rel_vel = (vel_x - min_value)/(max_value  - min_value)
    return rel_vel
    


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


def plot_trajectories_single_condition(parameters, key_file, subfolder = "tracking_data", figsize = (9,9)):

    fig, ax = plt.subplots(figsize=(9,9))

    output_folder = parameters["output_folder"]
    tracking_data_path = output_folder + subfolder + "/"

    average_data_df = pd.DataFrame(columns = ["condition", "treatment", "delta_x", "delta_y"])
    average_data_per_file_df = pd.DataFrame(columns = ["condition", "treatment", "tracking_file", "color", 
                                                       "delta_x", "delta_y", "total_distance"])



    return fig, ax



def plot_trajectories_from_origin_per_condition(parameters, key_file, subfolder = "tracking_data", number_of_tracks_per_condition = 1000):

    output_folder = parameters["output_folder"]
    tracking_data_path = output_folder + subfolder + "/"

    average_data_df = pd.DataFrame(columns = ["condition", "treatment", "delta_x", "delta_y"])
    average_data_per_file_df = pd.DataFrame(columns = ["condition", "treatment", "tracking_file", "color", 
                                                       "delta_x", "delta_y", "total_distance"])

    avg_data_idx = 0
    avg_data_per_file_idx = 0
    for condition in key_file["condition"].unique():
        key_file_condition = key_file[key_file["condition"] == condition]

        print("Plotting trajectories(starting at origin) for condition: ", condition)

        for treatment in key_file_condition["treatment"].unique():
            key_file_treatment = key_file_condition[key_file_condition["treatment"] == treatment]

            print("Plotting trajectories (starting at origin) for condition %s and %s: " % (condition, treatment) )
            print("Number of files: %s " % len(key_file_treatment.index))
            #print(key_file_treatment)


            number_of_tracks_per_file = int(number_of_tracks_per_condition/len(key_file_treatment.index))
            print("sample %s tracks per file" % number_of_tracks_per_file)

            track_counter = 0
            
            center_x = 0.0
            center_y = 0.0

            fig, ax = plt.subplots(figsize=(9,9))

            #colors = cm.rainbow(np.linspace(0, 1, len(key_file_treatment.index)+1))   
            #print("Colors")
            #print(colors)

            for index, row in key_file_treatment.iterrows():

                center_per_file_x = 0.0
                center_per_file_y = 0.0
                total_dist_per_file = 0.0

                tracking_file = "tracking_data_%s_%s_%s.csv" % (treatment, row["color"], row["experimentID"])
                
                data = pd.read_csv(tracking_data_path + tracking_file, low_memory=False)

                phase_1_data_df = data[data["FRAME"] <= parameters["end_phase_1"]]

                trackID_list = np.array(phase_1_data_df["TRACK_ID"].unique())
                print("Available tracks: %s" % len(trackID_list))
                trackID_list = np.random.choice(trackID_list, number_of_tracks_per_file)
                #phase_1_data_df = phase_1_data_df.sample(n = int(number_of_tracks_per_file))

                print("sampled %s tracks for file %s" % (len(trackID_list),tracking_file))
                track_counter += len(trackID_list)
                
                for track_id in trackID_list:
                    single_track_df = phase_1_data_df[phase_1_data_df["TRACK_ID"] == track_id]

                    #if len(single_track_df.index) < parameters["end_phase_1"]:
                    #    continue

                    end_x = np.array(single_track_df["ORIGIN_X"])[-1]
                    end_y = np.array(single_track_df["ORIGIN_Y"])[-1]

                    total_dist_per_file += np.sqrt(end_x**2 + end_y**2)

                    center_x += end_x
                    center_y += end_y

                    center_per_file_x += end_x
                    center_per_file_y += end_y

                    rel_vel = get_color_from_speed(-end_x)

                    # ax.plot(single_track_df["ORIGIN_X"],single_track_df["ORIGIN_Y"], color = "#ff7f00")
                    # ax.plot([end_x],[end_y], color = "black", marker = "o")

                    # color by track id             
                    #ax.plot(single_track_df["ORIGIN_X"], single_track_df["ORIGIN_Y"])
                    #ax.plot([end_x], [end_y], marker="o")

                    # color by velocity   
                    ax.plot(single_track_df["ORIGIN_X"],single_track_df["ORIGIN_Y"], color = cmap(rel_vel))
                    ax.plot([end_x],[end_y], color = "black", marker = "o", alpha=0.5)  
                    #ax.plot([end_x],[end_y], color = colors[index], marker = "o", alpha=0.5)  

                center_per_file_x = center_per_file_x/number_of_tracks_per_file
                center_per_file_y = center_per_file_y/number_of_tracks_per_file
                total_dist_per_file = total_dist_per_file/number_of_tracks_per_file

                average_data_per_file_df.at[avg_data_per_file_idx, "condition"] = condition
                average_data_per_file_df.at[avg_data_per_file_idx, "treatment"] = treatment
                average_data_per_file_df.at[avg_data_per_file_idx, "tracking_file"] = tracking_file
                average_data_per_file_df.at[avg_data_per_file_idx, "delta_x"] = center_per_file_x
                average_data_per_file_df.at[avg_data_per_file_idx, "delta_y"] = center_per_file_y
                average_data_per_file_df.at[avg_data_per_file_idx, "total_distance"] = total_dist_per_file
                average_data_per_file_df.at[avg_data_per_file_idx, "color"] = row["color"]

                avg_data_per_file_idx += 1

            center_x = center_x/track_counter
            center_y = center_y/track_counter

            average_data_df.at[avg_data_idx, "condition"] = condition
            average_data_df.at[avg_data_idx, "treatment"] = treatment
            average_data_df.at[avg_data_idx, "delta_x"] = center_x
            average_data_df.at[avg_data_idx, "delta_y"] = center_y

            avg_data_idx += 1

            ax.plot([center_x],[center_y], color = "black", marker = "x", markersize = 10)

            print("Plotted %s tracks for condition %s and %s" % (track_counter, condition, treatment))
            #print("Plotted %s tracks for condition %s and %s" % (number_of_tracks_per_condition, condition, treatment))

            ax.set_xlim(-50,50)
            ax.set_ylim(-50,50)
            ax.axhline(0, color = "red", linestyle = "--")
            ax.axvline(0, color = "red", linestyle = "--")
            ax.set_title("# %s tracks for treatment: %s in condition: %s," % (number_of_tracks_per_condition, treatment, condition))
            ax.set_xlabel("$\Delta x$ in $\mu m$")
            ax.set_ylabel("$\Delta y$ in $\mu m$")
            ax.set_aspect(1)
            plt.tight_layout()
            plt.savefig(output_folder + "/trajectory_plots/condition/origin_%s_trajectories_treatment_%s_condition_%s.png" % 
                        (number_of_tracks_per_condition, condition, treatment))
            plt.savefig(output_folder + "/trajectory_plots/condition/origin_%s_trajectories_treatment%s_condition_%s.pdf" % 
                        (number_of_tracks_per_condition, condition, treatment))
            plt.close()

    average_data_df.to_csv(output_folder + "/trajectory_plots/condition/average_origin_trajectories_%s_tracks_per_condition.csv" % number_of_tracks_per_condition, index = False)
    average_data_per_file_df.to_csv(output_folder + "/trajectory_plots/condition/average_origin_trajectories_%s_tracks_per_file.csv" % number_of_tracks_per_condition, index = False)
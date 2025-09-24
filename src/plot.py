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



def normalize_speed(vel_x, min_value, max_value):
    '''
    normalize the speed values to the range [0,1]
    '''
    rel_vel = (vel_x - min_value)/(max_value  - min_value)
    return rel_vel
    

def plot_trajectories(parameters, tracking_data, ax):

    for track_id in tracking_data["TRACK_ID"].unique():
        single_track_df = tracking_data[tracking_data["TRACK_ID"] == track_id]
        ax.plot(single_track_df["POSITION_X"],single_track_df["POSITION_Y"])

    return ax

def plot_trajectories_key_file(parameters, key_file, subfolder = "tracking_data"):

    output_folder = parameters["output_folder"]
    tracking_data_path = output_folder + subfolder + "/"


    for index, row in key_file.iterrows():

        tracking_file = "tracking_data_%s_%s_%s.csv" % (row["treatment"], row["color"], row["experimentID"])
        print("Plotting trajectories from file: ", tracking_file)
        data = pd.read_csv(tracking_data_path + tracking_file, low_memory=False)

        phase_1_data_df = data[data["FRAME"] <= parameters["observation_time"][1]]

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

        phase_1_data_df = data[data["FRAME"] <= parameters["observation_time"][1]]

        fig, ax = plt.subplots(figsize=(9,9))

        for track_id in phase_1_data_df["TRACK_ID"].unique():
            single_track_df = phase_1_data_df[phase_1_data_df["TRACK_ID"] == track_id]

            if len(single_track_df.index) < parameters["observation_time"][1]:
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

                phase_1_data_df = data[data["FRAME"] <= parameters["observation_time"][1]]

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
            _save(fig, out_base, f"qc5_gap_time_{row['treatment']}_{row['color']}_{row['experimentID']}.png")
        else:
            print("  ✓ No gaps detected.")
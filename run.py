import argparse
import os
import sys
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.io import read_parameters
from src.compute_speeds import prepare_tracking_data
from src.compute_speeds import plot_quality_control
from src.compute_speeds import compute_speeds
from src.plot_trajectories import plot_trajectories_per_file
from src.plot_trajectories import plot_trajectories_from_origin_per_file
from src.plot_trajectories import plot_trajectories_from_origin_per_condition


#########################
# read parameters
#########################

parser = argparse.ArgumentParser(
    description='This script runs ec movement trajectory analysis. '
                'You need to provide a parameter file and optionally '
                'an input folder, a key file, and an output folder.',
    epilog='Example usage: python run.py parameters.yaml --input_folder /path/to/input '
           '--key_file /path/to/key --output_folder /path/to/output',
)
#parser.add_argument('param', type=str, help='Path to the parameter file.')
parser.add_argument('param', type=str, help='Path to the parameter file.')
parser.add_argument('--input_folder', type=str, help='Path to the input folder.')
parser.add_argument('--key_file', type=str, help='Path to the key file.')
parser.add_argument('--output_folder', type=str, help='Path to the output folder.')

if len(sys.argv) < 4:
    #parser.print_usage()
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
print("-------")
print("reading parameters from: ", args.param)
print("-------")

parameter_file  = args.param
parameters = read_parameters(parameter_file)               

# Override parameters with command line arguments if provided
if args.input_folder:
    parameters["input_folder"] = args.input_folder
if args.output_folder:
    parameters["output_folder"] = args.output_folder
if args.key_file:
    parameters["key_file"] = args.key_file

print("-------")
print("used parameter values: ")
print(parameters)
print("-------")

output_folder = parameters["output_folder"]
# save parameter file to output
with open(output_folder + "/parameters.yml", 'w') as outfile:
    yaml.dump(parameters, outfile)

#########################
# read key file
#########################

key_file_path = parameters["key_file"]
key_file = pd.read_csv(key_file_path)
print(key_file.head())

key_file.to_csv(output_folder + "/key_file.csv", index=False)


#########################
# create subfolders
#########################

# create a subfolder for each subject
subjects = ["tracking_data", "speed_data","trajectory_plots","velocity_plots","quality_control"]

for subject in subjects:
    if not os.path.exists(output_folder + "/" + subject):
        os.mkdir(output_folder + "/" + subject)

workflows = parameters["workflows"]
print("The following workflows will be executed: ", workflows)

#["preprocess_trajectories", "tracking_data", "quality_control", "plot_trajectories", "plot_speeds"]

#########################
# 1.Step : preprocess data
#########################

<<<<<<< HEAD
tracking_data = prepare_tracking_data(parameters, key_file)
# tracking_data.to_csv(output_folder + "tracking_data/tracking_data.csv", index=False)

#print(tracking_data.head())
=======
if "preprocess_trajectories" in workflows:
    tracking_data = prepare_tracking_data(parameters, key_file)
    tracking_data.to_csv(output_folder + "tracking_data/tracking_data.csv", index=False)
>>>>>>> 418c66b8d7b56b6037263d6b1bf51369824ae4f2

#########################
# 2.Step : quality control
#########################


if "qc_metrics" in workflows:
    plot_quality_control(parameters, key_file, subfolder = "tracking_data")

#########################
# 3. Step : plot trajectories
#########################

if "plot_trajectories" in workflows:
    #plot_trajectories_per_file(parameters, key_file, subfolder = "tracking_data")
    #plot_trajectories_from_origin_per_file(parameters, key_file, subfolder = "tracking_data")
    plot_trajectories_from_origin_per_condition(parameters, key_file, subfolder = "tracking_data", number_of_tracks_per_condition = 1000)

#########################
# compute speeds
########################

if "compute_speeds" in workflows:
    compute_speeds(parameters, key_file, subfolder = "tracking_data")



#########################
# run analysis - velocity plots
#########################

# grouped by condition and filename

plt.rcParams.update({'font.size': 14})

plot_migration_speeds = migration_speed_df.dropna()
#plot_migration_speeds = plot_migration_speeds.iloc[::subsampling_n, :]
#plot_migration_speeds = plot_migration_speeds.sample(frac=subsample_frac, replace=True, random_state=1)
plot_migration_speeds = plot_migration_speeds[plot_migration_speeds["FRAME"] >= interval]


#fig, ax = plt.subplots(len(intervals),figsize=(15,30))
fig, ax = plt.subplots(1, figsize=(16,8))

sns.lineplot(x = "time_in_h", y = "vel_x_mu_per_h", hue = "condition", hue_order = hue_order,
             data = plot_migration_speeds, style="filename", ax=ax, ci = 90)# , errorbar = errorbar)

#sns.lineplot(x = "FRAME", y = "vel_x", hue = "condition", data = plot_migration_speeds, ax=ax, ci = 90)
#interval_length_min = 5*intervals[i]
#ax.set_title("time interval %s min" % interval_length_min)
ax.set_ylabel("velocity parallel to flow in microns/h")
ax.set_ylim(-7.0,7.0)
ax.axhline(y = 0.0, color = 'r', linestyle = 'dashed')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig(output_folder + "velocity_parallel_to_flow_filename_all.pdf")
plt.savefig(output_folder + "velocity_parallel_to_flow_filename_all.png")

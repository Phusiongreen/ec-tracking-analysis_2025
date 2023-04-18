import argparse
import os
import yaml

from utils.io import read_parameters


#########################
# read parameters
#########################

parser = argparse.ArgumentParser(description='Run ec movement trajectory analysis ')
parser.add_argument('param', type=str, help='Path to the parameter file.')

args = parser.parse_args()
print("-------")
print("reading parameters from: ", args.param)
print("-------")

parameter_file  = args.param
parameters = read_parameters(parameter_file)               

print("-------")
print("used parameter values: ")
print(parameters)
print("-------")


#########################
# create output folders
#########################

output_folder = parameters["output_folder"]
# save parameter file to output
with open(output_folder + "/parameters.yml", 'w') as outfile:
    yaml.dump(parameters, outfile)

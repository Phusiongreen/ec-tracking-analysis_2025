#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.append("../")
from src.io import read_parameters
from src.plot import plot_quality_control


# In[2]:

# read parameters and key file

parameter_file = "/home/jpa/PycharmProjects/ec-tracking-analysis_2025/data/collectivity/parameters_collectivity.yml"
parameters = read_parameters(parameter_file)        

output_folder = Path(parameters["output_folder"])

if not os.path.exists(output_folder.joinpath("quality_control")):
    os.mkdir(output_folder.joinpath("quality_control"))

key_file_path = parameters["key_file"]
key_file = pd.read_csv(key_file_path)
print(key_file.head())


# In[3]:

subfolder="tracking_data"
tracking_data_path = Path(parameters["output_folder"]).joinpath(subfolder)

plot_quality_control(parameters, key_file, tracking_data_path)


# In[ ]:

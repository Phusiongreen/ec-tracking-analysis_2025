#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import sys
from pathlib import Path
import pandas as pd

sys.path.append("../")
from src.io import read_parameters
from src.computation import prepare_tracking_data


# In[15]:

# read parameters and key file
parameter_file = "/home/jpa/PycharmProjects/ec-tracking-analysis_2025/data/collectivity/parameters_collectivity.yml"
parameters = read_parameters(parameter_file)       


key_file_path = parameters["key_file"]
key_file = pd.read_csv(key_file_path)
print(key_file.head())

output_folder = Path(parameters["output_folder"])


# In[16]:

key_file.columns = key_file.columns.str.strip()
print("Key file columns:", key_file.columns)

for index, row in key_file.iterrows():
    print("Processing file: ", row["filename"])


# In[17]:

if not os.path.exists(output_folder.joinpath("tracking_data")):
    os.mkdir(output_folder.joinpath("tracking_data"))
    
prepare_tracking_data(parameters, key_file)


# In[ ]:

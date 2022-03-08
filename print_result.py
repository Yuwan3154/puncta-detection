from puncta_detect_util import print_result
import os
import pandas as pd
import numpy as np

channels_of_interest = [0, 1]
result_files = []
for file in os.listdir("results"):
    if file.endswith(".csv"):
        result_files.append(os.path.join("results", file))

for result_csv_file in result_files:
    print("Starting on output file", result_csv_file)
    print_result(pd.read_pickle(result_csv_file[:-4]), channels_of_interest)

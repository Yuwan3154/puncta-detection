from puncta_detect_util import print_result
import os
import pandas as pd
import numpy as np

result_files = []
for file in os.listdir("."):
    if file.endswith(".csv"):
        result_files.append(file)

for result_csv_file in result_files:
    print_result(pd.read_pickle(result_csv_file[:-4]))

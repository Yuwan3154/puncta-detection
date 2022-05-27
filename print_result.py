from puncta_detect_util import print_result, manual_print_result
import os
from os.path import join
import pandas as pd
import numpy as np

targets = ["15_pixel_data_03-03-2022_89.5_ DOPC_10_DOPS_0.5_ Atto_200 nM Atto 488 ALG2"]

channels_of_interest = [0, 1]
detail, frame_quality, square_quality = True, True, False
result_files = []
if not os.path.exists("results"):
    os.mkdir("results")
for file in os.listdir("results"):
    if not file.endswith(".csv") and file in targets:
        result_files.append(join("results", file))

for result_file in result_files:
    print("Starting on output file", result_file)
    result_df = pd.read_pickle(result_file)
    if not os.path.exists(f"{result_file}.csv"):
        result_df.to_csv(f"{result_file}.csv", index=False)
    print_result(result_df, channels_of_interest, detail, frame_quality, square_quality)

if not os.path.exists("manual_results"):
    os.mkdir("manual_results")
manual_channels_of_interest = [0, 1, 2]
result_files = []
for file in os.listdir("manual_results"):
    if not file.endswith(".csv") and file in targets:
        result_files.append(join("manual_results", file))

for result_file in result_files:
    print("Starting on manual output file", result_file)
    manual_label_df = pd.read_pickle(result_file)
    if not os.path.exists(f"{result_file}.csv"):
        manual_label_df.to_csv(f"{result_file}.csv", index=False)
    manual_print_result(manual_label_df, manual_channels_of_interest, detail)

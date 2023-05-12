from genericpath import exists
import os
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import math
import itertools
from puncta_detect_util import channel_quality
from three_in_one import process_dir
import argparse
import warnings

def extract_summary(summary_df, result_df, channels_of_interest, index, frame_quality=True, square_quality=True):
    """
    Function used for extracting the summary stats using all images in the same experiment folder.
    """
    if square_quality:
        result_df = result_df[result_df["square quality"]]
    if frame_quality:
        result_df = channel_quality(result_df, channels_of_interest)
        result_df = result_df[result_df[f"quality ch{channels_of_interest[0]}"]]

    # number of GUV
    summary_df.loc[index, "number of GUV"] = len(result_df)

    # colocalization
    for ch1, ch2 in itertools.combinations(channels_of_interest, 2):
        cur_chs_df = result_df[result_df[f"colocalization weight ch{ch1} ch{ch2}"].notna()]
        weighted_prop = np.average(cur_chs_df[f"colocalization ch{ch1} ch{ch2}"].to_numpy(dtype=float), weights=cur_chs_df[f"colocalization weight ch{ch1} ch{ch2}"].to_numpy(dtype=float)) if sum(cur_chs_df[f"colocalization weight ch{ch1} ch{ch2}"].to_numpy(dtype=float)) > 0.0 else 0.0
        unweighted_prop = np.mean(cur_chs_df[f"colocalization ch{ch1} ch{ch2}"])
        summary_df.loc[index, f"weighted colocalization ch{ch1} ch{ch2}"], summary_df.loc[index, f"unweighted colocalization ch{ch1} ch{ch2}"] = weighted_prop, unweighted_prop
    if len(channels_of_interest) > 2:
        for ch1, ch2, ch3 in itertools.combinations(channels_of_interest, 3):
            cur_chs_df = result_df[result_df[f"colocalization weight ch{ch1} ch{ch2} ch{ch3}"].notna()]
            weighted_prop = np.average(cur_chs_df[f"colocalization ch{ch1} ch{ch2} ch{ch3}"].to_numpy(dtype=float), weights=cur_chs_df[f"colocalization weight ch{ch1} ch{ch2} ch{ch3}"].to_numpy(dtype=float)) if sum(cur_chs_df[f"colocalization weight ch{ch1} ch{ch2} ch{ch3}"].to_numpy(dtype=float)) > 0.0 else 0.0
            unweighted_prop = np.mean(cur_chs_df[f"colocalization ch{ch1} ch{ch2} ch{ch3}"])
            summary_df.loc[index, f"weighted colocalization ch{ch1} ch{ch2} ch{ch3}"], summary_df.loc[index, f"unweighted colocalization ch{ch1} ch{ch2} ch{ch3}"] = weighted_prop, unweighted_prop

    # punctateness
    for ch in channels_of_interest:
        ch_percent = np.mean(result_df[f"new punctate ch{ch}"])
        summary_df.loc[index, f"new punctate ch{ch}"] = ch_percent

if __name__ == "__main__":
    #warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Process GUV punctateness.')
    parser.add_argument("--file", metavar="Summary CSV", type=str, nargs=1,
                        help="the path to a .csv file with at least three columns: 1. channels of interest [int] "
                             "2. detect channel [int] 3. experiment folder [path].")
    parser.add_argument("--label", metavar="Detail CSV label", type=str, nargs="+",
                        help="the prefix labels for the result of the individual folders.")
    parser.add_argument("--detect-threshold", type=float, nargs=1,
                        help="the confidence level cutoff for GUV segmentation.")
    parser.add_argument("--puncta-threshold", type=int, nargs="+",
                        help="the minimum number of pixels in a group to call puncta.")
    parser.add_argument("--detail", type=bool, const=True, default=False, nargs="?")
    parser.add_argument("--zstack", metavar="Type of series", type=bool, const=True, default=False, nargs="?",
                        help="Type of series for the images taken; flag if the input is Z-stack")
    args = vars(parser.parse_args())
    print("Arguments", args)
    meta_summary_file, meta_labels, pixel_thresholds, detail, zstack, detect_threshold = args["file"][0], args["label"], args["puncta_threshold"], args["detail"], args["zstack"], args["detect_threshold"][0]
    frame_quality, square_quality = True, True
    assert len(meta_labels) == len(pixel_thresholds)
    for i in range(len(pixel_thresholds)):
        meta_label, pixel_threshold = meta_labels[i], pixel_thresholds[i]
        summary_file = f"{meta_label}_{meta_summary_file}"
        summary_df = pd.read_csv(summary_file, index_col=False) if os.path.exists(summary_file) else pd.read_csv(meta_summary_file, index_col=False)
        print(f"Starting on {summary_file}")
        for index, experiment_row in summary_df.iterrows():
            if pd.DataFrame.isna(experiment_row)["experiment folder"]:
                continue
            experiment_row.fillna(0, inplace=True)
            exp_dir, channels_of_interest, detect_channel = experiment_row["experiment folder"], list(range(int(experiment_row["channels of interest"]))), int(experiment_row["detect channel"])
            # try:
            #     if not np.isnan(summary_df.loc[index, "number of GUV"]):
            #         continue
            # except:
            #     pass
            print(f"Starting on {exp_dir}")
            cur_result_df = process_dir([exp_dir], channels_of_interest, detect_channel, meta_label, detect_threshold, pixel_threshold, zstack, detail)
            extract_summary(summary_df, cur_result_df, channels_of_interest, index, frame_quality, square_quality)
            summary_df.to_csv(summary_file, index=False)

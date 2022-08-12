from puncta_detect_util import *
import torch
import numpy as np
import pandas as pd
import math
import itertools
from scipy.signal import correlate2d
from skimage import io
from skimage.filters import threshold_otsu, median, gaussian
from skimage.morphology import disk
from skimage import measure
from imutils import contours, grab_contours
import scipy.ndimage.filters as filters
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture
import cv2 as cv
import joblib
import os
from os.path import join
import subprocess
from sklearn.decomposition import PCA
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

manual_label_file_path_list = ["Manual_label_Feb_20_2022 - manual_label_example.csv",
                                "Manual_label_Feb_21_2022_final.csv",
                                "Repeat_Manual_label_Feb_21_2022 - manual_label_example.csv"
                            ]
pixel_threshold = 5
for up_ch in [0]:
    date = "07_21_22"
    labels = [ f"{date}_02-20-2022_overlap_ch{up_ch}_upstream", f"{date}_02-21-2022_overlap_ch{up_ch}_upstream", f"{date}_02-21-2022_repeat_overlap_ch{up_ch}_upstream"]
    channels_of_interest = [0, 1, 2]
    upstream_channel = up_ch
    dataset_threshold_mode = "by_channel"                                                                                       # Choices are by_channel, all_channel, None
    combine_all_dataset = False

    puncta_pixel_threshold = dict()
    if combine_all_dataset and dataset_threshold_mode == "all_channel":
        all_ch_threshold = manual_dataset_threshold(manual_label_file_path_list, channels_of_interest)
        print("Combined all channel threshold is:", all_ch_threshold)
        for channel_of_interest in channels_of_interest:
            puncta_pixel_threshold[channel_of_interest] = all_ch_threshold
        dataset_threshold_mode = "given"
    elif combine_all_dataset and dataset_threshold_mode == "by_channel":
        for channel_of_interest in channels_of_interest:
            puncta_pixel_threshold[channel_of_interest] = manual_dataset_threshold(manual_label_file_path_list, channel_of_interest)
            print(f"Combined threshold on all datasets for channel {channel_of_interest} is:", puncta_pixel_threshold[channel_of_interest])
        dataset_threshold_mode = "given"

    if not os.path.exists("manual_results"):
        os.mkdir("manual_results")
    for manual_label_file_path, label in zip(manual_label_file_path_list, labels):
        save_path = join("manual_results", label)
        manual_label_df = manual_process_data(manual_label_file_path, channels_of_interest, upstream_channel, puncta_pixel_threshold, pixel_threshold, dataset_threshold_mode)
        manual_label_df.to_csv(path_or_buf=f"{save_path}.csv", sep=",", index=False)
        manual_label_df.to_pickle(save_path)
        manual_print_result(manual_label_df, channels_of_interest)

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
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

detection_threshold = 0.8                                                                                               # The cutoff to ignore GUVs that the GUV detection algorithm is less confident about
folder_list = [".\\data\\01-11-22\\30_DOPS 69.5_ DOPC 0.5 _Atto\\200nM ALG2 A78C ESCRT1"]
label = "01-11-22_30DOPS_ALG2_ESCRT1_whole_dataset_all_ch_background_subtraction_0.8detection_0.2diam_on_03_09_22"
folder_list = [".\\data\\11-27-21\\DOPC_DOPS_10__Atto\\200 nM ALG2",
               ".\\data\\11-27-21\\DOPC_DOPS_30__Atto\\200 nM ALG2",
               ".\\data\\11-27-21\\DOPC_DOPS_50__Atto",
               ".\\data\\01-11-22\\10_DOPS 89.5_DOPC 0.5_Atto\\200nM ALG2",
               ".\\data\\01-11-22\\30_DOPS 69.5_ DOPC 0.5 _Atto\\200nM ALG2 A78C",
               ".\\data\\01-11-22\\50_DOPS 49.5_ DOPC 0.5_Atto\\200nM ALG2",
               ".\\data\\03-02-2022\\49.5_ DOPC_50_ DOPS_0.5_ Atto 647\\200 nM Atto 488 ALG-2",
               ".\\data\\03-02-2022\\69.5_ DOPC_30_ DOPS_0.5_ Atto 647\\200 nM Atto 488 ALG-2",
               ".\\data\\03-03-2022\\89.5_ DOPC_10_DOPS_0.5_ Atto\\200 nM Atto 488 ALG2"]                               # Data folder(s); list all folders cotaining .tif images to be analyzed

label = "all_ALG-2_whole_dataset_all_ch_background_subtraction_0.8detection_0.2diam_on_03_09_22"                        # Name your output here
yolo_model_path = "06062021_best.pt"                                                                                    # Designate your yolo model path here
channels_of_interest = [0, 1]                                                                                           # Enter your protein channels (zero-indexing); if more than 1 channel is entered, result will also include colocalization analysis
lipid_channel = 2                                                                                                       # Enter the lipid channel (zero_indexing) for GUV recognition purposes
series_type = Z_Stack_Series

folder_list = [os.path.abspath(folder) for folder in folder_list]
if not os.path.exists("results"):
    os.mkdir("results")
save_path = join("results", label)
verbose = True
# Not in use
old_punctate = False
puncta_model_path = None                                                                                                # Designate your puncta model path here
puncta_model = joblib.load(puncta_model_path) if old_punctate else None
detection = False                                                                                                       # DON'T USE THIS! If True, then circle detection is used, which is an unfinished feature.
ML = False
frame_punctate = 8
num_bins = 24

path_list = []
# Get folders
for folder in folder_list:
    for file in os.listdir(folder):
        if file.endswith(".nd2"):
            path_list.append(join(folder, file + "-output"))

# Extract frames
for path in path_list:
  extract_image(path, lipid_channel)

if os.path.exists(".\\yolov5\\runs"):
    os.remove(".\\yolov5\\runs")
# Detect GUV using yolov5
for path in path_list:
  identify_img(path, yolo_model_path, detection_threshold)

# Analyze all GUVs
result = None
folder_index_count = 0
puncta_pixel_threshold = dict()

# for channel_of_interest in channels_of_interest:
#   puncta_pixel_threshold[channel_of_interest] = dataset_threshold_by_channel(path_list, channel_of_interest)
#   print(puncta_pixel_threshold[channel_of_interest])
all_ch_threshold = dataset_threshold(path_list, channels_of_interest)
print(all_ch_threshold)
for channel_of_interest in channels_of_interest:
  puncta_pixel_threshold[channel_of_interest] = all_ch_threshold

for path in path_list:
  result, folder_index_count = process_data(path, folder_index_count, result, num_bins, channels_of_interest, lipid_channel, series_type, puncta_model, old_punctate, frame_punctate, verbose, puncta_pixel_threshold)

# saves data as a .csv file
result.to_csv(path_or_buf=f"{save_path}.csv", sep=",", index=False)
result.to_pickle(save_path)
print_result(result, channels_of_interest)

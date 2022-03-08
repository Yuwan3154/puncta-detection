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

folder_list = [".\\data\\03-02-2022\\49.5_ DOPC_50_ DOPS_0.5_ Atto 647\\200 nM Atto 488 ALG-2",
               ".\\data\\03-02-2022\\69.5_ DOPC_30_ DOPS_0.5_ Atto 647\\200 nM Atto 488 ALG-2",
               ".\\data\\03-03-2022\\89.5_ DOPC_10_DOPS_0.5_ Atto\\200 nM Atto 488 ALG2"]                               # Data folder(s); list all folders cotaining .tif images to be analyzed
label = "03_05_22_whole_dataset_2.5_li_50_30_10_DOPC_ALG-2"                                                             # Name your output here
if not os.path.exists("results"):
    os.mkdir("results")
save_path = join("results", label)                                                                                      # Designate your save path here
channels_of_interest = [0, 1]                                                                                           # Enter your protein channels (zero-indexing); if more than 1 channel is entered, result will also include colocalization analysis
lipid_channel = 2                                                                                                       # Enter the lipid channel (zero_indexing) for GUV recognition purposes
series_type = Z_Stack_Series
verbose = True

# Not in use
old_punctate = False
puncta_model_path = "\\content\\drive\\MyDrive\\HurleyLab\\11_7_21_new_feat_wout_true_maxima_tiff_2_kmeans_cl_model.pkl"# Designate your puncta model path here
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

puncta_pixel_threshold = dict()
for ch_of_interest in channels_of_interest:
  puncta_pixel_threshold[ch_of_interest] = dataset_threshold(path_list, ch_of_interest)
  print(puncta_pixel_threshold[ch_of_interest])

# Analyze all GUVs
result = None
folder_index_count = 0
for path in path_list:
  result, folder_index_count = process_data(path, folder_index_count, result, num_bins, channels_of_interest, lipid_channel, series_type, puncta_model, old_punctate, frame_punctate, verbose, puncta_pixel_threshold)

# saves data as a .csv file
result.to_csv(path_or_buf=f"{save_path}.csv", sep=",", index=False)
result.to_pickle(save_path)
print_result(result)

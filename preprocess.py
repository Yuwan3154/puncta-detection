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

detection_threshold = 0.6                                                                                               # The cutoff to ignore GUVs that the GUV detection algorithm is less confident about
folder_list = [".\\data\\11-27-21\\DOPC_DOPS_10__Atto\\200 nM ALG2",
               ".\\data\\11-27-21\\DOPC_DOPS_30__Atto\\200 nM ALG2",
               ".\\data\\11-27-21\\DOPC_DOPS_50__Atto"]                                                                 # Data folder(s); list all folders cotaining .tif images to be analyzed
    # [".\data\10_DOPS 89.5_DOPC 0.5_Atto\200nM ALG2",
    #          ".\data\30_DOPS 69.5_ DOPC 0.5 _Atto\200nM ALG2 A78C",
    #          ".\data\30_DOPS 69.5_ DOPC 0.5 _Atto\200nM ALG2 A78C ESCRT1",
    #          ".\data\50_DOPS 49.5_ DOPC 0.5_Atto\200nM ALG2"]                                                         # Data folder(s); list all folders cotaining .tif images to be analyzed
yolo_model_path = "06062021_best.pt"                                                                                    # Designate your yolo model path here
lipid_channel = 2                                                                                                       # Enter the lipid channel (zero_indexing) for GUV recognition purposes
series_type = Z_Stack_Series

path_list = []
# Get folders
for folder in folder_list:
    for file in os.listdir(folder):
        if file.endswith(".nd2"):
            path_list.append(join(folder, file + "-output"))

# Extract frames
for path in path_list:
  extract_image(path, lipid_channel)

# Detect GUV using yolov5
for path in path_list:
  identify_img(path, yolo_model_path, detection_threshold)

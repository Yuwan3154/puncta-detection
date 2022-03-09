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

manual_label_fpath = "./manual_results/03_07_22_02-21-2022_repeat_all_z_stack.csv"
label = "03_07_22_02-21-2022_repeat_all_z_stack"
channels_of_interest = [0, 1, 2]

if not os.path.exists("manual_results"):
    os.mkdir("manual_results")
save_path = join("manual_results", label)

manual_label_df = pd.read_csv(manual_label_fpath)
manual_label_df = manual_process_data(manual_label_df, channels_of_interest)

manual_label_df.to_csv(path_or_buf=f"{save_path}.csv", sep=",", index=False)
manual_label_df.to_pickle(save_path)
manual_print_result(manual_label_df, channels_of_interest)

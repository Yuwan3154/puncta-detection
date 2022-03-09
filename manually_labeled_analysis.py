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

manual_label_fpath = ".\\data\\02-20-2022\\69.5% DOPC_30% DOPS_0.5% Atto\\200 nM Dark ALG2_100 nM Cy3 ALIX_10 nM CHMP4b_100 nM dark CHMP2A_100 nM dark CHMP3_100 nM LD 655 Vps4b\\Manual_label_Feb_20_2022 - manual_label_example.csv"      # Enter your manual label file address here
label = "03_07_22_02-20-2022_69.5% DOPC_30% DOPS_0.5% Atto_200 nM Dark ALG2_100 nM Cy3 ALIX_10 nM CHMP4b_100 nM dark CHMP2A_100 nM dark CHMP3_100 nM LD 655 Vps4b"
channels_of_interest = [0, 1, 2]

manual_label_df = pd.read_csv(manual_label_fpath)
manual_label_df = manual_process_data(manual_label_df, channels_of_interest)

manual_label_df.to_csv(path_or_buf=f"{save_path}.csv", sep=",", index=False)
manual_label_df.to_pickle(save_path)
manual_print_result(manual_label_df, channels_of_interest)

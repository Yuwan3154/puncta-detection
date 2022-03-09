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
if not os.path.exists("manual_results"):
    os.mkdir("manual_results")
save_path = join("manual_results", label)

puncta_pixel_threshold = dict()
for ch_of_interest in channels_of_interest:
  puncta_pixel_threshold[ch_of_interest] = None

local_file_path = []
for file_path in manual_label_df["file path"]:
  file_dir_decomp = file_path.split("/")
  tif_file_name = file_dir_decomp[-1]
  local_file_path.append("\\".join([".\\data", "\\".join(file_dir_decomp[5:8]), tif_file_name[:-4] + ".nd2-output", "(series 1).tif"]))
manual_label_df["file path"] = local_file_path

manual_coloc_result_cols = [f"colocalization ch{ch1} ch{ch2}" for ch1, ch2 in itertools.combinations(channels_of_interest, 2)] + [f"colocalization weight ch{ch1} ch{ch2}" for ch1, ch2 in itertools.combinations(channels_of_interest, 2)]
manual_label_df[manual_coloc_result_cols] = np.nan
manual_label_df["punctate frame"] = np.nan
manual_puncta_cols = []
max_frame = max(manual_label_df["num frame"])
for ch in channels_of_interest:
  manual_puncta_cols.extend([f"puncta {j} ch{ch}" for j in range(max_frame)])
manual_label_df[manual_puncta_cols] = None
column_dict = dict()
for col in manual_puncta_cols:
  column_dict[col] = object
manual_label_df.astype(column_dict, copy=False)
for file_path in manual_label_df["file path"].unique():
  cur_df = manual_label_df[manual_label_df["file path"] == file_path]
  cur_df_puncta_cols, cur_df_puncta_frames, cur_puncta_nums = [], np.zeros(len(cur_df), dtype=int), np.zeros(len(cur_df), dtype=int)
  all_frame_img = io.imread(file_path)
  for j in range(cur_df["num frame"].iloc[0]):
    if len(all_frame_img.shape) == 4:
      all_ch_img = all_frame_img[j, :, :, :]
    else:
      all_ch_img = all_frame_img
    for ch in channels_of_interest:
      cur_ch_img = preprocess_for_puncta(all_ch_img[:, :, ch], puncta_pixel_threshold[ch])
      i = 0
      for row in cur_df.index:
        x1, x2, y1, y2 = manual_label_position(cur_df.loc[row])
        try:
          cur_puncta = get_maxima(get_img_sec(cur_ch_img, x1, x2, y1, y2, None))
        except ValueError:
          cur_puncta = [0, [], []]
        manual_label_df.at[row, f"puncta {j} ch{ch}"] = cur_puncta
        if len(cur_puncta) > cur_puncta_nums[i]:
          cur_df_puncta_frames[i], cur_puncta_nums[i] = j, len(cur_puncta)
        i += 1
  manual_label_df.at[cur_df.index, "punctate frame"] = cur_df_puncta_frames

for ch in channels_of_interest:
  manual_label_df = new_punctate(manual_label_df, ch)
manual_label_df = manual_colocalization(manual_label_df, channels_of_interest)
manual_label_df = new_manual_colocalization(manual_label_df, channels_of_interest)

manual_label_df.to_csv(path_or_buf=f"{save_path}.csv", sep=",", index=False)
manual_label_df.to_pickle(save_path)
manual_print_result(manual_label_df, channels_of_interest)

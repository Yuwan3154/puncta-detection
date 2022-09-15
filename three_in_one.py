from puncta_detect_util import *
import numpy as np
import pandas as pd
import math
import itertools
# from scipy.signal import correlate2d
from skimage import io
# from skimage.filters import threshold_otsu, median, gaussian
from skimage.morphology import disk
# from skimage import measure
# from imutils import contours, grab_contours
# import scipy.ndimage.filters as filters
# from sklearn.preprocessing import StandardScaler, normalize
# from sklearn.mixture import GaussianMixture
import cv2 as cv
import joblib
import os
from os.path import join
import shutil
import subprocess
# from sklearn.decomposition import PCA
import scipy.ndimage as ndimage
# from scipy.optimize import curve_fit
import argparse

def process_dir(exp_dir, channels_of_interest, detect_channel, meta_label, detect_threshold, pixel_threshold, zstack, detail):
  # folder_list = [".\\data\\01-11-22\\30_DOPS 69.5_ DOPC 0.5 _Atto\\200nM ALG2 A78C ESCRT1"]
  # label = f"combined_threshold_{pixel_threshold}_pixel_March_ALG2_on_05_26_22"
  folder_list = exp_dir
  label = f"{meta_label}_{exp_dir[0]}".replace("/", "_") # Name your output here
  save_path = join("results", label)
  if os.path.exists(save_path): # Short circuits if result is already available
    return pd.read_pickle(save_path)
  yolo_model_path = os.path.abspath("06062021_best.pt") # Designate your yolo model path here
  # channels_of_interest = [0, 1] # Enter your protein channels (zero-indexing); if more than 1 channel is entered, result will also include colocalization analysis
  # lipid_channel = 2 # Enter the lipid channel (zero_indexing) for GUV recognition purposes
  lipid_channel = detect_channel
  frame_quality = False
  verbose = True
  if zstack:
    series_type = Z_Stack_Series
  else:
    series_type = Series

  folder_list = [os.path.abspath(folder) for folder in folder_list]
  if not os.path.exists("results"):
      os.mkdir("results")

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

  if os.path.exists("yolov5/runs"):
     print("Removing old detection!")
     shutil.rmtree("yolov5/runs", ignore_errors=False)
  # Detect GUV using yolov5
  for path in path_list:
    identify_img(path, yolo_model_path, detect_threshold)

  # Analyze all GUVs
  result = None
  folder_index_count = 0
  puncta_pixel_threshold = dict()

  for channel_of_interest in channels_of_interest:
    puncta_pixel_threshold[channel_of_interest] = dataset_threshold(path_list, channel_of_interest)
    print(f"Combined threshold on all datasets for channel {channel_of_interest} is:", puncta_pixel_threshold[channel_of_interest])
  # all_ch_threshold = dataset_threshold(path_list, channels_of_interest)
  # print(all_ch_threshold)
  # for channel_of_interest in channels_of_interest:
  #   puncta_pixel_threshold[channel_of_interest] = all_ch_threshold
  # for channel_of_interest in channels_of_interest:
  #   puncta_pixel_threshold[channel_of_interest] = None

  for path in path_list:
    result, folder_index_count = process_data(path, folder_index_count, result, num_bins, channels_of_interest, lipid_channel, series_type, puncta_model, old_punctate, frame_punctate, verbose, puncta_pixel_threshold, pixel_threshold)

  # saves data as a .csv file
  result.to_csv(path_or_buf=f"{save_path}.csv", sep=",", index=False)
  result.to_pickle(save_path)
  print_result(result, channels_of_interest, detail, frame_quality)
  return result

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process GUV punctateness.')
  parser.add_argument("--folder", metavar="Data folder", type=str, nargs=1,
                      help="the data folder containing .nd2 files and its output")
  parser.add_argument("--label", metavar="Detail CSV label", type=str, nargs=1,
                      help="the prefix labels for the result of the individual folders.")
  parser.add_argument("--detect-threshold", metavar="GUV Detection Threshold", type=float, nargs=1,
                      help="the minimum number of pixels in a group to call puncta.")
  parser.add_argument("--detail", metavar="Verbose", type=bool, const=True, default=False, nargs="?")
  parser.add_argument("--zstack", metavar="Type of Series", type=bool, const=True, default=False, nargs="?",
                      help="Type of series for the images taken; flag if the input is Z-stack")
  args = vars(parser.parse_args())
  print("Arguments", args)
  folder, meta_label, detect_threshold, detail, zstack = args["file"][0], args["label"], args["detect_threshold"][0], args["detail"], args["zstack"]
  frame_quality, square_quality = True, True
  pixel_thresholds = [100] # TODO: temporary solution
  for i in range(len(pixel_thresholds)):
    summary_file = os.path.sep.join("result", f"{meta_label}_{folder.replace(os.path.sep, '_')}")
    print(f"Starting on {summary_file}")
    cur_result_df = process_dir([exp_dir], channels_of_interest, detect_channel, meta_label, pixel_threshold, zstack,
                                detail)
    extract_summary(summary_df, cur_result_df, channels_of_interest, index, frame_quality, square_quality)
    summary_df.to_csv(summary_file, index=False)

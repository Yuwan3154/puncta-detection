import torch
import numpy as np
import pandas as pd
import math
import itertools
from scipy.signal import correlate2d
from skimage import io
from skimage.filters import threshold_otsu, threshold_li, median, gaussian
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

def extract_image(imfolder, lipid_channel):
  file_list = []
  for i in os.listdir(imfolder):
    if i.endswith(".tif"):
      file_list.append(i)

  for i in range(len(file_list)):
    file_name = file_list[i]
    file_path = join(imfolder, file_name)
    series_name = file_name[:-4]
    series_folder = join(imfolder, series_name)
    if not os.path.exists(series_folder):
      os.mkdir(series_folder)
    print(file_name)
    im = io.imread(file_path)
    try:
      lipid = im[:, :, :, lipid_channel]
      for j in range(len(lipid)):
            frame_file_name = os.path.join(series_folder, "{}.jpeg".format(j))
            io.imsave(file_name=frame_file_name, arr=lipid[j, :, :])
    except IndexError:
      lipid = im[:, :, lipid_channel]
      frame_file_name = os.path.join(series_folder, "0.jpeg")
      io.imsave(file_name=frame_file_name, arr=lipid)

def identify_img(imfolder, yolo_model_path, thresh=0.8):
    cwd = os.getcwd()
    os.chdir(".\\yolov5")
    file_list = []
    for i in os.listdir(imfolder):
        if i.endswith(".tif"):
            file_list.append(i)

    for x in file_list:
        img_sz = (cv.fastNlMeansDenoising(cv.imread(join(imfolder, x), 0), h=1.5)).shape[1]
        series_folder = join(imfolder, x[:-4])
        # os.system("python detect.py --weights /content/yolov5/best.pt --img 416 --conf 0.6 --source " + series_folder + " --save-txt")
        try:
            subprocess.check_output(
                ["python", "detect.py", "--weights", yolo_model_path, "--img", str(img_sz), "--conf",
                 str(thresh), "--source", f"{series_folder}", "--save-txt"])
        except subprocess.CalledProcessError as e:
            print(e.output.decode('UTF-8'))
    os.chdir(cwd)

def process_data(imfolder, folder_index_count, result, num_bins, channels_of_interest, lipid_ch, series_type, puncta_model, old_punctate, frame_punctate, verbose, puncta_pixel_threshold):
    file_list = []
    for f in os.listdir(imfolder):
        if f.endswith(".tif"):
            file_list.append(f)
    for i in range(len(file_list)):
        file_name = file_list[i]
        file_path = join(imfolder, file_name)
        if verbose:
            print(f"Starting on {file_path}.")
        im = io.imread(file_path)
        signal_df_lst = []

        # calculates the output dataframe for each channel for the current file
        for channel_of_interest in channels_of_interest:
            if len(im.shape) == 4:
                ch = np.array(im[:, :, :, channel_of_interest])
                lipid = np.array(im[:, :, :, lipid_ch])
            elif len(im.shape) == 3:
                ch = np.array(im[:, :, channel_of_interest]).reshape(1, im.shape[0], im.shape[1])
                lipid = np.array(im[:, :, lipid_ch]).reshape(1, im.shape[0], im.shape[1])
            else:
                raise ValueError("Input image should have 3 or 4 channels.")
            size = (5, 5)
            label_folder = ".\\yolov5\\runs\\detect\\exp{}".format(folder_index(folder_index_count))
            # defines local maxima in the lipid channel
            num_frame = len(lipid)
            cur_series = series_type(ch, lipid, imfolder, file_name, puncta_model, num_bins, num_frame, old_punctate, frame_punctate, puncta_pixel_threshold[channel_of_interest])
            for j in range(num_frame):
                label_file_name = os.path.join(label_folder, "labels\\{}.txt".format(j))
                try:
                    labels = pd.read_csv(label_file_name, sep=" ", header=None)
                    labels = np.array(labels)
                    # excluding all GUVs classified as low_quality as
                    processed_labels = [label[1:] for label in labels if label[0] == 0]
                    # processed_labels = labels[:, 1:]
                except FileNotFoundError:
                    print(f"No GUVs are detected in {file_path} frame {j}!")
                    processed_labels = []
                cur_series.process_frame(processed_labels)

            # saves each feature as a column of a DataFrame
            result_cols = cur_series.get_col_names(channel_of_interest)
            signal = cur_series.get_values()
            signal_df = pd.DataFrame(data=signal, columns=result_cols)
            if not old_punctate:
                signal_df = signal_df.drop(columns=f"punctateness ch{channel_of_interest}")

            temp = np.sum(np.array(signal_df[[f"value {i} ch{channel_of_interest}" for i in range(num_frame)]].isnull()),
                          axis=1)
            signal_df[f"quality ch{channel_of_interest}"] = temp / num_frame <= (1 - 6 / num_frame)
            signal_df["num frame"] = num_frame
            signal_df = new_punctate(signal_df, channel_of_interest)
            signal_df_lst.append(signal_df)

        # deals with when multiple channels of interest are present
        final_df = signal_df_lst[0].sort_values(by="x 0").reset_index(drop=True)
        if len(channels_of_interest) > 1:
            for cur_df in signal_df_lst[1:]:
                cur_df = cur_df.sort_values(by="x 0").reset_index(drop=True)
                cols_to_use = cur_df.columns.difference(final_df.columns).to_list()  # https://stackoverflow.com/questions/19125091/pandas-merge-how-to-avoid-duplicating-columns
                final_df = pd.DataFrame.merge(final_df, cur_df[cols_to_use], left_index=True, right_index=True, how='outer')
            final_df = colocalization(final_df, channels_of_interest)
            final_df = new_colocalization(final_df, im, channels_of_interest)

        if result is None:
            result = final_df
        else:
            result = pd.concat([result, final_df], axis=0, join="outer", ignore_index=True)

        # makes sure the correct label is read for the next image file
        folder_index_count += 1
    result = square_quality(result)
    return result, folder_index_count

def manual_process_data(manual_label_df_file_path, channels_of_interest, upstream_channel, puncta_pixel_threshold, dataset_threshold_mode="by_channel"):
    manual_label_df = pd.read_csv(manual_label_df_file_path)
    manual_label_df = manual_label_df[["file path", "image size", "num frame", "top left x", "top left y", "bottom right x", "bottom right y"]]
    if dataset_threshold_mode == "given":
        pass
    elif dataset_threshold_mode == "by_channel":
        puncta_pixel_threshold = dict()
        for channel_of_interest in channels_of_interest:
            puncta_pixel_threshold[channel_of_interest] = manual_dataset_threshold([manual_label_df_file_path], channel_of_interest)
            print(f"Threshold for channel {channel_of_interest} is:", puncta_pixel_threshold[channel_of_interest])
    elif dataset_threshold_mode == "all_ch":
        puncta_pixel_threshold = dict()
        all_ch_threshold = manual_dataset_threshold([manual_label_df_file_path], channels_of_interest)
        print("All channel threshold is:", all_ch_threshold)
        for channel_of_interest in channels_of_interest:
            puncta_pixel_threshold[channel_of_interest] = all_ch_threshold
    else:
        puncta_pixel_threshold = dict()
        for channel_of_interest in channels_of_interest:
            puncta_pixel_threshold[channel_of_interest] = None

    manual_coloc_result_cols = [f"colocalization ch{ch1} ch{ch2}" for ch1, ch2 in
                                itertools.combinations(channels_of_interest, 2)] + [
                                   f"colocalization weight ch{ch1} ch{ch2}" for ch1, ch2 in
                                   itertools.combinations(channels_of_interest, 2)]
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
        print("Starting on", file_path)
        cur_df = manual_label_df[manual_label_df["file path"] == file_path]
        cur_df_puncta_frames, cur_puncta_nums = np.zeros(len(cur_df), dtype=int), np.zeros(len(cur_df), dtype=int)
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
                        cur_puncta = Puncta(0, [], [])
                    manual_label_df.at[row, f"puncta {j} ch{ch}"] = cur_puncta
                    if cur_puncta.get_num_puncta() > cur_puncta_nums[i]:
                        cur_df_puncta_frames[i], cur_puncta_nums[i] = j, cur_puncta.get_num_puncta()
                    i += 1
        manual_label_df.at[cur_df.index, "punctate frame"] = cur_df_puncta_frames

    for ch in channels_of_interest:
        manual_label_df = new_punctate(manual_label_df, ch)
    manual_label_df = manual_colocalization(manual_label_df, channels_of_interest, upstream_channel)
    manual_label_df = new_manual_colocalization(manual_label_df, channels_of_interest, upstream_channel, puncta_pixel_threshold)
    return manual_label_df

def print_result(result, channels_of_interest):
    result = result[result["square quality"]]
    result["temp folder"] = list(map(lambda f: os.path.sep.join(f.split(os.path.sep)[:-2]), result["folder"]))

    print("2-channel colocalization")
    # 2-channel colocalization
    for folder in np.unique(result["folder"]):
        cur_folder_result = []
        for file_name in np.unique((result[result["folder"] == folder])["file name"]):
            cur_file_df = result[np.array(result["folder"] == folder) * np.array(result["file name"] == file_name)]
            for ch1, ch2 in itertools.combinations(channels_of_interest, 2):
                cur_chs_df = cur_file_df[cur_file_df[f"colocalization weight ch{ch1} ch{ch2}"].notna()]
                cur_percent = np.mean(cur_chs_df[f"colocalization ch{ch1} ch{ch2}"])
                cur_folder_result.extend([cur_percent] * len(cur_chs_df))
                print(f"The percent colocalization for {folder}{file_name} between channels {ch1} and {ch2} is {cur_percent}.")
        print()
        print(f"The percent colocalization for {folder} between channels {ch1} and {ch2} is {np.mean(cur_folder_result)}.")
        print()

    for ch1, ch2 in itertools.combinations(channels_of_interest, 2):
        print()
        ch_percent = np.mean(result[result[f"colocalization weight ch{ch1} ch{ch2}"].notna()][f"colocalization ch{ch1} ch{ch2}"])
        print(f"The overall percent colocalization between channels {ch1} and {ch2} is {ch_percent}.")
        print()

    print("2-channel new colocalization")
    # 2-channel new colocalization
    for folder in np.unique(result["folder"]):
        cur_folder_result = []
        for file_name in np.unique((result[result["folder"] == folder])["file name"]):
            cur_file_df = result[np.array(result["folder"] == folder) * np.array(result["file name"] == file_name)]
            for ch1, ch2 in itertools.combinations(channels_of_interest, 2):
                cur_chs_df = cur_file_df[cur_file_df[f"colocalization weight ch{ch1} ch{ch2}"].notna()]
                cur_percent = np.mean(cur_chs_df[f"new colocalization ch{ch1} ch{ch2}"])
                cur_folder_result.extend([cur_percent] * len(cur_chs_df))
                print(f"The percent new colocalization for {folder}{file_name} between channels {ch1} and {ch2} is {cur_percent}.")
        print()
        print(f"The percent new colocalization for {folder} between channels {ch1} and {ch2} is {np.mean(cur_folder_result)}.")
        print()

    for ch1, ch2 in itertools.combinations(channels_of_interest, 2):
        print()
        ch_percent = np.mean(result[result[f"colocalization weight ch{ch1} ch{ch2}"].notna()][f"new colocalization ch{ch1} ch{ch2}"])
        print(f"The overall percent new colocalization between channels {ch1} and {ch2} is {ch_percent}.")
        print()

    print("channel punctateness")
    # channel punctateness
    for folder in np.unique(result["folder"]):
      cur_folder_result = [[] for _ in range(len(channels_of_interest))]
      for file_name in np.unique((result[result["folder"] == folder])["file name"]):
        cur_file_df = result[np.array(result["folder"] == folder) * np.array(result["file name"] == file_name)]
        for ch in channels_of_interest:
          cur_percent = np.mean(cur_file_df[cur_file_df[f"quality ch{0}"]][f"new punctate ch{ch}"])
          cur_folder_result[ch].extend([cur_percent] * len(cur_file_df))
          print(f"The percent punctateness for {folder}{file_name} in channel {ch} is {cur_percent}.")
      for ch in channels_of_interest:
        print()
        print(f"The percent punctateness for {folder} in channel {ch} is {np.mean(cur_folder_result[ch])}.")

    for temp_folder in np.unique(result["temp folder"]):
        cur_folder_df = result[result["temp folder"] == temp_folder]
        for ch in channels_of_interest:
            print()
            ch_percent = np.mean(cur_folder_df[f"new punctate ch{ch}"])
            print(f"The overall percent punctateness for {temp_folder} in channel {ch} is {ch_percent}.")
            print()

    for ch in channels_of_interest:
        print()
        ch_percent = np.mean(result[f"new punctate ch{ch}"])
        print(f"The overall percent punctateness in channel {ch} is {ch_percent}.")
        print()

def manual_print_result(manual_label_df, channels_of_interest):
    print("2-channel colocalization")
    # 2-channel colocalization
    for file_path in np.unique(manual_label_df["file path"]):
        cur_file_df = manual_label_df[np.array(manual_label_df["file path"] == file_path)]
        for ch1, ch2 in itertools.combinations(channels_of_interest, 2):
            cur_chs_df = cur_file_df[
                np.array(cur_file_df[f"new punctate ch{ch1}"]) * np.array(cur_file_df[f"new punctate ch{ch2}"])]
            cur_chs_df = cur_chs_df[cur_chs_df[f"colocalization weight ch{ch1} ch{ch2}"].notna()]
            cur_percent = np.mean(cur_chs_df[f"colocalization ch{ch1} ch{ch2}"])
            print(f"The percent colocalization for {file_path} between channels {ch1} and {ch2} is {cur_percent}.")
    for ch1, ch2 in itertools.combinations(channels_of_interest, 2):
        ch_percent = np.mean(manual_label_df[f"colocalization ch{ch1} ch{ch2}"])
        print(f"The overall percent colocalization between channels {ch1} and {ch2} is {ch_percent}.")

    print("2-channel new colocalization")
    # 2-channel new colocalization
    for file_path in np.unique(manual_label_df["file path"]):
        cur_file_df = manual_label_df[np.array(manual_label_df["file path"] == file_path)]
        for ch1, ch2 in itertools.combinations(channels_of_interest, 2):
            cur_chs_df = cur_file_df[
                np.array(cur_file_df[f"new punctate ch{ch1}"]) * np.array(cur_file_df[f"new punctate ch{ch2}"])]
            cur_chs_df = cur_chs_df[cur_chs_df[f"colocalization weight ch{ch1} ch{ch2}"].notna()]
            cur_percent = np.mean(cur_chs_df[f"new colocalization ch{ch1} ch{ch2}"])
            print(f"The percent new colocalization for {file_path} between channels {ch1} and {ch2} is {cur_percent}.")
    for ch1, ch2 in itertools.combinations(channels_of_interest, 2):
        ch_percent = np.mean(manual_label_df[f"new colocalization ch{ch1} ch{ch2}"])
        print(f"The overall percent new colocalization between channels {ch1} and {ch2} is {ch_percent}.")

    print("3-channel colocalization")
    # 3-channel colocalization
    for file_path in np.unique(manual_label_df["file path"]):
        cur_file_df = manual_label_df[np.array(manual_label_df["file path"] == file_path)]
        for ch1, ch2, ch3 in itertools.combinations(channels_of_interest, 3):
            cur_chs_df = cur_file_df[np.array(cur_file_df[f"new punctate ch{ch1}"]) * np.array(
                cur_file_df[f"new punctate ch{ch2}"]) * np.array(cur_file_df[f"new punctate ch{ch3}"])]
            cur_chs_df = cur_chs_df[cur_chs_df[f"colocalization weight ch{ch1} ch{ch2} ch{ch3}"].notna()]
            cur_percent = np.mean(cur_chs_df[f"colocalization ch{ch1} ch{ch2} ch{ch3}"])
            print(f"The percent colocalization for {file_path} between channels {ch1}, {ch2} and {ch3} is {cur_percent}.")
    for ch1, ch2, ch3 in itertools.combinations(channels_of_interest, 3):
        ch_percent = np.mean(manual_label_df[f"colocalization ch{ch1} ch{ch2} ch{ch3}"])
        print(f"The overall percent colocalization between channels {ch1}, {ch2} and {ch3} is {ch_percent}.")

    # 3-channel new colocalization
    print("3-channel new colocalization")
    for file_path in np.unique(manual_label_df["file path"]):
        cur_file_df = manual_label_df[np.array(manual_label_df["file path"] == file_path)]
        for ch1, ch2, ch3 in itertools.combinations(channels_of_interest, 3):
            cur_chs_df = cur_file_df[np.array(cur_file_df[f"new punctate ch{ch1}"]) * np.array(
                cur_file_df[f"new punctate ch{ch2}"]) * np.array(cur_file_df[f"new punctate ch{ch3}"])]
            cur_chs_df = cur_chs_df[cur_chs_df[f"colocalization weight ch{ch1} ch{ch2} ch{ch3}"].notna()]
            cur_percent = np.mean(cur_chs_df[f"new colocalization ch{ch1} ch{ch2} ch{ch3}"])
            print(f"The percent new colocalization for {file_path} between channels {ch1}, {ch2} and {ch3} is {cur_percent}.")
    for ch1, ch2, ch3 in itertools.combinations(channels_of_interest, 3):
        ch_percent = np.mean(manual_label_df[f"new colocalization ch{ch1} ch{ch2} ch{ch3}"])
        print(f"The overall percent new colocalization between channels {ch1}, {ch2} and {ch3} is {ch_percent}.")

def folder_index(x):
    """
    The output of detect.py is put in directory of etc/runs/exp{}, where the number is determined by the order at which detect.py is run.
    """
    return str(x + 1) if x != 0 else ""

def get_coord(xywh, img_sz=None):
    """
    The format of xywh as output of detect.py records the coordinates of the center of the boxes and the width and height.
    """
    x, y, half_w, half_h = xywh[0], xywh[1], xywh[2] / 2, xywh[3] / 2
    x1, y1, x2, y2 = x - half_w, y - half_h, x + half_w, y + half_h
    if img_sz is not None:
        return max(0, round(x1 * img_sz)), min(img_sz, round(x2 * img_sz)), max(0, round(y1 * img_sz)), min(img_sz,round(y2 * img_sz))
    else:
        return int(x1), int(x2), int(y1), int(y2)

def adjust_brightness_contrast(img):
    """
    A generally good adjustment of brightness and contrast, where multiplicative factor is contrast and additive factor is brightness.
    """
    return img * 10 + 20000

def export_index(i):
    """
    Generate a four digit string version of input integer.
    """
    assert type(i) == int
    if i < 10:
        return "000" + str(i)
    elif i < 100:
        return "00" + str(i)
    elif i < 1000:
        return "0" + str(i)
    else:
        return str(i)

def similar_width_height(w, h):
    """
    Check that the given dimension is roughly square i.e., the image
    is not on the edge or blocked.
    """
    if min(w, h) / max(w, h) > 0.9:
        return True
    else:
        return False

def square_quality(df):
    quality_result = []
    for i in range(len(df)):
        init_frame = df["initial frame"][i]
        quality_result.append(similar_width_height(df[f"w {init_frame}"][i], df[f"h {init_frame}"][i]))
    df["square quality"] = quality_result
    return df

def quantify(xywh, c1, lipid, f, img_sz, detection=False):
    """
    Takes 5 parameters: xywh, c1, lipid, f, detection;
    returns a number representing the intensity of the image.
    """
    size = (3, 3)
    x1, x2, y1, y2 = get_coord(xywh, img_sz)
    if x1 == x2 or y1 == y2:
        return 0
    c1_sec = get_img_sec(c1, x1, x2, y1, y2, f)
    lipid_sec = get_img_sec(c1, x1, x2, y1, y2, f)
    im_max = filters.maximum_filter(lipid_sec, size)
    im_min = filters.minimum_filter(lipid_sec, size)
    im_diff = im_max - im_min
    try:
        thresh = threshold_otsu(im_diff)
    except ValueError:
        thresh = 0
    thresh = thresh * 0.45
    bool_diff = (im_diff <= thresh)

    masked_c1 = c1_sec.copy()
    masked_c1[bool_diff] = 0
    c1av = np.average(masked_c1[masked_c1 != 0])
    backgroundout = np.average(c1_sec[masked_c1 == 0])
    insidepix = [(y2 - y1) // 2, (x2 - x1) // 2]
    selem = disk(5)
    c1blur = median(c1_sec, selem=selem)
    backgroundin = c1blur[insidepix[0], insidepix[1]]
    background = (backgroundout + backgroundin) / 2

    if detection:
        Hough_success = False
        try:
            min_r = min(round(xywh[2] / 4),
                        round(xywh[3] / 4))  # Because the largest circle takes up most of the space of the box
            io.imsave("/content/yolov5/temp.png", lipid_sec)

            """im = Image.open("/content/yolov5/temp.png")
            print(0)
            enhancer = ImageEnhance.Contrast(im)
            factor = 3.0                             # increase contrast
            print(1)
            im_output = enhancer.enhance(factor)
            print(2)
            im_output.save("/content/yolov5/temp.png")"""

            img = cv.imread("/content/yolov5/temp.png", 0)
            img = 5 * img + 50  # increase brightness and contrast
            cv2_imshow(img)
            circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT_ALT, 1.5, 1, param1=200, param2=0.9, minRadius=min_r,
                                      maxRadius=0)
            if verbose:
                print("The dimensions are: {} {}".format(xywh[2] * img_sz, xywh[3] * img_sz))
                print("The Hough circles are: {}".format(circles))
                print(circles)
            largest_circle = get_largest(circles)
            c_x, c_y = largest_circle[0], largest_circle[1]
            inner_r, outer_r = round(largest_circle[2] * 0.9), round(largest_circle[2] * 1.1)
            Hough_success = True
            if verbose:
                print("Hough succeed!")
                print("The Hough radius is: {}".format(largest_circle[2]))
        except TypeError:
            pass
        try:
            img = 5 * lipid_sec + 50
            center_radius = get_center_radius(img, thresh)
            if Hough_success and center_radius[2] > largest_circle[2]:
                c_x, c_y = center_radius[0], center_radius[1]
                inner_r, outer_r = round(center_radius[2] * 0.9), round(center_radius[2] * 1.1)
            if verbose:
                print("me succeed!")
                print("The me radius is: {}".format(center_radius[2]))
        except IndexError:
            pass
        try:
            xy = np.mgrid[0:x:1, 0:y:1].reshape(2,
                                                -1).T  # src: https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy
            c1av = np.average([c1_sec[y, x] for (x, y) in xy if inner_r < (x - c_x) ** 2 + (y - c_y) ** 2 < outer_r])
        except NameError:
            pass
    return c1av - background

def get_img_sec(img, x1, x2, y1, y2, f):
    """
    Given a 2D/3D image, return the designated section of the image,
    assuming that it follows the shape ([frame, ]hieght, width).
    """
    if len(img.shape) == 3:
        return img[f, y1:y2, x1:x2]
    else:
        return img[y1:y2, x1:x2]

class Guv:
    """
    A class storing the positional information of a GUV across time series.
    """

    def __init__(self, first_xywh, initial_frame, folder, file_name, img_sz, model, num_bins, old_punctate, frame_punctate, puncta_pixel_threshold):
        self.xs, self.ys, self.ws, self.hs = [], [], [], []
        self.values = []
        self.candidate = first_xywh
        self.init_f = initial_frame
        self.adj = False
        self.punctate = 0 if old_punctate else np.nan
        self.folder = folder
        self.file_name = file_name
        self.img_sz = img_sz
        self.model = model
        self.num_bins = num_bins
        self.puncta_param = []
        self.old_punctate = old_punctate
        self.frame_punctate = frame_punctate
        self.puncta_pixel_threshold = puncta_pixel_threshold

    def update_pos(self, xywh):
        self.x = xywh[0]
        self.y = xywh[1]
        self.w = xywh[2]
        self.h = xywh[3]
        self.xs.append(xywh[0])
        self.ys.append(xywh[1])
        self.ws.append(xywh[2])
        self.hs.append(xywh[3])

    def similar_size(self, xywh):
        """return a boolean indicating if the size of the two GUVs are close enough by a threshold"""
        score = abs(self.w - xywh[2]) + abs(self.h - xywh[3])
        dim_threshold = 10
        return score < dim_threshold

    def distance_to_coord(self, xywh):
        """return a score representing the closeness of two GUVs; a smaller number implies closer"""
        score = (self.x - xywh[0]) ** 2 + (self.y - xywh[1]) ** 2
        return score

    def distance_to_guv(self, guv):
        dist = np.sqrt(sum((self.get_pos() - guv.get_pos()) ** 2))
        return dist

    def add_value(self, ch_mask, ch, lipid, f):
        if self.candidate is None:
            self.values.append(np.nan)  # TBD!!!!!
            self.update_pos([np.nan] * 4)
            self.puncta_param.append(Puncta(0, [], []))
        else:
            if self.old_punctate and f < self.frame_punctate:
                score = int(predict(self.candidate, ch, f, self.model, self.img_sz, self.num_bins))
                self.punctate += score
            x1, x2, y1, y2 = get_coord(self.candidate, self.img_sz)
            self.puncta_param.append(get_maxima(get_img_sec(ch_mask, x1, x2, y1, y2, f)))
            self.update_pos(self.candidate)
            self.values.append(quantify(self.candidate, ch, lipid, f, self.img_sz, False))
            self.candidate = None

    def get_values(self):
        value_empty = [np.nan for _ in range(self.init_f)]
        position_empty = [np.nan for _ in range(self.init_f)]
        puncta_param_empty = [Puncta(0, [], []) for _ in range(self.init_f)]
        return [self.folder, self.file_name, self.img_sz, self.get_averaged_punctateness(), self.adj,
                self.init_f] + value_empty + self.values + puncta_param_empty + self.puncta_param + position_empty + self.xs + position_empty + self.ys + position_empty + self.ws + position_empty + self.hs

    def get_averaged_punctateness(self):
        if np.isnan(self.punctate):
            return np.nan
        else:
            return round(self.punctate / self.frame_punctate)

    def get_pos(self):
        return np.array((self.xs[-1], self.ys[-1]))

    def check_adj(guvs):
        guv1, guv2 = guvs[0], guvs[1]
        if guv1.adj and guv2.adj:
            return
        g1_coord, g2_coord = get_coord(guv1.xywh), get_coord(guv2.xywh)
        l1, l2 = [g1_coord[0], g1_coord[2]], [g1_coord[1], g1_coord[3]]
        r1, r2 = [g2_coord[0], g2_coord[2]], [g2_coord[1], g2_coord[3]]

        # If one rectangle is on left side of other            @src https://www.geeksforgeeks.org/find-two-rectangles-overlap/
        if (l1[0] >= r2[0] or l2[0] >= r1[0]):
            return

        # If one rectangle is above other
        if (r1[1] >= l2[1] or r2[1] >= l1[1]):
            return
        guv1.adj, guv2.adj = True, True

class Z_Stack_Guv(Guv):

    def __init__(self, first_xywh, initial_frame, folder, file_name, img_sz, model, num_bins, old_punctate, frame_punctate, puncta_pixel_threshold):
        Guv.__init__(self, first_xywh, initial_frame, folder, file_name, img_sz, model, num_bins, old_punctate, frame_punctate, puncta_pixel_threshold)
        self.equa_frame = None
        self.equa_size = 0

    def add_value(self, ch_mask, ch, lipid, f):
        if self.candidate is None:
            self.values.append(np.nan)  # TBD!!!!!
            self.update_pos([np.nan] * 4)
            self.puncta_param.append(Puncta(0, [], []))
        else:
            if self.old_punctate and f < self.frame_punctate:
                score = int(predict(self.candidate, ch, f, self.model, self.img_sz, self.num_bins, self.old_punctate))
                self.punctate += score
            x1, x2, y1, y2 = get_coord(self.candidate, self.img_sz)
            self.puncta_param.append(get_maxima(get_img_sec(ch_mask, x1, x2, y1, y2, f)))
            self.update_pos(self.candidate)
            self.update_equa(self.candidate, f)
            self.values.append(quantify(self.candidate, ch, lipid, f, self.img_sz, False))
            self.candidate = None

    def update_equa(self, guv, f):
        if self.equa_size <= max(guv[2], guv[3]):
            self.equa_frame = f
            self.equa_size = max(guv[2], guv[3])

    def get_values(self):
        value_empty = [np.nan for _ in range(self.init_f)]
        position_empty = [np.nan for _ in range(self.init_f)]
        puncta_param_empty = [Puncta(0, [], []) for _ in range(self.init_f)]
        return [self.folder, self.file_name, self.img_sz, self.get_averaged_punctateness(), self.adj, self.init_f,
                self.equa_frame,
                self.equa_size] + value_empty + self.values + puncta_param_empty + self.puncta_param + position_empty + self.xs + position_empty + self.ys + position_empty + self.ws + position_empty + self.hs

class Series:
    """A class storing the GUVs from the same series."""

    def __init__(self, ch, lipid, folder, file_name, model, num_bins, num_frame, old_punctate, frame_punctate, puncta_pixel_threshold):
        self.guv_lst = []
        self.frame = 0
        self.ch = ch
        self.lipid = lipid
        self.folder = folder
        self.file_name = file_name
        self.img_sz = ch.shape[1]
        self.model = model
        self.num_bins = num_bins
        self.num_frame = num_frame
        self.old_punctate = old_punctate
        self.frame_punctate = frame_punctate
        self.puncta_pixel_threshold = puncta_pixel_threshold

    def process_frame(self, guvs):
        add_lst = self.match_all(guvs)
        self.guv_lst.extend(add_lst)
        self.add_value_all()
        self.check_adj_all()

    def add_value_all(self):
        for guv in self.guv_lst:
            ch = self.ch[self.frame, :, :]
            ch_mask = preprocess_for_puncta(ch[:, :], self.puncta_pixel_threshold)
            guv.add_value(ch_mask, ch, self.lipid, self.frame)
        self.frame += 1

    def match_all(self, guvs, num_trial=5):
        best_add_lst = None
        best_match_dic = None
        best_score = -1
        for _ in range(num_trial):
            cur_add_lst = []
            cur_score = 0
            cur_match_dic = {}
            shuffled = guvs.copy()
            np.random.shuffle(shuffled)
            for xywh in shuffled:
                best_match = None
                for guv in self.guv_lst:
                    if guv.similar_size(xywh):
                        if best_match is None or guv.distance_to_coord(xywh) < best_match.distance_to_coord(xywh):
                            best_match = guv
                if best_match is None:
                    cur_add_lst.append(
                        Guv(xywh, self.frame, self.folder, self.file_name, self.img_sz, self.model, self.num_bins,
                            self.old_punctate, self.puncta_pixel_threshold))
                elif cur_match_dic.get(best_match, None) is None:
                    cur_match_dic[best_match] = xywh
                    cur_score += best_match.distance_to_coord(xywh)
                else:
                    break
            if cur_score > best_score:
                best_match_dic = cur_match_dic
                best_add_lst = cur_add_lst
                best_score = cur_score

        for key, value in best_match_dic.items():
            key.candidate = value
        return best_add_lst

    def check_adj_all(self):
        pw_lst = itertools.combinations(self.guv_lst, 2)
        map(Guv.check_adj, pw_lst)

    def get_values(self):
        return [guv.get_values() for guv in self.guv_lst]

    def get_col_names(self, ch):
        return ["folder", "file name", "image size", f"punctateness ch{ch}", "adjacent", "initial frame"] + [
            f"value {i} ch{ch}" for i in range(self.num_frame)] + [f"puncta {i} ch{ch}" for i in
                                                                   range(self.num_frame)] + [f"x {i}" for i in
                                                                                             range(self.num_frame)] + [
                   f"y {i}" for i in range(self.num_frame)] + [f"w {i}" for i in range(self.num_frame)] + [f"h {i}" for
                                                                                                           i in range(
                self.num_frame)]

class Z_Stack_Series(Series):

    def __init__(self, ch, lipid, folder, file_name, model, num_bins, num_frame, old_punctate, frame_punctate, puncta_pixel_threshold):
        Series.__init__(self, ch, lipid, folder, file_name, model, num_bins, num_frame, old_punctate, frame_punctate, puncta_pixel_threshold)
        self.dist_thresh = 10

    def match_all(self, guvs, num_trial=5):
        best_add_lst = None
        best_match_dic = None
        best_score = float("inf")
        for _ in range(num_trial):
            cur_add_lst = []
            cur_score = 0
            cur_match_dic = {}
            shuffled = guvs.copy()
            np.random.shuffle(shuffled)
            remaining_guv = self.guv_lst.copy()
            for xywh in shuffled:
                best_match = None
                for guv in remaining_guv:
                    if guv.distance_to_coord(xywh) <= self.dist_thresh:
                        if best_match is None or guv.distance_to_coord(xywh) < best_match.distance_to_coord(xywh):
                            best_match = guv
                if best_match is None:
                    cur_add_lst.append(
                        Z_Stack_Guv(xywh, self.frame, self.folder, self.file_name, self.img_sz, self.model, self.num_bins,
                                    self.old_punctate, self.frame_punctate, self.puncta_pixel_threshold))
                elif cur_match_dic.get(best_match, None) is None:
                    cur_match_dic[best_match] = xywh
                    cur_score += best_match.distance_to_coord(xywh)
                    remaining_guv.remove(best_match)
                else:
                    break
            if cur_score < best_score:
                best_match_dic = cur_match_dic
                best_add_lst = cur_add_lst
                best_score = cur_score

        for key, value in best_match_dic.items():
            key.candidate = value
        return best_add_lst

    def get_col_names(self, ch):
        return ["folder", "file name", "image size", f"punctateness ch{ch}", "adjacent", "initial frame",
                "equatorial frame", "equatorial size"] + [f"value {i} ch{ch}" for i in range(self.num_frame)] + [
                   f"puncta {i} ch{ch}" for i in range(self.num_frame)] + [f"x {i}" for i in range(self.num_frame)] + [
                   f"y {i}" for i in range(self.num_frame)] + [f"w {i}" for i in range(self.num_frame)] + [f"h {i}" for
                                                                                                           i in range(
                self.num_frame)]

class Puncta:
    def __init__(self, num_puncta, puncta_coords, puncta_bbox):
        self.num_puncta = num_puncta
        self.puncta_coords = tuple(puncta_coords)
        self.puncta_bbox = tuple(puncta_bbox)

    def get_num_puncta(self):
        return self.num_puncta

    def get_puncta_coords(self):
        return self.puncta_coords

    def get_puncta_bbox(self):
        return self.puncta_bbox

# Puncta Quant
def get_maxima(img):
    """Use processed image to pick up puncta information.
    Input: a 2D array
    Output: a list with 3 entries:
      1. number of punctum found
      2. the pixel value of the punctum found
      3. a list of coordinates of the puncta in the form of tuples (x, y).
    """
    labels = measure.label(img)
    center_of_img = np.array([img.shape[1], img.shape[0]]) / 2
    # cutoff = max(img.shape) / 4
    coords, bboxes = [], []
    for label_property in measure.regionprops(labels):
        if label_property.area >= 5:
            cur_center = (int(label_property.centroid[1]), int(label_property.centroid[0]))
            # if sum(np.sqrt((cur_center - center_of_img)**2)) > cutoff:
            coords.append(cur_center)
            bboxes.append(label_property.bbox)
    # components = best_gaussian_mixture(np.array(coords), max(img.shape))
    return Puncta(len(coords), coords, bboxes)

def best_gaussian_mixture(data, diam):
    """
    Given a list of coordinates of candidate punctum, attempt to
    combine them into real punctum. The output is also a list of
    coordinates. All input and output has the form of (x, y).
    """
    if len(data) <= 1:
        return data
    num_puncta_punish = diam * 3 / 50
    best_num_cluster = 1
    best_model = GaussianMixture(covariance_type="spherical").fit(data)
    best_loss = np.sqrt(float(best_model.covariances_))
    for i in range(2, len(data) + 1):
        cur_model = GaussianMixture(n_components=i, covariance_type="spherical").fit(data)
        cur_loss = np.mean(np.sqrt(cur_model.covariances_))
        components = np.array(cur_model.means_)
        inter_group_distances = [1 / np.sqrt(sum((np.array(ctr1) - np.array(ctr2)) ** 2)) for ctr1, ctr2 in
                                 itertools.combinations(components, 2)]
        cur_loss += diam * np.mean(inter_group_distances)
        cur_loss += i * num_puncta_punish
        if cur_loss < best_loss:
            best_num_cluster, best_loss, best_model = i, cur_loss, cur_model
    return best_model.means_

def new_punctate(df, ch):
    """
    Given a signal DataFrame with puncta feature for the designated
    channel ch, an integer, create a new column that is True for
    rows that have more than x frames with identified puncta, and
    False otherwise.
    """
    result = []
    for i in range(len(df)):
        cur_punctate = False
        cur_GUV = df.iloc[i]
        for j in range(cur_GUV["num frame"]):
            if cur_GUV[f"puncta {j} ch{ch}"].get_num_puncta() > 0:
                cur_punctate = True
                break
        result.append(cur_punctate)
    df[f"new punctate ch{ch}"] = result
    return df
#     result = []
#     for i in range(len(df)):
#         cur_num_punctate = 0
#         for j in range(df.iloc[i]["num frame"]):
#             if df.iloc[i][f"puncta {j} ch{ch}"][0] > 0:
#                 cur_num_punctate += 1
#         result.append(cur_num_punctate)
#     result = np.array(result)
#     df[f"new punctate ch{ch}"] = (result >= 1)
#     return df

def colocalization(df, chs):
    """
    Given a signal DataFrame with puncta feature for the designated
    channels chs, a list of integers, create 2 * (len(ch2) choose 2)
    number of new columns. For each combination of channels, e.g., a
    column named "colocalizaiton ch0 ch1" has the percentage of colocalizing
    punctum between channel 0 and channel 1, and a column named
    "colocalization weight ch0 ch1" has the total number of punctum
    identified in all frames and 2 channels combined.
    """
    df = df.copy()
    for ch1, ch2 in itertools.combinations(chs, 2):
        df[f"colocalization ch{ch1} ch{ch2}"] = np.nan
        df[f"colocalization weight ch{ch1} ch{ch2}"] = np.nan
        for i in range(len(df)):
            img_sz = df["image size"].iloc[i]
            coloc, total = 0, 0
            for j in range(df["num frame"][i]):
                cur_GUV = df.iloc[i]
                ch1_puncta_coord, ch2_puncta_coord = cur_GUV[f"puncta {j} ch{ch1}"].get_puncta_coords(), cur_GUV[f"puncta {j} ch{ch2}"].get_puncta_coords()
                total += max(len(ch1_puncta_coord), len(ch2_puncta_coord))
                threshold = max(cur_GUV[f"w {j}"], cur_GUV[f"h {j}"]) * img_sz / 4
                if len(ch1_puncta_coord) > len(ch2_puncta_coord):
                    ch1_puncta_coord, ch2_puncta_coord = ch2_puncta_coord, ch1_puncta_coord
                for puncta2 in ch2_puncta_coord:
                    cur_coloc = 0
                    for puncta1 in ch1_puncta_coord:
                        if math.sqrt(sum((np.array(puncta1) - np.array(puncta2)) ** 2)) < threshold:
                            cur_coloc = 1
                            break
                    coloc += cur_coloc
                # coloc = pair_points(list(ch1_puncta_coord), list(ch2_puncta_coord), threshold)
            if total != 0:
                df[f"colocalization ch{ch1} ch{ch2}"].iloc[i] = coloc / total
                df[f"colocalization weight ch{ch1} ch{ch2}"].iloc[i] = total
    return df

def pair_points(group1, group2, threshold):
    if len(group1) == 0:
        return 0
    results = []
    for puncta1 in group1:
        new_group1 = group1[:]
        new_group1.remove(puncta1)
        for puncta2 in group2:
            new_group2 = group2[:]
            if math.sqrt(sum((np.array(puncta1) - np.array(puncta2)) ** 2)) < threshold:
                new_group2.remove(puncta2)
                results.append(1 + pair_points(new_group1[:], new_group2, threshold))
            else:
                results.append(pair_points(new_group1[:], new_group2, threshold))
    return max(results)

def preprocess_for_puncta(img, background):
    """
    Convert the given 2D image into a preprocessed image for puncta
    identification.
    """
    if background is not None:
        img = np.maximum(np.array(img) - background, np.zeros_like(img))
        img = img > threshold_otsu(img)
    else:
        io.imsave("temp.tif", img)
        img = cv.imread("temp.tif", 0)
        os.remove("temp.tif")
        img = cv.fastNlMeansDenoising(img, h=3)
        img = cv.GaussianBlur(img, (3, 3), 0)
        img_shape = img.shape
        img = normalize(np.array([np.ravel(img)])).reshape(img_shape)
        img = img > 1.2 * threshold_otsu(img)
    return img

# Manual Colocalization
def manual_label_position(row_series):
    return row_series["top left x"], row_series["bottom right x"], row_series["top left y"], row_series[
        "bottom right y"]

def manual_colocalization(df, chs, upstream_channel):
    """
    Given a signal DataFrame with puncta feature for the designated
    channels chs, a list of integers, create 2 * (len(ch2) choose 2)
    number of new columns. For each combination of channels, e.g., a
    column named "colocalizaiton ch0 ch1" has the percentage of colocalizing
    punctum between channel 0 and channel 1, and a column named
    "colocalization weight ch0 ch1" has the total number of punctum
    identified in all frames and 2 channels combined.
    """
    df = df.copy()
    for ch1, ch2 in itertools.combinations(chs, 2):
        df[f"colocalization ch{ch1} ch{ch2}"] = np.nan
        df[f"colocalization weight ch{ch1} ch{ch2}"] = np.nan
        coloc, total = [0 for _ in range(len(df))], [0 for _ in range(len(df))]
        for i in range(len(df)):
            cur_GUV = df.iloc[i]
            for j in range(cur_GUV["num frame"]):
                ch1_puncta_coord, ch2_puncta_coord = cur_GUV[f"puncta {j} ch{ch1}"].get_puncta_coords(), cur_GUV[f"puncta {j} ch{ch2}"].get_puncta_coords()
                x1, x2, y1, y2 = manual_label_position(cur_GUV)
                threshold = max(x2 - x1, y2 - y1) / 4
                if ch1 == upstream_channel or len(ch1_puncta_coord) > len(ch2_puncta_coord):
                    ch1_puncta_coord, ch2_puncta_coord = ch2_puncta_coord, ch1_puncta_coord
                total[i] += len(ch2_puncta_coord)
                for puncta2 in ch2_puncta_coord:
                    for puncta1 in ch1_puncta_coord:
                        if math.sqrt(sum((np.array(puncta1) - np.array(puncta2)) ** 2)) < threshold:
                            coloc[i] += 1
                            break
        df[f"colocalization ch{ch1} ch{ch2}"] = np.array(coloc) / np.array(total)
        df[f"colocalization weight ch{ch1} ch{ch2}"] = np.array(total)

    if len(chs) > 2:
        for ch1, ch2, ch3 in itertools.combinations(chs, 3):
            df[f"colocalization ch{ch1} ch{ch2} ch{ch3}"] = np.nan
            df[f"colocalization weight ch{ch1} ch{ch2} ch{ch3}"] = np.nan
            coloc, total = [0 for _ in range(len(df))], [0 for _ in range(len(df))]
            for i in range(len(df)):
                cur_GUV = df.iloc[i]
                for j in range(cur_GUV["num frame"]):
                    ch1_puncta_coord, ch2_puncta_coord, ch3_puncta_coord = cur_GUV[f"puncta {j} ch{ch1}"].get_puncta_coords(), cur_GUV[f"puncta {j} ch{ch2}"].get_puncta_coords(), cur_GUV[f"puncta {j} ch{ch3}"].get_puncta_coords()
                    chs_puncta_coord_dict = dict(list(zip([ch1, ch2, ch3], [ch1_puncta_coord, ch2_puncta_coord, ch3_puncta_coord])))
                    total[i] += len(chs_puncta_coord_dict[upstream_channel])
                    x1, x2, y1, y2 = manual_label_position(cur_GUV)
                    threshold = max(x2 - x1, y2 - y1) / 4
                    upstream_puncta_coord = chs_puncta_coord_dict.pop(upstream_channel)
                    for upstream_puncta in upstream_puncta_coord:
                        cur_coloc = 0
                        upstream_puncta = np.array(upstream_puncta)
                        for pair in itertools.product(zip(*chs_puncta_coord_dict.values())):
                            if len(pair) == 2:
                                puncta1, puncta2 = np.array(pair[0]), np.array(pair[1])
                                if math.sqrt(sum((upstream_puncta - puncta1) ** 2)) < threshold and math.sqrt(sum((upstream_puncta - puncta2) ** 2)) < threshold:
                                    coloc[i] += 1
                                    break
            df[f"colocalization ch{ch1} ch{ch2} ch{ch3}"] = np.array(coloc) / np.array(total)
            df[f"colocalization weight ch{ch1} ch{ch2} ch{ch3}"] = np.array(total)
    return df

def new_colocalization(df, im, chs, puncta_pixel_threshold):
    """
    Given a signal DataFrame with puncta feature for the designated
    channels chs, a list of integers, create 2 * (len(ch2) choose 2)
    number of new columns. For each combination of channels, e.g., a
    column named "colocalizaiton ch0 ch1" has the percentage of colocalizing
    punctum between channel 0 and channel 1, and a column named
    "colocalization weight ch0 ch1" has the total number of punctum
    identified in all frames and 2 channels combined.
    """
    df = df.copy()
    img_sz = im.shape[1]
    for ch1, ch2 in itertools.combinations(chs, 2):
        df[f"new colocalization ch{ch1} ch{ch2}"] = np.nan
        for folder in np.unique(df["folder"]):
            for file_name in np.unique((df[df["folder"] == folder])["file name"]):
                cur_df = df[np.array(df["folder"] == folder) * np.array(df["file name"] == file_name)]
                coloc_result = [0 for _ in range(len(cur_df))]
                for j in range(cur_df["num frame"].iloc[0]):
                    if len(im.shape) == 4:
                        all_ch_img = im[j, :, :, :]
                    else:
                        all_ch_img = im
                    for ch1, ch2 in itertools.combinations(chs, 2):
                        processed_ch_img1, processed_ch_img2 = preprocess_for_coloc(all_ch_img[:, :, ch1], puncta_pixel_threshold[ch1]), preprocess_for_coloc(all_ch_img[:, :, ch2], puncta_pixel_threshold[ch2])
                        coloc_img = processed_ch_img1 * processed_ch_img2
                        for i in range(len(cur_df)):
                            cur_GUV = cur_df.iloc[i]
                            try:
                                x1, x2, y1, y2 = get_coord([cur_GUV[f"x {j}"], cur_GUV[f"y {j}"], cur_GUV[f"w {j}"], cur_GUV[f"h {j}"]], img_sz=img_sz)
                                cur_GUV_img = coloc_img[y1:y2, x1:x2]
                                base_ch = ch1 if cur_GUV[f"puncta {j} ch{ch1}"].get_num_puncta() > cur_GUV[f"puncta {j} ch{ch2}"].get_num_puncta() else ch2
                                for bbox in cur_GUV[f"puncta {j} ch{base_ch}"].get_puncta_bbox():
                                    b_y1, b_x1, b_y2, b_x2 = bbox
                                    labels = measure.label(cur_GUV_img[b_y1:b_y2+1, b_x1:b_x2+1])
                                    for label_property in measure.regionprops(labels):
                                        if label_property.area >= 5:
                                            coloc_result[i] += 1
                                            break
                            except ValueError:
                                pass
                df.loc[cur_df.index, f"new colocalization ch{ch1} ch{ch2}"] = np.array(coloc_result) / np.array(df.loc[cur_df.index, f"colocalization weight ch{ch1} ch{ch2}"])
    return df

def preprocess_for_coloc(img, background):
    """
    Convert the given 2D image into a preprocessed image for puncta
    identification.
    """
    size = (5, 5)
    if background is not None:
        img = np.maximum(img - background, np.zeros_like(img))
    # io.imsave("temp.tif", img)
    # img = cv.imread("temp.tif", 0)
    # os.remove("temp.tif")
    # img_shape = img.shape
    # img = cv.fastNlMeansDenoising(img, h=5)
    # img = cv.GaussianBlur(img, size, 0)
    # img = normalize(np.array([np.ravel(img)])).reshape(img_shape)
    # img = cv.erode(img, None, iterations=2)
    # img = cv.dilate(img, None, iterations=2)
    img = gaussian(img)
    img_max = filters.maximum_filter(img, size)
    img_min = filters.minimum_filter(img, size)
    img = img_max - img_min
    img = img > threshold_li(img)
    return img

def new_manual_colocalization(df, chs, upstream_channel, puncta_pixel_threshold):
    """
    Given a signal DataFrame with puncta feature for the designated
    channels chs, a list of integers, create 2 * (len(ch2) choose 2)
    number of new columns. For each combination of channels, e.g., a
    column named "colocalizaiton ch0 ch1" has the percentage of colocalizing
    punctum between channel 0 and channel 1, and a column named
    "colocalization weight ch0 ch1" has the total number of punctum
    identified in all frames and 2 channels combined.
    """
    df = df.copy()
    for ch1, ch2 in itertools.combinations(chs, 2):
        df[f"new colocalization ch{ch1} ch{ch2}"] = np.nan
        for file_path in df["file path"].unique():
            cur_df = df[df["file path"] == file_path]
            coloc_result = [0 for _ in range(len(cur_df))]
            im = io.imread(cur_df["file path"].iloc[0])
            for j in range(cur_df["num frame"].iloc[0]):
                if len(im.shape) == 4:
                    all_ch_img = im[j, :, :, :]
                else:
                    all_ch_img = im
                processed_ch_img1, processed_ch_img2 = preprocess_for_coloc(all_ch_img[:, :, ch1], puncta_pixel_threshold[ch1]), preprocess_for_coloc(all_ch_img[:, :, ch2], puncta_pixel_threshold[ch2])
                coloc_img = processed_ch_img1 * processed_ch_img2
                for i in range(len(cur_df)):
                    cur_GUV = cur_df.iloc[i]
                    x1, x2, y1, y2 = manual_label_position(cur_GUV)
                    cur_GUV_img = coloc_img[y1:y2, x1:x2]
                    if ch1 != upstream_channel and ch2 != upstream_channel:
                        if cur_GUV[f"puncta {j} ch{ch1}"].get_num_puncta() > cur_GUV[f"puncta {j} ch{ch2}"].get_num_puncta():
                            cur_bbox = cur_GUV[f"puncta {j} ch{ch1}"].get_puncta_bbox()
                        else:
                            cur_bbox = cur_GUV[f"puncta {j} ch{ch2}"].get_puncta_bbox()
                    else:
                        cur_bbox = cur_GUV[f"puncta {j} ch{upstream_channel}"].get_puncta_bbox()
                    for bbox in cur_bbox:
                        b_y1, b_x1, b_y2, b_x2 = bbox
                        labels = measure.label(cur_GUV_img[b_y1:b_y2, b_x1:b_x2])
                        for label_property in measure.regionprops(labels):
                            if label_property.area >= 5:
                                coloc_result[i] += 1
                                break
            df.loc[cur_df.index, f"new colocalization ch{ch1} ch{ch2}"] = np.array(coloc_result) / np.array(df.loc[cur_df.index, f"colocalization weight ch{ch1} ch{ch2}"])

    if len(chs) > 2:
        for ch1, ch2, ch3 in itertools.combinations(chs, 3):
            df[f"new colocalization ch{ch1} ch{ch2} ch{ch3}"] = np.nan
            for file_path in df["file path"].unique():
                cur_df = df[df["file path"] == file_path]
                coloc_result = [0 for _ in range(len(cur_df))]
                im = io.imread(cur_df["file path"].iloc[0])
                for j in range(cur_df["num frame"].iloc[0]):
                    if len(im.shape) == 4:
                        all_ch_img = im[j, :, :, :]
                    else:
                        all_ch_img = im
                    processed_ch_img1, processed_ch_img2, processed_ch_img3 = preprocess_for_coloc(all_ch_img[:, :, ch1], puncta_pixel_threshold[ch1]), preprocess_for_coloc(all_ch_img[:, :, ch2], puncta_pixel_threshold[ch2]), preprocess_for_coloc(all_ch_img[:, :, ch3], puncta_pixel_threshold[ch3])
                    coloc_img = processed_ch_img1 * processed_ch_img2 * processed_ch_img3
                    for i in range(len(cur_df)):
                        cur_GUV = cur_df.iloc[i]
                        x1, x2, y1, y2 = manual_label_position(cur_GUV)
                        cur_GUV_img = coloc_img[y1:y2, x1:x2]
                        for bbox in cur_GUV[f"puncta {j} ch{upstream_channel}"].get_puncta_bbox():
                            b_y1, b_x1, b_y2, b_x2 = bbox
                            labels = measure.label(cur_GUV_img[b_y1:b_y2, b_x1:b_x2])
                            for label_property in measure.regionprops(labels):
                                if label_property.area >= 5:
                                    coloc_result[i] += 1
                                    break
                df.loc[cur_df.index, f"new colocalization ch{ch1} ch{ch2} ch{ch3}"] = np.array(coloc_result) / np.array(df.loc[cur_df.index, f"colocalization weight ch{ch1} ch{ch2} ch{ch3}"])
    return df

def dataset_threshold(path_list, channels_of_interest):
    all_picture = None
    for img_folder in path_list:
        file_list = []
        for f in os.listdir(img_folder):
            if f.endswith(".tif"):
                file_list.append(f)
        for i in range(len(file_list)):
            file_name = file_list[i]
            file_path = join(img_folder, file_name)
            img = io.imread(file_path)
            if len(img.shape) == 4:
                chs_img = np.ravel(img[:, :, :, channels_of_interest])
            elif len(img.shape) == 3:
                chs_img = np.ravel(img[:, :, channels_of_interest])
            else:
                raise ValueError("Input imgage should have 3 or 4 channels.")

            if all_picture is None:
                all_picture = chs_img
            else:
                all_picture = np.concatenate((all_picture, chs_img), axis=0)
    return 2.5 * threshold_li(all_picture)

def manual_dataset_threshold(manual_label_file_path_list, channels_of_interest):
    all_picture = None
    for manual_label_df_file_path in manual_label_file_path_list:
        manual_label_df = pd.read_csv(manual_label_df_file_path)
        file_list = manual_label_df["file path"].unique()
        for i in range(len(file_list)):
            file_path = file_list[i]
            img = io.imread(file_path)
            if len(img.shape) == 4:
                chs_img = np.ravel(img[:, :, :, channels_of_interest])
            elif len(img.shape) == 3:
                chs_img = np.ravel(img[:, :, channels_of_interest])
            else:
                raise ValueError("Input image should have 3 or 4 channels.")
            if all_picture is None:
                all_picture = chs_img
            else:
                all_picture = np.concatenate((all_picture, chs_img), axis=0)
    return threshold_otsu(all_picture)

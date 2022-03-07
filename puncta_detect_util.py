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

def extract_image(imfolder, lipid_channel):
  file_list = []
  if not imfolder.endswith(os.path.sep):
    imfolder += os.path.sep
  for i in os.listdir(imfolder):
    if i.endswith(".tif"):
      file_list.append(i)

  for i in range(len(file_list)):
    fname = file_list[i]
    fpath = imfolder + fname
    series_name = fname[:-4]
    series_folder = os.path.join(imfolder, series_name)
    if not os.path.exists(series_folder):
      os.mkdir(series_folder)
    print(fname)
    im = io.imread(fpath)
    try:
      lipid = im[:, :, :, lipid_channel]
      for j in range(len(lipid)):
            frame_fname = os.path.join(series_folder, "{}.jpeg".format(j))
            io.imsave(fname=frame_fname, arr=lipid[j, :, :])
    except IndexError:
      lipid = im[:, :, lipid_channel]
      frame_fname = os.path.join(series_folder, "0.jpeg")
      io.imsave(fname=frame_fname, arr=lipid)

def identify_img(imfolder, yolo_model_path, thresh=0.8):
    file_list = []
    if not imfolder.endswith(os.path.sep):
        imfolder += os.path.sep
    for i in os.listdir(imfolder):
        if i.endswith(".tif"):
            file_list.append(i)

    for x in file_list:
        img_sz = (cv.fastNlMeansDenoising(cv.imread(join(imfolder, x), 0), h=1.5)).shape[1]
        series_folder = imfolder + x[:-4]
        # os.system("python detect.py --weights /content/yolov5/best.pt --img 416 --conf 0.6 --source " + series_folder + " --save-txt")
        try:
            subprocess.check_output(
                ["python", ".\\yolov5\\detect.py", "--weights", yolo_model_path, "--img", str(img_sz), "--conf",
                 str(thresh), "--source", f"{series_folder}", "--save-txt"])
        except subprocess.CalledProcessError as e:
            print(e.output.decode('UTF-8'))

def process_data(imfolder, folder_index_count, result, num_bins, chs_of_interest, lipid_ch, series_type, puncta_model, old_punctate, frame_punctate, verbose, puncta_pixel_threshold):
    file_list = []
    # folder_num = imfolder.split("\\")[-1]
    if not imfolder.endswith(os.path.sep):
        imfolder += os.path.sep
    for f in os.listdir(imfolder):
        if f.endswith(".tif"):
            file_list.append(f)
    for i in range(len(file_list)):
        fname = file_list[i]
        fpath = join(imfolder, fname)
        if verbose:
            print(f"Starting on {fpath}.")
        im = io.imread(fpath)
        signal_df_lst = []

        # calculates the output dataframe for each channel for the current file
        for ch_of_interest in chs_of_interest:
            if len(im.shape) == 4:
                ch = np.array(im[:, :, :, ch_of_interest])
                lipid = np.array(im[:, :, :, lipid_ch])
            elif len(im.shape) == 3:
                ch = np.array(im[:, :, ch_of_interest]).reshape(1, im.shape[0], im.shape[1])
                lipid = np.array(im[:, :, lipid_ch]).reshape(1, im.shape[0], im.shape[1])
            else:
                raise ValueError("Input image should have 3 or 4 channels.")
            size = (5, 5)
            label_folder = ".\\yolov5\\runs\\detect\\exp{}".format(folder_index(folder_index_count))
            # defines local maxima in the lipid channel
            num_frame = len(lipid)
            cur_series = series_type(ch, lipid, imfolder, fname, puncta_model, num_bins, num_frame, old_punctate, frame_punctate, puncta_pixel_threshold)
            for j in range(num_frame):
                label_fname = os.path.join(label_folder, "labels\\{}.txt".format(j))
                try:
                    labels = pd.read_csv(label_fname, sep=" ", header=None)
                    labels = np.array(labels)
                    # excluding all GUVs classified as low_quality as
                    processed_labels = [label[1:] for label in labels if label[0] == 0]
                    # processed_labels = labels[:, 1:]
                except FileNotFoundError:
                    print(f"No GUVs are detected in {fpath} frame {j}!")
                    processed_labels = []
                cur_series.process_frame(processed_labels)

            # saves each feature as a column of a DataFrame
            result_cols = cur_series.get_col_names(ch_of_interest)
            signal = cur_series.get_values()
            signal_df = pd.DataFrame(data=signal, columns=result_cols)
            if not old_punctate:
                signal_df = signal_df.drop(columns=f"punctateness ch{ch_of_interest}")

            temp = np.sum(np.array(signal_df[[f"value {i} ch{ch_of_interest}" for i in range(num_frame)]].isnull()),
                          axis=1)
            signal_df[f"quality ch{ch_of_interest}"] = temp / num_frame <= (1 - 6 / num_frame)
            signal_df["num frame"] = num_frame
            signal_df = new_punctate(signal_df, ch_of_interest)
            signal_df_lst.append(signal_df)

        # deals with when multiple channels of interest are present
        final_df = signal_df_lst[0].sort_values(by="x 0").reset_index(drop=True)
        if len(chs_of_interest) > 1:
            for cur_df in signal_df_lst[1:]:
                cur_df = cur_df.sort_values(by="x 0").reset_index(drop=True)
                cols_to_use = cur_df.columns.difference(final_df.columns).to_list()  # https://stackoverflow.com/questions/19125091/pandas-merge-how-to-avoid-duplicating-columns
                final_df = pd.DataFrame.merge(final_df, cur_df[cols_to_use], left_index=True, right_index=True, how='outer')
            final_df = colocalization(final_df, chs_of_interest)
            final_df = new_colocalization(final_df, im, chs_of_interest)

        if result is None:
            result = final_df
        else:
            result = pd.concat([result, final_df], axis=0, join="outer", ignore_index=True)

        # makes sure the correct label is read for the next image file
        folder_index_count += 1
    result = square_quality(result)
    return result, folder_index_count

def print_result(result, channels_of_interest):
    print("2-channel colocalization")
    # 2-channel colocalization
    for folder in np.unique(result["folder"]):
        cur_folder_result = []
        for file_name in np.unique((result[result["folder"] == folder])["file name"]):
            cur_file_df = result[np.array(result["folder"] == folder) * np.array(result["file name"] == file_name)]
            for ch1, ch2 in itertools.combinations(channels_of_interest, 2):
                cur_chs_df = cur_file_df[np.array(cur_file_df["square quality"])]
                cur_chs_df = cur_chs_df[cur_chs_df[f"colocalization weight ch{ch1} ch{ch2}"].notna()]
                cur_percent = np.mean(cur_chs_df[f"colocalization ch{ch1} ch{ch2}"])
                cur_folder_result.extend([cur_percent] * len(cur_chs_df))
                print(
                    f"The percent colocalization for {folder}{file_name} between channels {ch1} and {ch2} is {cur_percent}.")
        print()
        print(
            f"The percent colocalization for {folder} between channels {ch1} and {ch2} is {np.mean(cur_folder_result)}.")
        print()

    print("2-channel new colocalization")
    # 2-channel new colocalization
    for folder in np.unique(result["folder"]):
        cur_folder_result = []
        for file_name in np.unique((result[result["folder"] == folder])["file name"]):
            cur_file_df = result[np.array(result["folder"] == folder) * np.array(result["file name"] == file_name)]
            for ch1, ch2 in itertools.combinations(channels_of_interest, 2):
                cur_chs_df = cur_file_df[np.array(cur_file_df["square quality"])]
                cur_chs_df = cur_chs_df[cur_chs_df[f"colocalization weight ch{ch1} ch{ch2}"].notna()]
                cur_percent = np.mean(cur_chs_df[f"new colocalization ch{ch1} ch{ch2}"])
                cur_folder_result.extend([cur_percent] * len(cur_chs_df))
                print(
                    f"The percent new colocalization for {folder}{file_name} between channels {ch1} and {ch2} is {cur_percent}.")
        print()
        print(
            f"The percent new colocalization for {folder} between channels {ch1} and {ch2} is {np.mean(cur_folder_result)}.")
        print()

    # channel punctateness
    # for folder in np.unique(result["folder"]):
    #   cur_folder_result = [[] for _ in range(len(channels_of_interest))]
    #   for file_name in np.unique((result[result["folder"] == folder])["file name"]):
    #     cur_file_df = result[np.array(result["folder"] == folder) * np.array(result["file name"] == file_name)]
    #     cur_file_df = cur_file_df[np.array(cur_file_df["square quality"])]
    #     for ch in channels_of_interest:
    #       cur_percent = np.mean(cur_file_df[f"new punctate ch{ch}"])
    #       cur_folder_result[ch].extend([cur_percent] * len(cur_file_df))
    #       print(f"The percent punctateness for {folder}{file_name} in channel {ch} is {cur_percent}.")
    #   for ch in channels_of_interest:
    #     print()
    #     print(f"The percent punctateness for {folder} in channel {ch} is {np.mean(cur_folder_result[ch])}.")
    #     print()
    print("channel punctateness")
    result["temp folder"] = list(map(lambda f: "/".join(f.split("/")[:-2]), result["folder"]))
    for temp_folder in np.unique(result["temp folder"]):
        cur_folder_df = result[result["temp folder"] == temp_folder]
        for ch in channels_of_interest:
            print()
            ch_percent = np.mean(cur_folder_df[cur_folder_df["square quality"]][f"new punctate ch{ch}"])
            print(f"The overall percent punctateness for {temp_folder} in channel {ch} is {ch_percent}.")
            print()

    for ch in channels_of_interest:
        print()
        ch_percent = np.mean(result[result["square quality"]][f"new punctate ch{ch}"])
        print(f"The overall percent punctateness in channel {ch} is {ch_percent}.")
        print()

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

    def __init__(self, first_xywh, initial_frame, folder, fname, img_sz, model, num_bins, old_punctate, frame_punctate, puncta_pixel_treshold):
        self.xs, self.ys, self.ws, self.hs = [], [], [], []
        self.values = []
        self.candidate = first_xywh
        self.init_f = initial_frame
        self.adj = False
        self.punctate = 0 if old_punctate else np.nan
        self.folder = folder
        self.fname = fname
        self.img_sz = img_sz
        self.model = model
        self.num_bins = num_bins
        self.puncta_param = []
        self.old_punctate = old_punctate
        self.frame_punctate = frame_punctate
        self.puncta_pixel_treshold = puncta_pixel_treshold

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
            self.puncta_param.append([0, [], []])
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
        puncta_param_empty = [[0, [], []] for _ in range(self.init_f)]
        return [self.folder, self.fname, self.img_sz, self.get_averaged_punctateness(), self.adj,
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

    def __init__(self, first_xywh, initial_frame, folder, fname, img_sz, model, num_bins, old_punctate, frame_punctate, puncta_pixel_treshold):
        Guv.__init__(self, first_xywh, initial_frame, folder, fname, img_sz, model, num_bins, old_punctate, frame_punctate, puncta_pixel_treshold)
        self.equa_frame = None
        self.equa_size = 0

    def add_value(self, ch_mask, ch, lipid, f):
        if self.candidate is None:
            self.values.append(np.nan)  # TBD!!!!!
            self.puncta_param.append([0, [], []])
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
        puncta_param_empty = [[0, [], []] for _ in range(self.init_f)]
        return [self.folder, self.fname, self.img_sz, self.get_averaged_punctateness(), self.adj, self.init_f,
                self.equa_frame,
                self.equa_size] + value_empty + self.values + puncta_param_empty + self.puncta_param + position_empty + self.xs + position_empty + self.ys + position_empty + self.ws + position_empty + self.hs

class Series:
    """A class storing the GUVs from the same series."""

    def __init__(self, ch, lipid, folder, fname, model, num_bins, num_frame, old_punctate, frame_punctate, puncta_pixel_threshold):
        self.guv_lst = []
        self.frame = 0
        self.ch = ch
        self.lipid = lipid
        self.folder = folder
        self.fname = fname
        self.img_sz = ch.shape[1]
        self.model = model
        self.num_bins = num_bins
        self.num_frame = num_frame
        self.old_punctate = old_punctate
        self.frame_punctate = frame_punctate
        self.puncta_pixel_treshold

    def process_frame(self, guvs):
        add_lst = self.match_all(guvs)
        self.guv_lst.extend(add_lst)
        self.add_value_all()
        self.check_adj_all()

    def add_value_all(self):
        for guv in self.guv_lst:
            ch = self.ch[self.frame, :, :]
            ch_mask = preprocess_for_puncta(ch[:, :], self.puncta_pixel_treshold)
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
                        Guv(xywh, self.frame, self.folder, self.fname, self.img_sz, self.model, self.num_bins,
                            self.old_punctate, self.puncta_pixel_treshold))
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

    def __init__(self, ch, lipid, folder, fname, model, num_bins, num_frame, old_punctate, frame_punctate, puncta_pixel_threshold):
        Series.__init__(self, ch, lipid, folder, fname, model, num_bins, num_frame, old_punctate, frame_punctate, puncta_pixel_threshold)
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
                        Z_Stack_Guv(xywh, self.frame, self.folder, self.fname, self.img_sz, self.model, self.num_bins,
                                    self.old_punctate, self.frame_punctate, self.puncta_pixel_treshold))
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
            cur_center = np.array([int(label_property.centroid[1]), int(label_property.centroid[0])])
            # if sum(np.sqrt((cur_center - center_of_img)**2)) > cutoff:
            coords.append(cur_center)
            bboxes.append(label_property.bbox)
    # components = best_gaussian_mixture(np.array(coords), max(img.shape))
    return [len(coords), coords, bboxes]

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
    rows that have more than 3 frames with identified puncta, and
    False otherwise.
    """
    result = []
    for i in range(len(df)):
        cur_num_punctate = 0
        for j in range(df.iloc[i]["num frame"]):
            if df.iloc[i][f"puncta {j} ch{ch}"][0] > 0:
                cur_num_punctate += 1
        result.append(cur_num_punctate)
    result = np.array(result)
    df[f"new punctate ch{ch}"] = (result >= 1)
    return df

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
                ch1_puncta_coord, ch2_puncta_coord = cur_GUV[f"puncta {j} ch{ch1}"][1], cur_GUV[f"puncta {j} ch{ch2}"][1]
                total += max(len(ch1_puncta_coord), len(ch2_puncta_coord))
                threshold = max(cur_GUV[f"w {j}"], cur_GUV[f"h {j}"]) * img_sz / 3
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

def preprocess_for_puncta(img, theshold):
    """
    Convert the given 2D image into a preprocessed image for puncta
    identification.
    """
    io.imsave("temp.tif", img)
    img = cv.imread("temp.tif", 0)
    os.remove("temp.tif")
    img = cv.fastNlMeansDenoising(img, h=3)
    img = cv.GaussianBlur(img, (3, 3), 0)
    img_shape = img.shape
    img = normalize(np.array([np.ravel(img)])).reshape(img_shape)
    if thresold is None:
        img = img > 1.2 * threshold_otsu(img)
    else:
        img = img > threshold
    return img

# Manual Colocalization
def manual_label_position(row_series):
    return row_series["top left x"], row_series["bottom right x"], row_series["top left y"], row_series[
        "bottom right y"]

def manual_colocalization(df, chs):
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
            for j in range(df["num frame"].iloc[0]):
                cur_GUV = df.iloc[i]
                ch1_puncta_coord, ch2_puncta_coord = cur_GUV[f"puncta {j} ch{ch1}"][1], cur_GUV[f"puncta {j} ch{ch2}"][
                    1]
                x1, x2, y1, y2 = manual_label_position(cur_GUV)
                threshold = max(x2 - x1, y2 - y1) / 3
                if ch1 == 1 or len(ch1_puncta_coord) > len(ch2_puncta_coord):
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
                for j in range(df["num frame"].iloc[0]):
                    cur_GUV = df.iloc[i]
                    ch1_puncta_coord, ch2_puncta_coord, ch3_puncta_coord = cur_GUV[f"puncta {j} ch{ch1}"][1], \
                                                                           cur_GUV[f"puncta {j} ch{ch2}"][1], \
                                                                           cur_GUV[f"puncta {j} ch{ch3}"][1]
                    total[i] += len(ch2_puncta_coord)
                    x1, x2, y1, y2 = manual_label_position(cur_GUV)
                    threshold = max(x2 - x1, y2 - y1) / 3
                    for puncta2 in ch2_puncta_coord:
                        cur_coloc = 0
                        for puncta1 in ch1_puncta_coord:
                            if math.sqrt(sum((np.array(puncta2) - np.array(puncta1)) ** 2)) < threshold:
                                for puncta3 in ch3_puncta_coord:
                                    if math.sqrt(sum((np.array(puncta2) - np.array(puncta3)) ** 2)) < threshold:
                                        cur_coloc = 1
                                        break
                                if cur_coloc == 1:
                                    break
                        coloc[i] += cur_coloc
            df[f"colocalization ch{ch1} ch{ch2} ch{ch3}"] = np.array(coloc) / np.array(total)
            df[f"colocalization weight ch{ch1} ch{ch2} ch{ch3}"] = np.array(total)
    return df

def new_colocalization(df, im, chs):
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
                    if len(im) == 4:
                        all_ch_img = im[j, :, :, :]
                    else:
                        all_ch_img = im
                    for ch1, ch2 in itertools.combinations(chs, 2):
                        processed_ch_img1, processed_ch_img2 = preprocess_for_coloc(all_ch_img[:, :, ch1]), preprocess_for_coloc(all_ch_img[:, :, ch2])
                        coloc_img = processed_ch_img1 * processed_ch_img2
                        for i in range(len(cur_df)):
                            cur_GUV = cur_df.iloc[i]
                            try:
                                x1, x2, y1, y2 = get_coord([cur_GUV[f"x {j}"], cur_GUV[f"y {j}"], cur_GUV[f"w {j}"], cur_GUV[f"h {j}"]], img_sz=img_sz)
                                cur_GUV_img = coloc_img[y1:y2, x1:x2]
                                base_ch = ch1 if cur_GUV[f"puncta {j} ch{ch1}"][0] > cur_GUV[f"puncta {j} ch{ch2}"][0] else ch2
                                for bbox in cur_GUV[f"puncta {j} ch{base_ch}"][2]:
                                    b_y1, b_x1, b_y2, b_x2 = bbox
                                    labels = measure.label(cur_GUV_img[b_y1:b_y2+1, b_x1:b_x2+1])
                                    try:
                                        for label_property in measure.regionprops(labels):
                                            if label_property.area >= 5:
                                                coloc_result[i] += 1
                                                break
                                    except ValueError:
                                        pass
                            except ValueError:
                                pass
                df.loc[cur_df.index, f"new colocalization ch{ch1} ch{ch2}"] = np.array(coloc_result) / np.array(df.loc[cur_df.index, f"colocalization weight ch{ch1} ch{ch2}"])
    return df

def preprocess_for_coloc(img):
    """
    Convert the given 2D image into a preprocessed image for puncta
    identification.
    """
    size = (5, 5)
    io.imsave("temp.tif", img)
    img = cv.imread("temp.tif", 0)
    os.remove("temp.tif")
    img_shape = img.shape
    img = cv.fastNlMeansDenoising(img, h=5)
    img = cv.GaussianBlur(img, size, 0)
    img = normalize(np.array([np.ravel(img)])).reshape(img_shape)
    # img = cv.erode(img, None, iterations=2)
    # img = cv.dilate(img, None, iterations=2)
    img_max = filters.maximum_filter(img, size)
    img_min = filters.minimum_filter(img, size)
    img = img_max - img_min
    img = img > threshold_otsu(img)
    return img

def new_manual_colocalization(df, chs):
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
        base_ch = 1
        df[f"new colocalization ch{ch1} ch{ch2}"] = np.nan
        for file_path in df["file path"].unique():
            cur_df = df[df["file path"] == file_path]
            coloc_result = [0 for _ in range(len(cur_df))]
            im = io.imread(cur_df["file path"].iloc[0])
            for j in range(cur_df["num frame"].iloc[0]):
                if len(im) == 4:
                    all_ch_img = im[j, :, :, :]
                else:
                    all_ch_img = im
                processed_ch_img1, processed_ch_img2 = preprocess_for_coloc(
                    all_ch_img[:, :, ch1]), preprocess_for_coloc(all_ch_img[:, :, ch2])
                coloc_img = processed_ch_img1 * processed_ch_img2
                for i in range(len(cur_df)):
                    cur_GUV = cur_df.iloc[i]
                    x1, x2, y1, y2 = manual_label_position(cur_GUV)
                    cur_GUV_img = coloc_img[y1:y2, x1:x2]
                    if ch1 != 1 and ch2 != 1:
                        if cur_GUV[f"puncta {j} ch{ch1}"][0] > cur_GUV[f"puncta {j} ch{ch2}"][0]:
                            base_ch = ch1
                        else:
                            base_ch = ch2
                    for bbox in cur_GUV[f"puncta {j} ch{base_ch}"][2]:
                        b_y1, b_x1, b_y2, b_x2 = bbox
                        labels = measure.label(cur_GUV_img[b_y1:b_y2, b_x1:b_x2])
                        for label_property in measure.regionprops(labels):
                            if label_property.area >= 5:
                                coloc_result[i] += 1
                                break
            df.loc[cur_df.index, f"new colocalization ch{ch1} ch{ch2}"] = np.array(coloc_result) / np.array(df.loc[cur_df.index, f"colocalization weight ch{ch1} ch{ch2}"])

    if len(chs) > 2:
        for ch1, ch2, ch3 in itertools.combinations(chs, 3):
            base_ch = 1
            df[f"new colocalization ch{ch1} ch{ch2} ch{ch3}"] = np.nan
            for file_path in df["file path"].unique():
                cur_df = df[df["file path"] == file_path]
                coloc_result = [0 for _ in range(len(cur_df))]
                im = io.imread(cur_df["file path"].iloc[0])
                for j in range(cur_df["num frame"].iloc[0]):
                    if len(im) == 4:
                        all_ch_img = im[j, :, :, :]
                    else:
                        all_ch_img = im
                    processed_ch_img1, processed_ch_img2, processed_ch_img3 = preprocess_for_coloc(
                        all_ch_img[:, :, ch1]), preprocess_for_coloc(all_ch_img[:, :, ch2]), preprocess_for_coloc(
                        all_ch_img[:, :, ch3])
                    coloc_img = processed_ch_img1 * processed_ch_img2 * processed_ch_img3
                    for i in range(len(cur_df)):
                        cur_GUV = cur_df.iloc[i]
                        x1, x2, y1, y2 = manual_label_position(cur_GUV)
                        cur_GUV_img = coloc_img[y1:y2, x1:x2]
                        for bbox in cur_GUV[f"puncta {j} ch{base_ch}"][2]:
                            b_y1, b_x1, b_y2, b_x2 = bbox
                            labels = measure.label(cur_GUV_img[b_y1:b_y2, b_x1:b_x2])
                            for label_property in measure.regionprops(labels):
                                if label_property.area >= 5:
                                    coloc_result[i] += 1
                                    break
                df.loc[cur_df.index, f"new colocalization ch{ch1} ch{ch2} ch{ch3}"] = np.array(coloc_result) / np.array(df.loc[cur_df.index, f"colocalization weight ch{ch1} ch{ch2} ch{ch3}"])
    return df

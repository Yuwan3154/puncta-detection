from genericpath import exists
import os
from os.path import join
from re import U
import numpy as np
import pandas as pd
from skimage import io
from puncta_detect_util import extract_image, identify_img, folder_index, get_coord
import shutil
import argparse

def process_data(imfolder, result, folder_index_count, lipid_ch, verbose):
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

        # calculates GUV info for the current file
        if len(im.shape) == 4:
            pass
        elif len(im.shape) == 3:
            im = np.array(im).reshape(1, im.shape[0], im.shape[1])
        else:
            raise ValueError("Input image should have 3 or 4 channels.")
        label_folder = os.path.sep.join(["yolov5", "runs", "detect", "exp{}".format(folder_index(folder_index_count))])
        # defines local maxima in the lipid channel
        num_frame = len(im)
        cur_data = []
        for j in range(num_frame):
            label_file_name = os.path.sep.join([label_folder, "labels", "{}.txt".format(j)])
            try:
                labels = pd.read_csv(label_file_name, sep=" ", header=None)
                labels = np.array(labels)
                # excluding all GUVs classified as low_quality
                processed_labels = [label[1:] for label in labels if label[0] == 0]
                # processed_labels = labels[:, 1:]
            except FileNotFoundError:
                print(f"No GUVs are detected in {file_path} frame {j}!")
                processed_labels = []
            
            for k in range(len(processed_labels)):
                img_sz = im.shape[1]
                cur_GUV_dict = dict()
                processed_coords = get_coord(processed_labels[k], img_sz)
                cur_GUV_dict["file path"] = file_path
                cur_GUV_dict["frame"] = j
                cur_GUV_dict["x1"] = processed_coords[0]
                cur_GUV_dict["x2"] = processed_coords[1]
                cur_GUV_dict["y1"] = processed_coords[2]
                cur_GUV_dict["y2"] = processed_coords[3]
                cur_data.append(cur_GUV_dict)

        cur_df = pd.DataFrame(data=cur_data)
        result = pd.concat([result, cur_df], axis=0, join="outer", ignore_index=True)
        # makes sure the correct label is read for the next image file
        folder_index_count += 1
    return result, folder_index_count

def detect_GUV(folder, detect_channel, threshold, model, verbose):
    label = f"GUV_coordinates_ch{detect_channel}_thresh{threshold}_" + folder.replace("/", "_") # Name your output here
    save_path = join("results", label)
    # if os.path.exists(save_path): # Short circuits if result is already available
    #     return pd.read_pickle(save_path)
    yolo_model_path = os.path.abspath(model) # Designate your yolo model path here

    folder = os.path.abspath(folder)
    if not os.path.exists("results"):
        os.mkdir("results")

    # Get folders
    path_list = []
    for file in os.listdir(folder):
        if not file.startswith(".") and file.endswith("-output"):
            path_list.append(join(folder, file))

    # Extract frames
    for path in path_list:
        extract_image(path, detect_channel)

    if os.path.exists("yolov5/runs"):
        print("Removing old detection!")
        shutil.rmtree("yolov5/runs", ignore_errors=False)
    # Detect GUV using yolov5
    for path in path_list:
        identify_img(path, yolo_model_path, threshold)

    # Save the coordinates
    result, folder_index_count = pd.DataFrame(), 0
    for path in path_list:
        result, folder_index_count = process_data(path, result, folder_index_count, detect_channel, verbose)
    
    result.to_csv(save_path + ".csv", index=False)
    return result



if __name__ == "__main__":
    #warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Process GUV punctateness.')
    # parser.add_argument("--file", metavar="Summary CSV", type=str, nargs=1, help="the path to a .csv file with at least three columns:\n1. channels of interest [int] 2. detect channel [int] 3. experiment folder [path].")
    parser.add_argument("--folder", metavar="Data folder", type=str, nargs=1, help="the data folder containing .nd2 files and its output")
    parser.add_argument("--channel", metavar="GUV detection channel", type=int, nargs=1, help="the channel for detecting GUV, using zero-indexing")
    parser.add_argument("--threshold", metavar="GUV detection threshold", type=float, nargs=1, help="the confidence threshold for detecting GUV")
    parser.add_argument("--model", metavar="Yolov5 model path", type=str, nargs=1, help="the file path for the yolov5 model for detecting GUV")
    parser.add_argument("--verbose", type=bool, const=True, default=False, nargs="?")
    args = vars(parser.parse_args())
    print(args)
    folder, detect_channel, threshold, model, verbose = args["folder"][0], args["channel"][0], args["threshold"][0], args["model"][0], args["verbose"]
    detect_GUV(folder, detect_channel, threshold, model, verbose)

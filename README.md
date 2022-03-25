# puncta-detection
Chenxi Ou Hurley Lab 03/17/2022

This script is for 1. Automated detection and tracking of GUVs in z-stack images 2. Quantification of intensities of GUVs and a rough estimate of puncta and colocalization information. NOTE that this script assumes the protein recruitment is punctate i.e., even if your protein of interest are distributed continuously on the GUV surface, puncta will be detected (likely with a centroid at the center of the GUV).
To get better results, please try to minimize the amount of protein aggregates in the input images and try to have brighter images. 
If your data is in .nd2 format, please convert all your files to .tif Included in this repo is an imageJ macros "nd2totiff.ijm" that can be used for converting .nd2 files to .tif files; when prompted, choose the folder that contains your .nd2 files.

One-time setup:
1. Initialize the submodule yolov5; please Google if you don't know how to do this. 
2. Install the environments using the "environment.yml" file by using "conda env create -f environment.yml" to create a virtual environment (named "puncta"); install Anaconda if you haven't.

Do the following edition in three_in_one.py using your favorite text editor.
1. Edit the folder_list variable (a list) to contain strings of addresses for all of the folders that you wish to analyze together.
2. Edit the "channels_of_interest" variable (a list) to be a list of integers with zero-indexing.
3. Edit the "lipid_channel" variable (an int) to be the channel to be used for GUV detection; lipid channel is preferrable because they tend to be cleaner but protein channel is also fine.
4. Change the "details" variable to True if you wish to see the results for individual Z-stacks (can be very long); by default this is False.
5. Activate the virtual environment created in the one-time setup by "conda activate puncta" and run three_in_one.py in command line with the working directory being this repo folder.

The result will be saved to "results" folder with a copy of .csv file and a copy of pickle file (a DataFrame Object when read out); each row of the DataFrame corresponds to a GUV.
Columns of interest:
"quality ch{ch}": True if the "ch"th channel of this GUV is not blank in at least 6 frames of the Z-stack or if this Z-stack only has one frame
"square quality": True if at least 90% of the GUV is in the picture; this is used to exclude GUVs at the edges
"value {i} ch{ch}": The average intensity of this GUV at the "i"th frame for the "ch"th channel
"new punctate ch{ch}": True if there is at least one puncta present for this GUV in one of the frames for the "ch"th channel
"coloc ch{ch1} ch{ch2}": The proportion of colocalizing puncta for this GUV across all frames between the "ch1"th channel and the "ch2"th channel
"coloc weight ch{ch1} ch{ch2}": The denominator used for calculating the previous proportion

Troubleshooting:
1. Manually remove the folder "yolov5/runs" after running the script once is recommended because removing files with the python script is not guaranteed to work because of permission issue.
2. If prompted with FileNotFoundError, check to make sure that you have followed the method for nd2 to tiff conversion mentioned above and the addresses you entered are correct.
3. For any other errors or concerns please email me at co3154@berkeley.edu
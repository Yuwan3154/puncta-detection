import pandas as pd
from os.path import join
from puncta_detect_util import *

manual_label_fpath = ".\\data\\02-21-2022\\70_ DOPC_30_ DOPS\\200 nM Dark ALG2_100 nM Cy3 ALIX_10 nM CHMP4b_100 nM dark CHMP2A_100 nM dark CHMP3_100 nM LD 655 Vps4b\\Manual_label_Feb_21_2022_final.csv"
manual_label_df = pd.read_csv(manual_label_fpath)

local_file_path = []
for file_path in manual_label_df["file path"]:
  file_dir_decomp = file_path.split("/")
  tif_file_name = file_dir_decomp[-1]
  local_file_path.append("\\".join([".\\data", "\\".join(file_dir_decomp[4:8]), tif_file_name[:-4] + ".nd2-output", "(series 1).tif"]))
manual_label_df["file path"] = local_file_path

manual_label_df.to_csv(manual_label_fpath)

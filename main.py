import numpy as np
import pandas as pd
import math
import itertools
from three_in_one import process_dir
import warnings
warnings.filterwarnings('ignore')
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

def extract_summary(summary_df, result_df, channels_of_interest, index):
    return summary_df

if __name__ == "__main__":
    summary_file = "Sankalp Data Summary - Distance.csv"
    summary_df = pd.read_csv(summary_file, index_col=False)
    for index, experiment_row in summary_df.iterrows():
        exp_dir, channels_of_interest, detect_channel = experiment_row["experiment folder"], list(range(experiment_row["channels of interest"])), experiment_row["detect channel"]
        print(f"Starting on {exp_dir}")
        cur_result_df = process_dir(exp_dir, channels_of_interest, detect_channel)
        extract_summary(summary_df, cur_result_df, channels_of_interest, index)
        summary_df.to_csv(summary_file, index=False)

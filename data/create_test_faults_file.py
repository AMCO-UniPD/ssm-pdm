"""
Python script to create the test_faults file from the test_ttf files
of the 5 test machines of the PHM Data Challenge
"""

import os
import sys
import glob
import ipdb
import pandas as pd

src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(src_path)

from config_vars import PHM_TEST_TOOLS

PHM_DATAPATH = "/mnt/disk1/davide_frizzo/datasets/phm_data_challenge_2018_complete/test_after"
test_ttf_path = os.path.join(PHM_DATAPATH,"test_ttf")
test_faults_path = os.path.join(PHM_DATAPATH,"test_faults")

for test_tool in PHM_TEST_TOOLS:

    print("-"*50)
    print(f"Processing test tool: {test_tool}")
    print("-"*50)

    file_to_search = os.path.join(test_ttf_path,f"{test_tool}*.csv")
    files = glob.glob(file_to_search)

    tool_csv_path = files[0]
    print("-"*50)
    print(f"Reading file: {tool_csv_path}")
    print("-"*50)

    test_ttf_data = pd.read_csv(tool_csv_path)
    test_faults = []

    #NOTE: Find the fault times when the RUL gets to 0

    zero_mask = test_ttf_data == 0.0
    zero_rows = test_ttf_data[zero_mask.any(axis=1)].index

    for row in zero_rows:

        fault_row = test_ttf_data.iloc[row]
        cols_with_zero = fault_row.drop("time")[fault_row == 0.0].index.tolist()

        for col_name in cols_with_zero:
            test_faults.append({
                "time": int(fault_row["time"]),
                "fault_name": col_name,
                "Tool": test_tool
            })

    test_faults_df = pd.DataFrame(test_faults)

    if test_faults_df.shape[0] == 0:

        print("-"*50)
        print(f"Faults df for {test_tool} is empty, let's continue to the next one")
        print("-"*50)
        continue

    test_faults_df["fault_name"] = test_faults_df["fault_name"].str.removeprefix("TTF_")
    test_faults_df["Tool"] = test_faults_df["Tool"].str.replace("_","",regex=False)

    print("-"*50)
    print(f"Test fault df for {test_tool}")
    print("-"*50)
    print(test_faults_df)

    filename = f"{test_tool}_test_fault_data.csv"
    test_faults_df_path = os.path.join(test_faults_path,filename)
    test_faults_df.to_csv(test_faults_df_path,index=False)
    print("-"*50)
    print(f"Test faults df saved at: {test_faults_df_path}")
    print("-"*50)


"""
Python script to concatenate the data in the test and test_after folder since they
are consecutive
"""

import os
import sys
import glob
import ipdb
import pandas as pd

src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(src_path)

PHM_COMPLETE_DATAPATH = "/mnt/disk1/davide_frizzo/datasets/phm_data_challenge_2018_complete"
PHM_TEST_DATAPATH = os.path.join(PHM_COMPLETE_DATAPATH,"test")
PHM_TEST_AFTER_DATAPATH = os.path.join(PHM_COMPLETE_DATAPATH,"test_after")

PHM_TEST_CONCAT_PATH = os.path.join(PHM_COMPLETE_DATAPATH,"test_concat")
if not os.path.exists(PHM_TEST_CONCAT_PATH):
    os.makedirs(PHM_TEST_CONCAT_PATH)

def concatenate_files(
    element_path:str,
    after_element_path:str,
    test_tool: str = "01_M02",
    concatenate_dirpath: str = PHM_TEST_CONCAT_PATH
) -> None:
    """
    Concatenate the two dataframes contained in the test and
    test_after directory and save the concatenated df inside a new folder

    Args:
        element_path (str): path to the file in the test directory
        after_element_path (str): path to the file in the test_after directory
        test_tool (str): name of the ion milling machine tool
        concatenare_dirpath (str): dirpath where to save the concatenated df

    Returns:
        None: the function concatenates the two dfs, saves the concatenated_df in the test_concat
        directory and does not return anything
    """

    element_df = pd.read_csv(element_path)
    after_element_df = pd.read_csv(after_element_path)
    concatenated_df = pd.concat([element_df,after_element_df],axis=0)

    print("-"*50)
    print(f"Element df shape: {element_df.shape}")
    print(f"After Element df shape: {after_element_df.shape}")
    print(f"Concat Element df shape: {concatenated_df.shape}")
    print("-"*50)

    filename = f"{test_tool}_test_concat.csv"
    test_concat_path = os.path.join(concatenate_dirpath,filename)
    concatenated_df.to_csv(test_concat_path,index=False)

    print("-"*50)
    print(f"Concatenated dataframe saved at: {test_concat_path}")
    print("-"*50)

def concatenate_dirs(
    element_dirpath: str,
    after_element_dirpath: str,
    concatenate_dirpath: str
) -> None:
    """
    Modification of concatenate_file to concatenate all the df in the
    subdirectories of test and test_after (i.e. test_ttf and test_faults directories)

    Args:
        element_dirpath (str): path to the subdirectory in the test directory
        after_element_dirpath (str): path to the subdirectory in the test_after directory
        concatenate_dirpath (str): dirpath where to save the concatenated df

    Returns:
        None: the function iterates over the files inside the directory and concatenates them
        but does not return anything
    """

    for element, after_element in zip(os.listdir(element_dirpath),os.listdir(after_element_dirpath)):

        element_path = os.path.join(element_dirpath,element)
        after_element_path = os.path.join(after_element_dirpath,after_element)
        tool_name = os.path.basename(element_path)[0:6]
        after_tool_name = os.path.basename(after_element_path)[0:6]

        assert tool_name == after_tool_name, f"Tool and after tool name should be equal but got {tool_name} and {after_tool_name}"

        print("-"*50)
        print(f"Concatenating {tool_name} inside {os.path.basename(element_dirpath)}")
        print("-"*50)

        concatenate_files(
            element_path = element_path,
            after_element_path = after_element_path,
            test_tool = tool_name,
            concatenate_dirpath = concatenate_dirpath
        )

for element,after_element in zip(os.listdir(PHM_TEST_DATAPATH),os.listdir(PHM_TEST_AFTER_DATAPATH)):

    element_path = os.path.join(PHM_TEST_DATAPATH,element)
    after_element_path = os.path.join(PHM_TEST_AFTER_DATAPATH,after_element)
    tool_name = os.path.basename(element_path)[0:6]
    after_tool_name = os.path.basename(after_element_path)[0:6]

    assert tool_name == after_tool_name, f"Tool and after tool name should be equal but got {tool_name} and {after_tool_name}"

    if os.path.isfile(element_path) and os.path.isfile(after_element_path):

        print("-"*50)
        print(f"Concatenating {tool_name}")
        print("-"*50)

        concatenate_files(
            element_path = element_path,
            after_element_path = after_element_path,
            test_tool = tool_name
        )

    elif os.path.isdir(element_path) and os.path.isdir(after_element_path):

        print("-"*50)
        print(f"Concatenating {os.path.basename(element_path)} directory")
        print("-"*50)

        concatenate_dirpath = os.path.join(PHM_TEST_CONCAT_PATH,os.path.basename(element_path))

        if not os.path.exists(concatenate_dirpath):
            os.makedirs(concatenate_dirpath)

        concatenate_dirs(
            element_dirpath = element_path,
            after_element_dirpath = after_element_path,
            concatenate_dirpath = concatenate_dirpath
        )

    else:
        print("-"*50)
        print("Inside the else it means that element_path and after_element_path are a directory and a file, cannot process that, skipping to next iteration")
        print("-"*50)
        continue


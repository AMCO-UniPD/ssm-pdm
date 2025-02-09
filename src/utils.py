"""
Script containing some utility functions for the `chronos-pdm` project
"""

import time
import yaml

def get_current_time() -> str:
    """
    This function returns the current time in the format 'dd-mm-YYYY_HH-MM-SS'.
    It is used to produce the name of the files saved

    Returns:
        current_time: string representing the current time

    """

    t = time.localtime()
    current_time = time.strftime("%d-%m-%Y_%H-%M-%S", t)
    return current_time

def load_yaml_to_dict(file_path:str) -> dict:
    """
    Loads the contents of a YAML file into a Python dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: A dictionary containing the data from the YAML file, or None if an error occurs.
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)  # Use safe_load to prevent arbitrary code execution
        return data
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

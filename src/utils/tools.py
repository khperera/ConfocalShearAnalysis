"""
Module for tools for maintaining directory 
and other tasks that aren't directly image processing related. 
I.E. generating metadata and image files from lifs 
"""
import os
import json
import shutil

def clean_data_directory():
    """Deletes files and directories in the data folder"""
    files_to_delete = os.listdir("./data/")
    for location in files_to_delete:
        shutil.rmtree(location)


def load_config(config_file_path: str = "./config/defaultconfig.json") -> json:
    """Loads in configuration files"""
    config_file_path = os.path.abspath(config_file_path)

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    with open(config_file_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config

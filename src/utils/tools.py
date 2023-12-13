"""
Module for tools for maintaining directory 
and other tasks that aren't directly image processing related. 
I.E. generating metadata and image files from lifs 
"""
import os
import json
import shutil
import readlif
from readlif.reader import LifFile

class ReadLiFFile:
    """Class to read and generate .tif files from .lif files.
    """
    def __init__(self, lif_location: str = "") -> None:
        self.lif_location = lif_location


def clean_data_directory():
    """Deletes files and directories in the data folder"""
    files_to_delete = os.listdir("./data/")
    print(files_to_delete)
    for location in files_to_delete:
        shutil.rmtree("./data/"+location)


def load_config(config_file_path: str = "./config/defaultconfig.json") -> json:
    """Loads in configuration files"""
    config_file_path = os.path.abspath(config_file_path)

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    with open(config_file_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config

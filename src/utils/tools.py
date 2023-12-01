#class for cleaning up directory and miscellaneous files.
import os
import pathlib
import shutil

class cleanDataDirectory:
    def __init__(self):
        cleaned = False
        self.deleteList = []
    #finds the directories to be cleaned.
    def listDirectories(self):
        self.deleteList = os.listdir("./data/")

    def cleanData(self):
        for location in self.deleteList:
            shutil.rmtree(location)

            
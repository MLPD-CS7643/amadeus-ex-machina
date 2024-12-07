import json
import os
import gdown
from pathlib import Path


ID_LOOKUP_FILE = "secrets/gdrive.json" 
# populate this file with id from "anyone can view" share links
# example:
# {
#   "myfile.zip": "1fxK9YK6avSzkIvvcUHbwjd_5s381CMBT"
# }
# drive.google.com/file/d/ >> 1fxK9YK6avSzkIvvcUHbwjd_5s381CMBT << /view?usp=sharing


def download_from_gdrive(target, destination):
    id = __lookup_file_id(target)
    if id is None:
        return
    __check_path(destination)
    gdown.download(id=id, output=destination)

def download_folder_from_gdrive(target, destination):
    id = __lookup_file_id(target)
    if id is None:
        return
    __check_path(destination)
    gdown.download_folder(id=id, output=destination)

def __lookup_file_id(target):
    json_path = Path(__file__).parents[1] / ID_LOOKUP_FILE
    with open(json_path, 'r') as file:
        lookup = json.load(file)
        if target not in lookup:
            print(f"<{target}> not found in gdrive.")
            return
        return lookup[target]

def __check_path(path):
    normalized_path = os.path.normpath(path)
    directory_path = os.path.dirname(normalized_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
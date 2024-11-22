import requests
import json
import os


CHUNK_SIZE = 32768
ID_LOOKUP_FILE = "secrets/gdrive.json" 
# populate this file with id from "anyone can view" share links
# example:
# {
#   "myfile.zip": "1fxK9YK6avSzkIvvcUHbwjd_5s381CMBT"
# }
# drive.google.com/file/d/ >> 1fxK9YK6avSzkIvvcUHbwjd_5s381CMBT << /view?usp=sharing


def download_from_gdrive(filename, destination):
    file_id = __lookup_file_id(filename)
    if file_id is None:
        return
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params = {'id' : file_id}, stream = True)
    token = __get_confirm_token(response)

    if token:
        params = {'id' : file_id, 'confirm' : token}
        response = session.get(URL, params = params, stream = True)

    __save_response_content(response, destination)

def __lookup_file_id(filename):
    with open(ID_LOOKUP_FILE, 'r') as file:
        lookup = json.load(file)
        if filename not in lookup:
            print(f"File <{filename}> not found in gdrive.")
            return
        return lookup[filename]

def __check_path(path):
    normalized_path = os.path.normpath(path)
    directory_path = os.path.dirname(normalized_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def __get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def __save_response_content(response, destination):
    __check_path(destination)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
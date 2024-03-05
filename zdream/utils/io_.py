import json
import os
import pickle
from typing import Any, Dict, List

from PIL import Image

# --- DIRECTORIES ---

def rmdir(dir: str):
    '''
    Recursively removes the contents of a directory and the directory itself.
    '''

    # Iterate over the contents of the directory
    for item in os.listdir(dir):

        # Construct the full path of the item
        item_path = os.path.join(dir, item)

        # Check if the item is a file
        if os.path.isfile(item_path):
            os.remove(item_path)
        # If the item is a directory, recursively remove its contents
        elif os.path.isdir(item_path):
            rmdir(item_path)

    # After removing all contents, remove the directory itself
    os.rmdir(dir)

# --- JSON ---

def read_json(path: str) -> Dict[str, Any]:
    '''
    Read JSON data from a file.

    :param path: The path to the JSON file.
    :type path: str
    :raises FileNotFoundError: If the specified file is not found.
    :return: The JSON data read from the file.
    :rtype: Dict[str, Any]
    '''

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f'File not found at path: {path}')


def save_json(data: Dict[str, Any], path: str):
    '''
    Save JSON data to a file.

    :param data: The JSON data to be saved.
    :type data: Dict[str, Any]
    :param path: The path to save the JSON file.
    :type path: str
    '''

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

# --- PICKLE ---
        
def store_pickle(data: Dict[str, Any], path: str):
    """
    Store a dictionary as a pickle file.

    :param data: Dictionary to be pickled.
    :type data: Dict[str, Any]
    :param path: File path where the pickled dictionary will be stored.
    :type path: str
    :return: None
    """

    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Dict[str, Any]:
    """
    Load a dictionary from a pickle file.

    :param path: File path from which to load the pickled dictionary.
    :type path: str
    :return: The loaded dictionary.
    :rtype: Dict[str, Any]
    """

    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

# --- IMAGES ---

def to_gif(image_list: List[Image.Image], out_fp: str, duration: int = 100):
    '''
    Save a list of input images as a .gif file.

    :param image_list: List of images to be saved as .gif file.
    :type image_list: List[Image.Image]
    :param out_fp: File path where to save the image.
    :type out_fp: str
    :param duration: Duration of image frame in milliseconds, defaults to 100.
    :type duration: int, optional
    '''

    image_list[0].save(
        out_fp,
        save_all=True,
        optimize=False,
        append_images=image_list[1:],
        loop=0, duration=duration
    )

# --- TXT ---

def numbers_from_file(file_path: str) -> List[int]:
    ''' 
    Read a set of number from files which is expected   
    to contain a number per line.
    '''
    with open(file_path, 'r') as file:
        numbers = [int(line.strip()) for line in file]
    return numbers
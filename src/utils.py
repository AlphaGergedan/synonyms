import os
import json
import re
import numpy as np

def is_valid_text(text):
    return type(text) == str and len(text) > 0

def get_file_path(file_name, subfolder):
    """
    subfolder must be located relative to the project root
    """
    current_directory_path = os.getcwd()

    file_path = os.path.join(current_directory_path, file_name)

    subfolder_path = os.path.join(current_directory_path, subfolder)

    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    file_path = os.path.join(subfolder_path, file_name)

    return file_path

def convert_string_to_float_array(df, column_name):
    """
    Use case: string stored float values can be transformed
    """
    # returns numpy array of floats given dataframe and column name
    str_list = df.loc[:,column_name]
    # replace characters between digits with commas
    str_list = str_list.apply(lambda row: re.sub(r'(?<=\d)\s+(?=-?\d)', ',', row))
    float_list = str_list.apply(lambda row: np.asarray(json.loads(row)))
    return np.vstack(float_list)

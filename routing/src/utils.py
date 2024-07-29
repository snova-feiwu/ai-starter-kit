import os
import pickle

def read_txt_files(directory):
    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} doesn't exist!")
    file_contents = []
    for filename in os.listdir(directory):
        # if filename.endswith(".txt"):
        if filename.startswith("sambatune"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                file_contents.append(file.read())
    return file_contents

def extract_first_values(data, return_list=False):
    if return_list:
        result = []
        for sublist in data:
            sublist_result = []
            for s in sublist:
                # Extract the first element from the set
                first_value = next(iter(s))
                sublist_result.append(first_value)
            result.append(sublist_result)
    else:
        result = set()
        for sublist in data:
            for s in sublist:
                # Extract the first element from the set
                first_value = next(iter(s))
                result.add(first_value)
    
    return result

def read_keywords(filepath):
    with open(filepath, "rb") as file:
        keywords = pickle.load(file)
        return keywords
import glob
import os

def get_filelist(path: str, pattern: str) -> list:
    """
    Get list of files in a directory matching a given pattern.

    Author: Chi Zhang

    Parameters
    ----------
    path : str
        Directory path, e.g., "F:\\data\\maven\\data\\sci\\mag\\matlab_data_1s"
    pattern : str
        Filename pattern, e.g., "*.mat"

    Returns
    -------
    file_list : list of str
        List of full file paths matching the pattern.
    """
    search_path = os.path.join(path, pattern)
    file_list = glob.glob(search_path)
    file_list.sort()
    return file_list

if __name__ == "__main__":
    path = "F:\\data\\maven\\data\\sci\\mag\\matlab_data_1s"
    file_list = get_filelist(path = path, pattern = "*.mat")

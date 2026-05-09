# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:21:04 2024

Author: Chi Zhang
"""

import os
import glob

def dir_file(folder_path: str, filename: str, recursive: bool = False, case_sensitive: bool = True) -> list:
    """
    Search for files matching a filename pattern in a folder.

    Parameters
    ----------
    folder_path : str
        Folder path to search.
    filename : str
        Filename or pattern to search for. Wildcards such as * and ? are supported.
    recursive : bool, optional
        Whether to search subfolders recursively. Default is False.
    case_sensitive : bool, optional
        Whether the search should be case-sensitive. Default is True.

    Returns
    -------
    list
        Matching file paths.

    Examples
    --------
    >>> dir_file('/path/to/folder', '*.txt')
    ['/path/to/folder/file1.txt', '/path/to/folder/file2.txt']

    >>> dir_file('/path/to/folder', 'data_*.csv', recursive=True)
    ['/path/to/folder/data_2021.csv', '/path/to/folder/subdir/data_2022.csv']

    >>> dir_file('/path/to/folder', 'README.md', case_sensitive=False)
    ['/path/to/folder/readme.md']
    """
    # Build the search pattern.
    pattern = os.path.join(folder_path, '**' if recursive else '', filename)

    # Select the search function.
    search_options = glob.glob
    if not case_sensitive:
        search_options = glob.iglob

    # Execute the search.
    if recursive:
        results = search_options(pattern, recursive=True)
    else:
        results = search_options(pattern)

    # Return results as a list.
    return list(results)

# Example usage
if __name__ == "__main__":
    # Example 1: search for all .py files in the current directory.
    print("Example 1: search for all .py files in the current directory")
    py_files = dir_file('.', '*.py')
    print(f"Found .py files: {py_files}\n")

    # Example 2: recursively search for all .txt files in a directory.
    print("Example 2: recursively search for all .txt files in a directory")
    txt_files = dir_file('/path/to/documents', '*.txt', recursive=True)
    print(f"Found .txt files: {txt_files}\n")

    # Example 3: case-insensitive search for README files.
    print("Example 3: case-insensitive search for README files")
    readme_files = dir_file('/path/to/project', 'README.md', case_sensitive=False)
    print(f"Found README files: {readme_files}\n")

    # Example 4: search for a specific filename pattern.
    print("Example 4: search for a specific filename pattern")
    data_files = dir_file('/path/to/data', 'data_20??.csv')
    print(f"Found data files: {data_files}")

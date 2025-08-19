# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:21:04 2024

@author: Win
"""

import os
import glob

def dir_file(folder_path: str, filename: str, recursive: bool = False, case_sensitive: bool = True) -> list:
    """
    在指定文件夹中搜索匹配指定文件名模式的文件。

    参数:
    folder_path : str
        要搜索的文件夹路径。
    filename : str
        要搜索的文件名或模式。支持通配符 * 和 ?。
    recursive : bool, 可选
        是否递归搜索子文件夹。默认为 False。
    case_sensitive : bool, 可选
        搜索是否区分大小写。默认为 True。

    返回:
    list
        匹配的文件路径列表。

    示例:
    >>> dir_file('/path/to/folder', '*.txt')
    ['/path/to/folder/file1.txt', '/path/to/folder/file2.txt']

    >>> dir_file('/path/to/folder', 'data_*.csv', recursive=True)
    ['/path/to/folder/data_2021.csv', '/path/to/folder/subdir/data_2022.csv']

    >>> dir_file('/path/to/folder', 'README.md', case_sensitive=False)
    ['/path/to/folder/readme.md']
    """
    # 构建搜索模式
    pattern = os.path.join(folder_path, '**' if recursive else '', filename)

    # 设置搜索选项
    search_options = glob.glob
    if not case_sensitive:
        search_options = glob.iglob  # iglob 支持不区分大小写的搜索

    # 执行搜索
    if recursive:
        results = search_options(pattern, recursive=True)
    else:
        results = search_options(pattern)

    # 转换结果为列表并返回
    return list(results)

# 使用示例
if __name__ == "__main__":
    # 示例 1：搜索当前目录中的所有 .py 文件
    print("示例 1: 搜索当前目录中的所有 .py 文件")
    py_files = dir_file('.', '*.py')
    print(f"找到的 .py 文件: {py_files}\n")

    # 示例 2：递归搜索指定目录中的所有 .txt 文件
    print("示例 2: 递归搜索指定目录中的所有 .txt 文件")
    txt_files = dir_file('/path/to/documents', '*.txt', recursive=True)
    print(f"找到的 .txt 文件: {txt_files}\n")

    # 示例 3：不区分大小写搜索 README 文件
    print("示例 3: 不区分大小写搜索 README 文件")
    readme_files = dir_file('/path/to/project', 'README.md', case_sensitive=False)
    print(f"找到的 README 文件: {readme_files}\n")

    # 示例 4：搜索特定命名模式的文件
    print("示例 4: 搜索特定命名模式的文件")
    data_files = dir_file('/path/to/data', 'data_20??.csv')
    print(f"找到的数据文件: {data_files}")
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

# Built-in imports
import os


def db_init(local_data_dir):
    r"""Setup the default path of MAVEN data.

    Parameters
    ----------
    local_data_dir : str
        Path to the data.

    """

    # Normalize the path and make sure that it exists
    local_data_dir = os.path.normpath(local_data_dir)
    assert os.path.exists(local_data_dir), f"{local_data_dir} doesn't exists!!"

    # Path to the configuration file.
    pkg_path = os.path.dirname(os.path.abspath(__file__))

    # If config.json exists, read it; otherwise, create a default one
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fs:
            try:
                config = json.load(fs)
            except json.JSONDecodeError:
                print("⚠️ config.json is corrupted or empty, recreating a new one.")
                config = {}
    else:
        print("⚠️ config.json not found, creating a new one.")
        config = {}

    # Update the local data directory path
    config["local_data_dir"] = local_data_dir

    # Write back to config.json with pretty formatting
    with open(config_path, "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)
        print(f"✅ Updated config.json -> local_data_dir = {local_data_dir}")

if __name__ == "__main__":
    db_init("F:\\data\\maven\\data\\sci")
    db_init("/pscratch/sd/c/chizhang/data/maven/data/sci")

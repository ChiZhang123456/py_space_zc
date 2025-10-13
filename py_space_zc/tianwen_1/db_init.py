#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os


def db_init(local_data_dir):
    """
    Setup or update the default path of Tianwen-1 data.

    Parameters
    ----------
    local_data_dir : str
        Path to the Tianwen-1 data directory.
    """

    # Normalize and validate the path
    local_data_dir = os.path.normpath(local_data_dir)
    if not os.path.exists(local_data_dir):
        raise FileNotFoundError(f"{local_data_dir} doesn't exist!")

    # Determine config file location (same dir as this script)
    pkg_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(pkg_path, "config.json")

    # If config.json doesn't exist, create an empty config
    if not os.path.exists(config_path):
        config = {}
    else:
        # Read existing config
        with open(config_path, "r", encoding="utf-8") as fs:
            config = json.load(fs)

    # Update the Tianwen-1 data path
    config["tianwen_data_dir"] = local_data_dir

    # Save updated config
    with open(config_path, "w", encoding="utf-8") as fs:
        json.dump(config, fs, indent=4)

    print(f"Config updated: tianwen_data_dir = {local_data_dir}")


if __name__ == "__main__":
    db_init("F:\\Tianwen_1")

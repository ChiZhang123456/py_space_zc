import os
import json
def get_base_path():
    """
    Retrieve the base path("local_data_dir") for Maven data from the JSON settings file.

    Returns:
    - str: The base path ("local_data_dir") stored in the JSON settings file.

    Raises:
    - FileNotFoundError: If the settings file does not exist.
    - KeyError: If the base path is not found in the settings file.
    """
    # Define the settings file path
    package_directory = os.path.dirname(__file__)
    settings_file = os.path.join(package_directory, 'config.json')  
    # Check if the settings file exists
    if not os.path.isfile(settings_file):
        raise FileNotFoundError('Settings file not found.')
    
    # Load the settings from the JSON file
    with open(settings_file, 'r') as f:
        settings = json.load(f)
    
    # Retrieve the base path from the settings
    if "local_data_dir" not in settings:
        raise KeyError('Base path not found in settings file.')
    
    return settings["local_data_dir"]
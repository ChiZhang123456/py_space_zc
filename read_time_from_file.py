import re
import numpy as np

def _format_time_string(s):
    """Convert 'YYYYMMDD_HHMMSS' to 'YYYY-MM-DDTHH:MM:SS'."""
    date_part, time_part = s.split('_')
    formatted = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}T{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
    return formatted

def read_time_from_file(filenames):
    """
    Extract start and end times from filenames and convert to np.datetime64.

    Parameters
    ----------
    filenames : str or list of str
        One or more filenames containing time info in 'YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS' format.

    Returns
    -------
    start_times, end_times : list of np.datetime64
        Lists of start and end times extracted from filenames.
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    start_times = []
    end_times = []

    for fname in filenames:
        match = re.search(r'(\d{8}_\d{6})_(\d{8}_\d{6})', fname)
        if not match:
            raise ValueError(f"Filename '{fname}' does not contain properly formatted timestamps.")

        start_str, end_str = match.groups()
        start_formatted = _format_time_string(start_str)
        end_formatted = _format_time_string(end_str)

        start_time = np.datetime64(start_formatted)
        end_time = np.datetime64(end_formatted)

        start_times.append(start_time)
        end_times.append(end_time)

    return start_times, end_times

if __name__ == '__main__':
    filenames = [
        '20141202_014112_20141202_061635.mat',
        '20230101_120000_20230101_180000.mat'
    ]
    start_times, end_times = read_time_from_file(filenames)
    for s, e in zip(start_times, end_times):
        print("Start:", s, "| End:", e)


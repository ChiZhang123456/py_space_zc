from spacepy import pycdf
import numpy as np
from typing import Union, List, Tuple

def get_cdf_var_info(cdf_filename: str) -> dict:
    """
    Retrieve information about all variables in a CDF file.

    Parameters
    ----------
    cdf_filename : str
        Path to the CDF file.

    Returns
    -------
    dict
        Dictionary of variable metadata including shape and dtype.
    """
    try:
        cdf_file = pycdf.CDF(cdf_filename)
        var_info = {}
        for var_name in cdf_file:
            var_data = cdf_file[var_name]
            var_info[var_name] = {
                'shape': var_data.shape,
                'dtype': str(var_data.dtype)
            }
        cdf_file.close()
        return var_info

    except Exception as e:
        raise IOError(f"Failed to read CDF file: {e}")



def get_cdf_var(cdf_filename: str, variable_name: Union[str, List[str]], istime: Union[int, List[int]] = 0) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Retrieve one or multiple variables from a CDF file.

    Parameters
    ----------
    cdf_filename : str
        Path to the CDF file.
    variable_name : str or List[str]
        Name(s) of the variable(s) to retrieve.
    istime : int or List[int], optional
        Indicates whether each variable is time (1) or not (0).
        If reading multiple variables, provide a list of same length.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Variable data. If multiple variables are requested, returns a tuple.
    """
    try:
        cdf_file = pycdf.CDF(cdf_filename)
    except Exception:
        raise FileNotFoundError(f"CDF file not found: {cdf_filename}")

    # Normalize input
    if isinstance(variable_name, str):
        variable_name = [variable_name]
    if isinstance(istime, int):
        istime = [istime] * len(variable_name)

    if len(variable_name) != len(istime):
        raise ValueError("Length of variable_name and istime must match.")

    results = []
    for var, is_time in zip(variable_name, istime):
        try:
            data = cdf_file[var][...]
        except KeyError:
            raise KeyError(f"Variable '{var}' not found in CDF file.")

        if is_time == 1:
            results.append(data.astype('datetime64[ns]'))
        else:
            results.append(data)

    cdf_file.close()

    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)

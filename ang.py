import numpy as np

def ang(x1: np.ndarray, x2: np.ndarray) -> np.ndarray | float:
    """
    Compute the angle (in degrees) between pairs of 3D vectors.

    Parameters
    ----------
    x1 : ndarray of shape (3,) or (n, 3)
        First set of vectors.
    x2 : ndarray of shape (3,) or (n, 3)
        Second set of vectors. Must match shape of x1.

    Returns
    -------
    angles : float or ndarray of shape (n,)
        Angle(s) in degrees between each pair of vectors, in the range [0, 180].

    Raises
    ------
    ValueError
        If shapes are incompatible or not 3D vectors.

    Examples
    --------
    >>> ang([1, 0, 0], [0, 1, 0])
    90.0

    >>> ang([[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [1, 0, 0]])
    array([90., 90.])
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    # Reshape (3,) → (1, 3) to unify handling
    if x1.ndim == 1:
        x1 = x1[np.newaxis, :]
    if x2.ndim == 1:
        x2 = x2[np.newaxis, :]

    if x1.shape != x2.shape or x1.shape[1] != 3:
        raise ValueError("Inputs must be (3,) or (n, 3) shaped 3D vectors.")

    dot = np.einsum("ij,ij->i", x1, x2)
    norm1 = np.linalg.norm(x1, axis=1)
    norm2 = np.linalg.norm(x2, axis=1)
    cos_theta = dot / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angles_deg = np.degrees(np.arccos(cos_theta))

    return angles_deg[0] if angles_deg.size == 1 else angles_deg

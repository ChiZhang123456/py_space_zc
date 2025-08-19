from .dot import dot
from .cross import cross
from .norm_normalize import normalize
import numpy as np


def lmn(data, L, M, N):
    """
    Transform 3D vector data from standard Cartesian coordinates (XYZ) to a custom
    orthogonal coordinate system defined by three basis vectors L, M, N.

    This function performs a coordinate system transformation by projecting each data
    point onto the three orthonormal basis vectors L, M, N. The transformation is
    equivalent to rotating the coordinate system such that the new axes align with
    the L, M, N directions.

    Parameters:
    -----------
    data : numpy.ndarray
        Input data array of shape (n, 3) where each row represents a 3D vector
        in the original XYZ coordinate system

    L : numpy.ndarray
        First basis vector of shape (1, 3) or (3,) defining the new coordinate system.
        Will be normalized internally to ensure unit length.

    M : numpy.ndarray
        Second basis vector of shape (1, 3) or (3,) defining the new coordinate system.
        Will be normalized internally to ensure unit length.

    N : numpy.ndarray
        Third basis vector of shape (1, 3) or (3,) defining the new coordinate system.
        Will be normalized internally to ensure unit length.

    Returns:
    --------
    numpy.ndarray
        Transformed data array of shape (n, 3) where each row contains the coordinates
        of the corresponding input vector in the L-M-N coordinate system.
        - Column 0: projections onto L axis
        - Column 1: projections onto M axis
        - Column 2: projections onto N axis

    Notes:
    ------
    - IMPORTANT: This function requires that L, M, N form an orthogonal set of vectors
      (i.e., L⊥M, L⊥N, M⊥N). The function does NOT verify orthogonality - it is the
      caller's responsibility to ensure this precondition.

    - The input vectors L, M, N are automatically normalized to unit vectors, so they
      don't need to be unit length initially, but they must be orthogonal.

    - This transformation preserves the relative geometric relationships between points
      but expresses them in terms of the new coordinate system.

    - The transformation is a rotation if L, M, N form a right-handed orthonormal basis,
      or a rotation + reflection if they form a left-handed basis.

    Mathematical Background:
    ------------------------
    For each input vector v = [x, y, z], the transformation computes:
    - v_L = v · L̂ (projection onto normalized L)
    - v_M = v · M̂ (projection onto normalized M)
    - v_N = v · N̂ (projection onto normalized N)

    The result [v_L, v_M, v_N] represents the same vector v expressed in the L-M-N
    coordinate system.

    Example Usage:
    --------------
    # Define a new coordinate system with L=X+Y, M=X-Y, N=Z
    L = np.array([1, 1, 0])    # not normalized
    M = np.array([1, -1, 0])   # not normalized
    N = np.array([0, 0, 1])    # already normalized

    # Transform some 3D points
    points = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [1, 1, 1]])

    transformed_points = lmn(points, L, M, N)
    """

    # Normalize the basis vectors to ensure they are unit vectors
    # This is crucial for correct coordinate transformation
    L = normalize(L)  # L̂ = L/|L|
    M = normalize(M)  # M̂ = M/|M|
    N = normalize(N)  # N̂ = N/|N|

    # Project each data point onto the three orthonormal basis vectors
    # This computes the coordinates in the new coordinate system
    data_new_L = dot(data, L)  # L-coordinates: data · L̂
    data_new_M = dot(data, M)  # M-coordinates: data · M̂
    data_new_N = dot(data, N)  # N-coordinates: data · N̂

    # Construct the result array with the same shape as input data
    # Each column represents coordinates along one of the new axes
    res = np.zeros_like(data)
    res[:, 0] = data_new_L  # First column: L-axis coordinates
    res[:, 1] = data_new_M  # Second column: M-axis coordinates
    res[:, 2] = data_new_N  # Third column: N-axis coordinates
    return res

# Example usage and test
if __name__ == "__main__":
    print("=== Testing LMN Coordinate Transformation ===\n")

    # Test 1: Simple orthogonal transformation
    print("Test 1: Transform to L-M-N system")

    # Define test data points
    data = np.array([[1, 0, 0],  # Point along X-axis
                     [0, 1, 0],  # Point along Y-axis
                     [0, 0, 1],  # Point along Z-axis
                     [1, 1, 1]])  # Point at (1,1,1)

    # Define new coordinate system (must be orthogonal!)
    L = np.array([1, 1, 0])  # 45° from X in XY plane
    M = np.array([1, -1, 0])  # Orthogonal to L in XY plane
    N = np.array([0, 0, 1])  # Same as Z-axis

    result = lmn(data, L, M, N)

    print("Original data (XYZ coordinates):")
    print(data)
    print("\nBasis vectors:")
    print(f"L: {L} (normalized: {normalize(L).flatten()})")
    print(f"M: {M} (normalized: {normalize(M).flatten()})")
    print(f"N: {N} (normalized: {normalize(N).flatten()})")
    print("\nTransformed data (LMN coordinates):")
    print(result)

    # Verify orthogonality of basis vectors
    L_norm = normalize(L).flatten()
    M_norm = normalize(M).flatten()
    N_norm = normalize(N).flatten()

    print(f"\nOrthogonality check:")
    print(f"L·M = {np.dot(L_norm, M_norm):.6f} (should be ~0)")
    print(f"L·N = {np.dot(L_norm, N_norm):.6f} (should be ~0)")
    print(f"M·N = {np.dot(M_norm, N_norm):.6f} (should be ~0)")
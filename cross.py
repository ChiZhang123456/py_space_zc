import numpy as np


def cross(a, b):
    """
    Compute cross product between two vectors, supporting multiple dimension combinations

    Parameters:
    a: numpy array, shape (1, 3) or (n, 3)
    b: numpy array, shape (1, 3) or (n, 3)

    Returns:
    - If a is (1, 3), b is (1, 3): returns array of shape (1, 3)
    - If a is (n, 3), b is (1, 3): returns array of shape (n, 3), each row cross with b
    - If a is (1, 3), b is (n, 3): returns array of shape (n, 3), a cross with each row
    - If a is (n, 3), b is (m, 3): if n==m, returns array of shape (n, 3), corresponding row cross products
    """
    a = np.array(a)
    b = np.array(b)

    # Ensure inputs are 2D arrays
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    # Check if last dimension is 3
    if a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("Last dimension of vectors must be 3")

    # Case 1: Both are single vectors (1, 3)
    if a.shape[0] == 1 and b.shape[0] == 1:
        result = np.cross(a[0], b[0]).reshape(1, 3)
        return result

    # Case 2: a is multiple vectors, b is single vector
    elif a.shape[0] > 1 and b.shape[0] == 1:
        result = np.cross(a, b)
        return result

    # Case 3: a is single vector, b is multiple vectors
    elif a.shape[0] == 1 and b.shape[0] > 1:
        result = np.cross(a, b)
        return result

    # Case 4: Both are multiple vectors, dimensions must match
    elif a.shape[0] == b.shape[0]:
        result = np.cross(a, b)
        return result

    else:
        raise ValueError(f"Incompatible dimensions: a.shape={a.shape}, b.shape={b.shape}")


# Test examples
if __name__ == "__main__":
    # Test data
    print("=== Testing Vector Cross Product Function ===\n")

    # Test 1: Two single vectors (1, 3) - i × j = k
    a1 = np.array([[1, 0, 0]])  # i vector
    b1 = np.array([[0, 1, 0]])  # j vector
    result1 = cross(a1, b1)
    print(f"Test 1 - Single vector × Single vector (i × j):")
    print(f"a: {a1}")
    print(f"b: {b1}")
    print(f"Result:\n{result1}")
    print(f"Expected: k vector [[0, 0, 1]]\n")


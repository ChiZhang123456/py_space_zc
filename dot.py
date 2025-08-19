import numpy as np


def dot(a, b):
    """
    Compute dot product between two vectors, supporting multiple dimension combinations

    Parameters:
    a: numpy array, shape (1, 3) or (n, 3)
    b: numpy array, shape (1, 3) or (n, 3)

    Returns:
    - If a is (1, 3), b is (1, 3): returns scalar
    - If a is (n, 3), b is (1, 3): returns array of shape (n,), each row dot with b
    - If a is (1, 3), b is (n, 3): returns array of shape (n,), a dot with each row
    - If a is (n, 3), b is (m, 3): if n==m, returns array of shape (n,), corresponding row dot products
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
        result = np.dot(a[0], b[0])
        return result

    # Case 2: a is multiple vectors, b is single vector
    elif a.shape[0] > 1 and b.shape[0] == 1:
        result = np.sum(a * b, axis=1)
        return result

    # Case 3: a is single vector, b is multiple vectors
    elif a.shape[0] == 1 and b.shape[0] > 1:
        result = np.sum(a * b, axis=1)
        return result

    # Case 4: Both are multiple vectors, dimensions must match
    elif a.shape[0] == b.shape[0]:
        result = np.sum(a * b, axis=1)
        return result

    else:
        raise ValueError(f"Incompatible dimensions: a.shape={a.shape}, b.shape={b.shape}")


# Test examples
if __name__ == "__main__":
    # Test data
    print("=== Testing Vector Dot Product Function ===\n")

    # Test 1: Two single vectors (1, 3)
    a1 = np.array([[1, 2, 3]])  # 1x3
    b1 = np.array([[4, 5, 6]])  # 1x3
    result1 = dot(a1, b1)
    print(f"Test 1 - Single vector × Single vector:")
    print(f"a: {a1}")
    print(f"b: {b1}")
    print(f"Result: {result1}")
    print(f"Verification: 1*4 + 2*5 + 3*6 = {1 * 4 + 2 * 5 + 3 * 6}\n")

    # Test 2: Multiple vectors × Single vector (n, 3) × (1, 3)
    a2 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])  # 3x3
    b2 = np.array([[1, 1, 1]])  # 1x3
    result2 = dot(a2, b2)
    print(f"Test 2 - Multiple vectors × Single vector:")
    print(f"a:\n{a2}")
    print(f"b: {b2}")
    print(f"Result: {result2}")
    print(f"Verification: [1+2+3, 4+5+6, 7+8+9] = [6, 15, 24]\n")

    # Test 3: Single vector × Multiple vectors (1, 3) × (n, 3)
    a3 = np.array([[2, 2, 2]])  # 1x3
    b3 = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])  # 3x3
    result3 = dot(a3, b3)
    print(f"Test 3 - Single vector × Multiple vectors:")
    print(f"a: {a3}")
    print(f"b:\n{b3}")
    print(f"Result: {result3}")
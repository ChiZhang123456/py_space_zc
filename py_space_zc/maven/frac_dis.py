import numpy as np
from py_space_zc.maven import bs_mpb_theta
from py_space_zc import acosd, norm


def frac_dis(Pmso):
    """
    Compute fractional distance between MPB and BS.

    Input:
        Pmso : array of shape (3,), (1,3), or (N,3)
               Coordinates in MSO, units: Rm

    Output:
        res  : scalar or array of fractional distances
    """

    # ----------------------------
    # 1. Normalize input shape to (N,3)
    # ----------------------------
    P = np.asarray(Pmso, dtype=float)

    if P.ndim == 1:
        # (3,) → (1,3)
        P = P.reshape(1, 3)

    elif P.ndim == 2 and P.shape[0] == 1 and P.shape[1] == 3:
        # already (1,3)
        pass

    elif P.ndim == 2 and P.shape[1] == 3:
        # (N,3)
        pass

    else:
        raise ValueError("Pmso must have shape (3,), (1,3), or (N,3)")

    # ----------------------------
    # 2. Compute r and theta
    # ----------------------------
    r = norm(P).reshape(-1)                         # shape (N,)
    th = acosd(P[:, 0] / r)             # degrees

    # ----------------------------
    # 3. BS–MPB interpolation
    # ----------------------------
    p = bs_mpb_theta(th)

    # ----------------------------
    # 4. Fractional position
    # ----------------------------
    res = (r - p["rbs"]) / (p["rbs"] - p["rmpb"])

    # ----------------------------
    # 5. Return scalar if input was scalar
    # ----------------------------
    if np.asarray(Pmso).ndim == 1:
        return res[0]
    else:
        return res


if __name__ == "__main__":
    Pmso = np.array([[1.2,2,3],[4,5,6],[7,8,9]])
    print(frac_dis(Pmso))
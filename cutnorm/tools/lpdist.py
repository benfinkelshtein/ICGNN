"""
This package computes the lp distances given two matrices.

Matrices of different dimension and weights will be rescaled and reweighted using the same method in Cutnorm package.
"""
import numpy as np
from ..compute import _compute_C_weighted, _compute_C_eqdim_unweighted, _compute_C_uneqdim_unweighted


def compute_lp_distance(p, A, B, w1=None, w2=None):
    # Input checking
    if type(A) is not np.ndarray:
        A = np.array(A)
    if type(B) is not np.ndarray:
        B = np.array(B)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D matrices")
    n, n2 = np.shape(A)
    m, m2 = np.shape(B)
    if n != n2 or m != m2:
        raise ValueError("A and B must be square matrices")
    if (w1 is None and w2 is not None) or (w1 is not None and w2 is None):
        raise ValueError("Weight vectors required for both matrices")
    if (w1 is not None and w2 is not None and (n != len(w1) or m != len(w2))):
        raise ValueError("Weight vectors need to have the same lenght "
                         "as the first dimension of the corresponding "
                         "matrices")

    if w1 is not None:
        w, C = _compute_C_weighted(A, B, w1, w2)
    else:
        if n == m:
            w, C = _compute_C_eqdim_unweighted(A, B)
        else:
            w, C = _compute_C_uneqdim_unweighted(A, B)

    if p == np.inf:
        return np.linalg.norm(C.flatten(), ord=p) * (n**2)
    return np.linalg.norm(C.flatten(), ord=p) * (n**2) / (n**(2 / p))

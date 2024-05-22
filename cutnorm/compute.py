"""
This package computes the Cutnorm.
"""
import time
import math
import numpy as np
from .OptManiMulitBallGBB import opt_mani_mulit_ball_gbb, cutnorm_quad


def compute_cutnorm(A,
                    B,
                    w1=None,
                    w2=None,
                    max_round_iter=100,
                    logn_lowrank=False,
                    extra_info=False):
    """
    Computes the cutnorm of the differences between the two matrices

    Args:
        A: ndarray, (n, n) matrix
        B: ndarray, (m, m) matrix
        w1: ndarray, (n, 1) array of weights for A
        w2: ndarray, (m, 1) array of weights for B
        max_round_iter: int, number of iterations for gaussian rounding
        logn_lowrank: boolean to toggle log2(n) low rank approximation
        extra_info: boolean, generate extra computational information
    Returns:
        (cutnorm_round, cutnorm_sdp, info)

        cutnorm_round: objective function value from gaussian rounding

        cutnorm_sdp: objective function value from sdp solution

        info: dictionary containing computational information
            Computational information from OptManiMulitBallGBB:
                sdp_augm_n: dimension of augmented matrix
                sdp_relax_rank_p: rank
                sdp_tsolve: computation time
                sdp_itr, sdp_nfe, sdp_feasi, sdp_fval, sdp_g, sdp_nrmG: information from OptManiMulitBallGBB
            Computational information from gaussian rounding:
                round_tsolve: computation time for rounding
                round_approx_list: list of rounded objf values
                round_uis_list: list of uis
                round_vjs_list: list of vjs
                round_uis_opt: optimum uis
                round_vjs_opt: optimum vjs

            Computational information from processing the difference:
                weight_of_C: weight vector of C, the difference matrix

            Cutnorm information:
               cutnorm_sets (S,T): vectors of cutnorm
    Raises:
        ValueError: if A and B are of wrong dimension, or if weight vectors
            does not match the corresponding A and B matrices
    """
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

    cutnorm_round, cutnorm_sdp, info = _compute_cutnorm(
        C, max_round_iter, logn_lowrank, extra_info)

    # Add weight vector into extra_info info
    info['weight_of_C'] = w

    return cutnorm_round, cutnorm_sdp, info


def gaussian_round(U,
                   V,
                   C,
                   max_round_iter,
                   logn_lowrank=False,
                   extra_info=False):
    '''
    Gaussian Rounding for Cutnorm

    The algorithm picks a random standard multivariate gaussian vector
    w in R^p and computes the rounded solution based on sgn(w \dot ui).

    Adopted from David Koslicki's cutnorm rounding code
    https://github.com/dkoslicki/CutNorm
    and Peter Diao's modifications

    Args:
        U: ndarray, (p, n) shaped matrices of relaxed solutions
        V: ndarray, (p, n) shaped matrices of relaxed solutions
        C: ndarray, original (n, n) shaped matrix to compute cutnorm
        max_round_iter: maximum number of rounding operations
        logn_lowrank: boolean to toggle log2(n) low rank approximation
        extra_info: boolean, generate extra computational information
    Returns:
        (approx_opt, uis_opt, vjs_opt, round_info)

        approx_opt: approximated objective function value

        uis_opt: rounded u vector

        vis_opt: rounded v vector

        round_info: information for rounding operation
    '''
    (p, n) = U.shape
    approx_opt = 0
    uis_opt = np.zeros(n)
    vjs_opt = np.zeros(n)
    G = np.random.randn(max_round_iter, p)

    # Computational information
    round_info = {}
    if extra_info:
        approx_list = np.zeros(max_round_iter)
        uis_list = np.zeros((max_round_iter, n))
        vjs_list = np.zeros((max_round_iter, n))

    # Decomposition for low rank
    if logn_lowrank:
        C_U, C_s, C_V = np.linalg.svd(C)
        low_rank = int(np.log2(n))

    for i in range(max_round_iter):
        uis = np.sign(np.matmul(G[i], U))
        vjs = np.sign(np.matmul(G[i], V))

        # Rounding
        if logn_lowrank:
            C_U_low_filtered_sum = np.sum(
                C_U[:, :low_rank] * uis[:, np.newaxis], axis=0)
            C_V_low_filtered_sum = np.sum(
                np.dot(np.diag(C_s[:low_rank]), C_V[:low_rank] * vjs), axis=1)
            approx = np.abs(
                np.sum(C_U_low_filtered_sum * C_V_low_filtered_sum))
        else:
            approx = np.abs(np.sum(C * np.outer(uis, vjs)))

        if approx > approx_opt:
            approx_opt = approx
            uis_opt = uis
            vjs_opt = vjs

        if extra_info:
            approx_list[i] = approx / 4.
            uis_list[i] = uis
            vjs_list[i] = vjs

    # Cutnorm is 1/4 of infinity norm
    approx_opt = approx_opt / 4.

    if extra_info:
        round_info = {
            "round_approx_list": approx_list,
            "round_uis_list": uis_list,
            "round_vjs_list": vjs_list,
            "round_uis_opt": uis_opt,
            "round_vjs_opt": vjs_opt
        }

    return approx_opt, uis_opt, vjs_opt, round_info


def cutnorm_sets(uis, vjs):
    """
    Generates the cutnorm sets from the rounded SDP solutions

    Args:
        uis: ndarray, (n+1, ) shaped array of rounded +- 1 solution
        vis: ndarray, (n+1, ) shaped array of rounded +- 1 solution
    Returns:
        (S, T) Reconstructed S and T sets that are {1, 0}^n

        S: Cutnorm set axis = 0

        T: Cutnorm set axis = 1
    """
    S = -1 * uis[-1] * uis[:-1]
    T = -1 * vjs[-1] * vjs[:-1]

    S = (S + 1) / 2
    T = (T + 1) / 2
    return S, T


def _compute_cutnorm(C, max_round_iter, logn_lowrank=False, extra_info=False):
    """
    Computes the cutnorm of square matrix C

    Args:
        C: ndarray, (n, n) matrix
        max_round_iter: int, maximum rounding iterations
        logn_lowrank: boolean to toggle log2(n) low rank approximation
        extra_info: boolean, extra computational information generation
    Returns:
        (cutnorm_round, cutnorm_sdp, info)

        cutnorm_round: objective function value from gaussian rounding

        cutnorm_sdp: objective function value from sdp solution

        info: dictionary containing computational information
            Computational information from OptManiMulitBallGBB:
                sdp_augm_n: dimension of augmented matrix
                sdp_relax_rank_p: rank
                sdp_tsolve: computation time
                sdp_itr, sdp_nfe, sdp_feasi, sdp_fval, sdp_g, sdp_nrmG: information from OptManiMulitBallGBB
            Computational information from gaussian rounding:
               round_tsolve: computation time for rounding
               round_approx_list: list of rounded objf values
               round_uis_list: list of uis
               round_vjs_list: list of vjs
               round_uis_opt: optimum uis
               round_vjs_opt: optimum vjs

            Cutnorm information:
               cutnorm_sets (S,T): vectors of cutnorm
    """
    n1 = len(C)
    C_col_sum = np.sum(C, axis=0)
    C_row_sum = np.sum(C, axis=1)
    C_tot = np.sum(C_col_sum)
    # Transformation to preserve cutnorm and
    # enforces infinity one norm = 4*cutnorm
    C = np.c_[C, -1.0 * C_row_sum]
    C = np.r_[C, [np.concatenate((-1.0 * C_col_sum, [C_tot]))]]

    # Modify rank estimation for SDP relaxation
    p = int(max(min(round(np.sqrt(2 * n1) / 2), 100), 1))

    # Dim for augmented matrix for SDP
    n2 = 2 * n1 + 2

    # Initial point normalized
    x0 = np.random.randn(p, n2)
    nrmx0 = np.sum(x0 * x0, axis=0)
    x0 = np.divide(x0, np.sqrt(nrmx0))

    tic_sdp = time.time()
    x, g, out = opt_mani_mulit_ball_gbb(
        x0,
        cutnorm_quad,
        C,
        record=0,
        mxitr=600,
        gtol=1e-8,
        xtol=1e-8,
        ftol=1e-10,
        tau=1e-3)
    toc_sdp = time.time()
    tsolve_sdp = toc_sdp - tic_sdp

    # SDP upper bound approximation
    U = x[:, :n2 // 2]
    V = x[:, n2 // 2:]
    cutnorm_sdp = np.abs(np.sum(C * np.matmul(U.T, V))) / 4.0

    # Gaussian Rounding
    tic_round = time.time()
    (cutnorm_round, uis, vjs, round_info) = gaussian_round(
        U, V, C, max_round_iter, logn_lowrank, extra_info)
    toc_round = time.time()
    tsolve_round = toc_round - tic_round

    # Generate cutnorm sets
    (S, T) = cutnorm_sets(uis, vjs)

    info = {
        "cutnorm_sets": (S, T),
        "sdp_tsolve": tsolve_sdp,
        "round_tsolve": tsolve_round
    }
    if extra_info:
        info.update({
            "sdp_augm_n": n2,
            "sdp_relax_rank_p": p,
            "sdp_itr": out['itr'],
            "sdp_nfe": out['nfe'],
            "sdp_feasi": out['feasi'],
            "sdp_fval": out['fval'],
            "sdp_g": g,
            "sdp_nrmG": out['nrmG']
        })
        # Join rounding info
        info.update(round_info)

    return cutnorm_round, cutnorm_sdp, info


def _compute_C_weighted(A, B, w1, w2):
    """
    Generates the difference matrix of the two weighted matricies

    Args:
        A: ndarray, (n, n) matrix
        B: ndarray, (m, m) matrix
        w1: ndarray, (n, 1) array of weights for A
        w2: ndarray, (m, 1) array of weights for B
    Returns:
        C: ndarray, the difference matrix
    """
    v1 = np.hstack((0, np.cumsum(w1)[:-1], 1.))
    v2 = np.hstack((0, np.cumsum(w2)[:-1], 1.))
    v = np.unique(np.hstack((v1, v2)))
    w = np.diff(v)
    n1 = len(w)

    a = np.zeros(n1, dtype=np.int32)
    b = np.zeros(n1, dtype=np.int32)
    for i in range(n1 - 1):
        val = (v[i] + v[i + 1]) / 2
        a[i] = np.argwhere(v1 > val)[0] - 1
        b[i] = np.argwhere(v2 > val)[0] - 1
    # Last element is always itself in new weights
    a[-1] = len(w1) - 1
    b[-1] = len(w2) - 1
    A_sq = A[a]
    A_sq = A_sq[:, a]
    B_sq = B[b]
    B_sq = B_sq[:, b]
    C = (A_sq - B_sq)

    # Normalize C according to weights
    C = C * (np.outer(w, w))
    return w, C


def _compute_C_eqdim_unweighted(A, B):
    """
    Generates the difference matrix of the two equal dimension unweighted matrices

    Args:
        A: ndarray, (n, n) matrix
        B: ndarray, (m, m) matrix
    Returns:
        C: ndarray, the difference matrix
    """
    n, n2 = np.shape(A)
    n1 = n
    w = np.ones(n) / n
    C = (A - B) / (n1 * n1)  # Normalized C
    return w, C


def _compute_C_uneqdim_unweighted(A, B):
    """
    Generates the difference matrix of the two equal dimension unweighted matrices

    Args:
        A: ndarray, (n, n) matrix
        B: ndarray, (m, m) matrix
    Returns:
        C: ndarray, the difference matrix
    """
    n, n2 = np.shape(A)
    m, m2 = np.shape(B)
    d = math.gcd(n, m)
    k = n / d
    l = m / d
    c = k + l - 1
    v1 = np.arange(k) / n
    v2 = np.arange(1, l + 1) / m
    v = np.hstack((v1, v2))
    np.sort(v)
    w = np.diff(v)
    w = np.tile(w, d)

    # Create matrix of differences
    n1 = len(w)
    vals = np.tile(v[:-1], d) + np.floor(np.arange(n1) / c) / d + 1. / (2 * n1)
    a = np.floor(vals * n).astype(int)
    b = np.floor(vals * m).astype(int)
    A_sq = A[a]
    A_sq = A_sq[:, a]
    B_sq = B[b]
    B_sq = B_sq[:, b]
    C = (A_sq - B_sq)

    # Normalize C according to weights
    C = C * (np.outer(w, w))
    return w, C

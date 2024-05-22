"""
This package contains the algorithm in 'A feasible method for optimizationwith orthogonality constraints' by Zaiwen Wen and Wotao Yin.

We have reinterpreted the algorithm from Matlab by Wen & Yin to Python.
"""

import numpy as np


def opt_mani_mulit_ball_gbb(x,
                            fun,
                            args,
                            xtol=1e-6,
                            ftol=1e-12,
                            gtol=1e-6,
                            rho=1e-4,
                            eta=0.1,
                            gamma=0.85,
                            tau=1e-3,
                            nt=5,
                            mxitr=1000,
                            record=0):
    """
    Line search algorithm for optimization on manifold
    Reinterpreted directly from Zaiwen Wen and Wotao Yin's
    Matlab implementation of their paper on
    'A feasible method for optimizationwith orthogonality constraints'

    Args:
        x: Numpy array where each column lies on the unit
            sphere ||x_i||_2 = 1
        fun: Function that returns the objective function value
            and its gradient.
            Params: [x, args]
            Returns: [f, g]
        args: args to be used in fun
        kwargs: Options
            record = 0, no print out
            mxitr       max number of iterations
            xtol        stop control for ||X_k - X_{k-1}||
            gtol        stop control for the projected gradient
            ftol        stop control for abs(F_k - F_{k-1})/(1+|F_{k-1}|) usually, max{xtol, gtol} > ftol
    Returns:
        (x, g, out)

        x: solution

        g: gradient of x

        Out: output information

    """
    out = {}

    crit = np.ones((mxitr, 3))

    # Normalize x
    (n, p) = x.shape
    nrmx = np.sum(x * x, axis=0)
    if np.linalg.norm(nrmx) > 1e-8:
        x = np.divide(x, np.sqrt(nrmx))

    # Initial function value and gradient
    f, g = fun(x, args)
    xtg = np.sum(x * g, axis=0)
    gg = np.sum(g * g, axis=0)
    xx = np.sum(x * x, axis=0)
    xxgg = xx * gg
    dtX = x * xtg - g
    nrmG = np.linalg.norm(dtX, 'fro')

    Q = 1
    Cval = f
    tau_orig = tau
    tau = tau_orig
    if (record >= 1):
        print('----------- Gradient Method with Line search -----------')
        print(
            '{:4} \t {:10} \t {:10} \t  {:10} \t {:5} \t {:9} \t {:7}'.format(
                'Iter', 'tau', 'f(X)', 'nrmG', 'Exit', 'funcCount', 'ls-Iter'))
        print(
            '{:4} \t {:10} \t {:10} \t  {:10} \t {:5} \t {:9} \t {:7}'.format(
                0, 0, f, 0, 0, 0))
    if record == 10:
        out['fvec'] = f

    # Main Iteration
    for itr in range(mxitr):
        xp = x
        fp = f
        gp = g
        dtXP = dtX

        nls = 1
        deriv = rho * nrmG**2
        while True:
            tau2 = tau / 2
            beta = 1 + tau2**2 * (-xtg**2 + xxgg)
            a1 = ((1 + tau2 * xtg)**2 - tau2**2 * xxgg) / beta
            a2 = -tau * xx / beta
            x = xp * a1 + gp * a2

            f, g = fun(x, args)
            out['nfe'] = out['nfe'] + 1 if 'nfe' in out else 1

            if f <= Cval - tau * deriv or nls >= 5:
                break
            tau = eta * tau
            nls = nls + 1

        if record == 10:
            out['fvec'] = [out['fvec'], f] if 'fvec' in out else [f]

        # Recalculate
        xtg = np.sum(x * g, axis=0)
        gg = np.sum(g * g, axis=0)
        xx = np.sum(x * x, axis=0)
        xxgg = xx * gg
        dtX = x * xtg - g
        nrmG = np.linalg.norm(dtX, 'fro')
        s = x - xp
        XDiff = np.linalg.norm(s, 'fro') / np.sqrt(n)
        FDiff = abs(fp - f) / (abs(fp) + 1)

        if record >= 1:
            print(('{:4d} \t {:3.2e} \t {:7.6e} \t {:3.2e} \t {:3.2e}'
                   '\t {:3.2e} \t {:2d}\n').format(itr, tau, f, nrmG, XDiff,
                                                   FDiff, nls))

        crit[itr, :] = [nrmG, XDiff, FDiff]
        mcrit = np.mean(crit[itr - min(nt, itr):itr + 1, :], axis=0)

        if ((XDiff < xtol and FDiff < ftol) or nrmG < gtol
                or np.all(mcrit[1:] < 10 * np.array([xtol, ftol]))):
            out['msg'] = 'converge'
            break

        y = dtX - dtXP
        sy = np.sum(s * y)
        sy = abs(sy)
        tau = tau_orig
        if sy > 0:
            if np.mod(itr, 2) == 0:
                tau = np.sum(s * s) / sy
            else:
                tau = sy / np.sum(y * y)

            # Safeguarding on tau
            tau = max(min(tau, 1e20), 1e-20)

        Qp = Q
        Q = gamma * Qp + 1
        Cval = (gamma * Qp * Cval + f) / Q

    if itr >= mxitr:
        out['msg'] = 'exceed max iteration'

    out['feasi'] = np.linalg.norm(np.sum(x * x, axis=0) - 1)
    if out['feasi'] > 1e-14:
        nrmx = np.sum(x * x, axis=0)
        x = np.divide(x, np.sqrt(nrmx))
        f, g = fun(x, args)
        out['nfe'] = out['nfe'] + 1
        out['feasi'] = np.linalg.norm(np.sum(x * x, axis=0) - 1)

    out['nrmG'] = nrmG
    out['fval'] = f
    out['itr'] = itr
    return x, g, out


def maxcut_quad(V, C):
    """
    Maxcut function to compute objective function value and gradient

    maxcut SDP:
    X is n by n matrix
    max Tr(C*X), s.t., X_ii = 1, X psd

    Args:
        V: ndarray, Low rank model X = V' * V;
        C: ndarray, Objective matrix to compute maxcut

    Returns:
        (f, g)

        f: float, objective funciton value

        g: ndarray, gradient
    """
    # Only taking first arg
    g = 2 * np.matmul(V, C)
    f = np.sum(g * V) / 2
    return f, g


def cutnorm_quad(V, C):
    """
    Cutnorm function to compute objective function value and gradient

    Args:
        V: ndarray, Low rank model X = V' * V;
        C: ndarray, Objective matrix to compute maxcut

    Returns:
        (f, g)

        f: float, objective funciton value

        g: ndarray, gradient
    """
    n = len(C)
    Vs = V[:, n:]
    Us = V[:, :n]

    g = 2 * np.c_[np.matmul(Vs, C.T), np.matmul(Us, C)]
    f = (np.sum(g[:, :n] * Us) + np.sum(g[:, n:] * Vs)) / 2
    return f, g

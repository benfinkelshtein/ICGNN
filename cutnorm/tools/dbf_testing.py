"""
Implementation of DBF statistic

from "Distance-based analysis of variance: approximate inference and an application to genome-wide association studies", Christopher Minas and Giovanni Montana
"""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from scipy.stats import pearson3
from scipy.stats import gamma


def distance_variability(dmatrix, Ic):
    '''
    fstat, trW, trB = distance_variability(dmatrix, Ic)

    compute
    (i) within group variability,
    (ii) between group variability
    (iii) total variability
    (iv) distance based F statistic (DBF)

    Args:
        dmatrix: N by N array of distances
        Ic: N  by G array group membership

    Returns:
        fstat: DBF distance based F statistic, between vs. within group variability
        trW: Within group variability
        trB: Between group variability
    '''
    N = float(dmatrix.shape[0])  # total number of elements
    Ng = np.array(
        (Ic).sum(axis=0),
        dtype=float)  # compute number of elemets in each group

    # Compute the H_c
    Hc = np.dot(Ic * (1.0 / Ng), Ic.T) - (1.0 / N)

    # Compute the Gower's centered product matrix
    Gdel = -0.5 * dmatrix * dmatrix
    #As = np.eye(len(Gs)) - (1.0/N)
    #Gdel = np.dot(np.dot(As, Gs), As)
    Gdel -= Gdel.mean(axis=0)[None, :]
    Gdel -= Gdel.mean(axis=1)[:, None]

    trB = np.trace(np.dot(Hc, Gdel))
    Hc1 = np.eye(len(Hc)) - Hc
    trW = np.trace(np.dot(Hc1, Gdel))

    # Compute the dbf statistic
    fstat = trB / trW

    return fstat, trW, trB


def distribution_parameters(dmatrix, Ic):
    '''
    mc, vc, gc  = distribution_parameters(dmatrix, Ic)
    compute (pearson III) null distribution parameters

    Args:
        dmatrix: N by N array of distances
        Ic: N  by G array group membership

    Returns:
        mc: mean
        vc: variance
        gc: skewness
    '''

    # Compute the test statistic
    N = float(dmatrix.shape[0])  # total number of elements
    n = N  #for convenience
    Ng = np.array(
        (Ic).sum(axis=0),
        dtype=float)  # compute number of elemets in each group

    # Compute the Gower's centered product matrix
    Gdel = -0.5 * dmatrix * dmatrix
    Gdel -= Gdel.mean(axis=0)[None, :]
    Gdel -= Gdel.mean(axis=1)[:, None]

    # Compute the H_c
    Hc = np.dot(Ic * (1.0 / Ng), Ic.T) - (1.0 / N)

    # compute sub-parameters
    A1 = np.diag(Hc).sum()  # trace(Hc)
    A2 = (Hc * Hc).sum()  # Trace(Hc Hc) matrix-wise
    A3 = np.diag(Hc * Hc).sum()  # trace(Hc^2) element-wise
    B1 = np.diag(Gdel).sum()  # tr(Gdel)
    B2 = (Gdel * Gdel).sum()  # Tr(Gdel Gdel) matrix-wise
    B3 = np.diag(Gdel * Gdel).sum()  # trace(Gdel^2) element-wise

    # compute first moment / mean
    mom1 = A1 * B1 / (N - 1)
    mc = mom1

    p11 = ((n - 1) * A3 * B3 + (B1 * B1 - B3) * (A1 * A1 - A3) + 2 *
           (A2 - A3) * (B2 - B3) + 4 * A3 * B3) / (n * (n - 1))
    p22 = (4 * (n - 3) * (2 * A3 - A2) * (2 * B3 - B2) + 2 *
           (2 * A3 - A1 * A1) * (2 * B3 - B1 * B1) * (n - 3) +
           (2 * A2 + A1 * A1 - 6 * A3) *
           (2 * B2 + B1 * B1 - 6 * B3)) / (n * (n - 1) * (n - 2) * (n - 3))
    answermom2 = p11 + p22
    answervc = answermom2 - mc**2

    # compute second moment
    vc1 = (2 * ((N - 1) * A2 - A1 * A1) * ((N - 1) * B2 - B1 * B1))
    vc1 /= (N - 1) * (N - 1) * (N + 1) * (N - 2)
    vc2 = N * (N + 1) * B3 - (N - 1) * (B1 * B1 + 2 * B2)
    vc2 /= (N + 1) * N * (N - 1) * (N - 2) * (N - 3)
    vc2 *= N * (N + 1) * A3 - (N - 1) * (A1 * A1 + 2 * A2)
    mom2 = vc1 + vc2
    # compute var from mom2
    vc = mom2 - mc**2

    # compute subparameters for skew
    A4 = np.diag(np.dot(Hc, np.dot(Hc, Hc))).sum()  # trace(Hc Hc Hc)
    A5 = (np.diag(Hc)**3).sum()  # trace (Hc^3) elementwise
    A6 = (Hc**3).sum()  # sum of elements of Hc^3 elementwise
    A7 = np.dot(np.diag(Hc).T, np.diag(np.dot(Hc, Hc)))
    A8 = np.dot(np.dot(np.diag(Hc).T, Hc), np.diag(Hc))

    B4 = np.diag(np.dot(Gdel, np.dot(Gdel,
                                     Gdel))).sum()  # trace(Gdel Gdel Gdel)
    B5 = (np.diag(Gdel)**3).sum()  # trace (Gdel^3) elementwise
    B6 = (Gdel**3).sum()  # sum of elements of Gdel^3 elementwise
    B7 = np.dot(np.diag(Gdel).T, np.diag(np.dot(Gdel, Gdel)))
    B8 = np.dot(np.dot(np.diag(Gdel).T, Gdel), np.diag(Gdel))

    N2 = N * N
    N3 = N2 * N
    N4 = N3 * N

    n2 = N2
    n3 = N3
    n4 = N4

    answermom3 = (
        (n2 * (n + 1) * (n2 + 15 * n - 4) * A5 * B5 + 4 *
         (n4 - 8 * n3 + 19 * n2 - 4 * n - 16) * A6 * B6 + 24 * (n2 - n - 4) *
         (A6 * B8 + B6 * A8) + 6 *
         (n4 - 8 * n3 + 21 * n2 - 6 * n - 24) * A8 * B8 + 12 *
         (n4 - n3 - 8 * n2 + 36 * n - 48) * A7 * B7 + 12 *
         (n3 - 2 * n2 + 9 * n - 12) * (A1 * A3 * B7 + A7 * B1 * B3) + 3 *
         (n4 - 4 * n3 - 2 * n2 + 9 * n - 12) * A1 * B1 * A3 * B3 + 24 *
         ((n3 - 3 * n2 - 2 * n + 8) * (A7 * B6 + A6 * B7) +
          (n3 - 2 * n2 - 3 * n + 12) *
          (A7 * B8 + A8 * B7)) + 12 * (n2 - n + 4) *
         (A1 * A3 * B6 + B1 * B3 * A6) + 6 * (2 * n3 - 7 * n2 - 3 * n + 12) *
         (A1 * A3 * B8 + A8 * B1 * B3) - 2 * n * (n - 1) * (n2 - n + 4) *
         ((2 * A6 + 3 * A8) * B5 + (2 * B6 + 3 * B8) * A5) - 3 * n * (n - 1) *
         (n - 1) * (n + 4) * ((A1 * A3 + 4 * A7) * B5 +
                              (B1 * B3 + 4 * B7) * A5) + 2 * n * (n - 1) *
         (n - 2) * ((A1 * A1 * A1 + 6 * A1 * A2 + 8 * A4) * B5 +
                    (B1 * B1 * B1 + 6 * B1 * B2 + 8 * B4) * A5) +
         (A1 * A1 * A1) * ((n3 - 9 * n2 + 23 * n - 14) * (B1 * B1 * B1) + 6 *
                           (n - 4) * B1 * B2 + 8 * B4) + 6 * A1 * A2 *
         ((n - 4) * (B1 * B1 * B1) +
          (n3 - 9 * n2 + 24 * n - 14) * B1 * B2 + 4 * (n - 3) * B4) + 8 * A4 *
         ((B1 * B1 * B1) + 3 * (n - 3) * B1 * B2 +
          (n3 - 9 * n2 + 26 * n - 22) * B4) - 16 * ((A1 * A1 * A1) * B6 + A6 *
                                                    (B1 * B1 * B1)) - 6 *
         (A1 * A2 * B6 + A6 * B1 * B2) * (2 * n2 - 10 * n + 16) - 8 *
         (A4 * B6 + A6 * B4) * (3 * n2 - 15 * n + 16) -
         ((A1 * A1 * A1) * B8 + A8 *
          (B1 * B1 * B1)) * (6 * n2 - 30 * n + 24) - 6 *
         (A1 * A2 * B8 + A8 * B1 * B2) * (4 * n2 - 20 * n + 24) - 8 *
         (A4 * B8 + A8 * B4) * (3 * n2 - 15 * n + 24) - (n - 2) *
         (24 * ((A1 * A1 * A1) * B7 + A7 *
                (B1 * B1 * B1)) + 6 * (A1 * A2 * B7 + A7 * B1 * B2) *
          (2 * n2 - 10 * n + 24) + 8 * (A4 * B7 + A7 * B4) *
          (3 * n2 - 15 * n + 24) + (3 * n2 - 15 * n + 6) *
          ((A1 * A1 * A1) * B1 * B3 + A1 * A3 *
           (B1 * B1 * B1)) + 6 * (A1 * A2 * B1 * B3 + A1 * A3 * B1 * B2) *
          (n2 - 5 * n + 6) + 48 * (A4 * B1 * B3 + A1 * A3 * B4))) /
        (n * (n - 1) * (n - 2) * (n - 3) * (n - 4) * (n - 5)))

    answerskew = (answermom3 - 3 * mc * answervc - mc**3) / (answervc**
                                                             (3.0 / 2.0))

    return mc, answervc, answerskew


def inv_f_fn(mc, vc, gc, fstat, trT):
    '''
    inverse fn

    Args:
        mc: mean
        vc: variance
        gc: skewness
        fstat: DBF distance based F statistic, between vs. within group variability
        trT: total group varaibility for DBF statistic

    Returns:
        inv fn val
    '''
    ans = ((trT - mc) * fstat - mc) / (((vc)**(0.5)) * (1 + fstat))
    return (ans)


def dbf_pvalue(mc, vc, gc, fstat, trW, trB):
    '''
    pval = pearson_three(mc, vc, gc, fstat)
    compute one sided p value from standardized pearson three distribution

    Args:
        mc: mean
        vc: variance
        gc: skewness
        fstat: DBF distance based F statistic, between vs. within group variability
        trW: within group variability for DBF statistic
        trB: between group varaibility for DBF statistic

    Returns:
        pval = one sided p value
    '''
    trT = trW + trB  #compute total variability of data

    beta = (trT - mc) / vc  #discontinuity point in trB space
    #alpha = (gc*mc - 2*vc)
    #alpha /= gc*(trT - mc) + 2*vc #endpoint of support in fspace

    alpha = 1.0 / ((gc * trT) / (gc * mc - 2.0 * (vc)**(0.5)) - 1.0)

    a = fstat

    if gc >= 0:
        a1 = gamma.cdf(
            ((trT - mc) / (vc)**(0.5)) + 2 / gc, (4 / (gc**2)), scale=(gc / 2))
        if a >= alpha:
            ans = 1 + gamma.cdf(
                inv_f_fn(mc, vc, gc, fstat, trT) + 2 / gc, (4 / (gc**2)),
                scale=(gc / 2)) - a1
        if a <= -1:
            ans = gamma.cdf(
                inv_f_fn(mc, vc, gc, fstat, trT) + 2 / gc, (4 / (gc**2)),
                scale=(gc / 2)) - a1
        if (-1 < a) and (a < alpha):
            ans = 1 + gamma.cdf(
                inv_f_fn(mc, vc, gc, fstat, trT) + 2 / gc, (4 / (gc**2)),
                scale=(gc / 2)) - a1
    if (gc < 0) and (alpha < -1):
        a1 = gamma.cdf(
            (2 / abs(gc)) - ((trT - mc) / (vc)**(0.5)), (4 / (gc**2)),
            scale=(abs(gc) / 2))
        a1 = 1 - a1
        if a <= alpha:
            ans = gamma.cdf(
                (2 / abs(gc)) - inv_f_fn(mc, vc, gc, fstat, trT),
                (4 / (gc**2)),
                scale=(abs(gc) / 2)) - a1
            ans = 1 - ans
        if a > -1:
            ans = 1 + gamma.cdf(
                (2 / abs(gc)) - inv_f_fn(mc, vc, gc, fstat, trT),
                (4 / (gc**2)),
                scale=(abs(gc) / 2)) - a1
            ans = 1 - ans
        if (alpha < a) and (a <= -1):
            ans = 1 + gamma.cdf(
                (2 / abs(gc)) - inv_f_fn(mc, vc, gc, -1, trT), (4 / (gc**2)),
                scale=(abs(gc) / 2)) - a1
            ans = 1 - ans
    if (gc < 0) and (alpha > -1):
        a1 = gamma.cdf(
            (2 / abs(gc)) - ((trT - mc) / (vc)**(0.5)), (4 / (gc**2)),
            scale=(abs(gc) / 2))
        a1 = 1 - a1
        if a < -1:
            ans = 0
        if (-1 < a) and (a <= alpha):
            ans = gamma.cdf(
                (2 / abs(gc)) - inv_f_fn(mc, vc, gc, fstat, trT),
                (4 / (gc**2)),
                scale=(abs(gc) / 2))
            ans = 1 - ans
        if a > alpha:
            ans = 1
    return (1 - ans)


def dbf_test(dmatrix, labels):
    '''
    pval, fstat, Bvar, Wvar = dbf_test(dmatrix, labels)
    run dbf test

    Args:
        dmatrix: N by N array of distances
        Labels: N array group membership

    Returns:
        pval: one sided p value
        fstat: DBF distance based F statistic, between vs. within group variability
        Bvar: between vs overall group variability
        Wvar: within vs overall group variability
    '''
    G = len(np.unique(labels))
    N = len(labels)
    # check at least 2 grroups
    assert G >= 2, 'Group variability with less than 2 groups not allowed, input G=%d' % (
        G, )
    # check at least 5 samples
    assert N >= 5, 'Test statistic requires N >=5, input N=%d' % (N, )

    # create label binarizer object
    lb = LabelBinarizer()
    # form N x G group label matrix
    # NOTE: when there are only two groups, label binarizer only returns a one column matrix.
    # TEMP FIX: check if G==2, and ad an extra column
    Ic = np.array(lb.fit_transform(labels), dtype=float)
    if G < 3:  # G==2
        Ic = np.hstack((Ic, 1.0 - Ic))

    # compute test statistic
    fstat, trW, trB = distance_variability(dmatrix, Ic)
    Bvar = trB / (trW + trB)  # between vs overall group variability
    Wvar = trW / (trW + trB)  # within vs overall group variability

    # compuete Null parameters
    mc, vc, gc = distribution_parameters(dmatrix, Ic)

    # evaluate P value
    pval = dbf_pvalue(mc, vc, gc, fstat, trW, trB)

    return pval, fstat, Bvar, Wvar

"""
This package generates stochastic matrices.

Currently supported models are general Stochastic Block Models, Erdos Renyi, and Autoregressive Models.
"""
import numpy as np


def sbm(community_sizes, prob_mat, symmetric=True):
    '''
    Generates a stochastic block matrix

    Community_sizes indicate the size of each community and the
    probability matrix indicate the probability that a 1 will be
    generated for each element within the community.

    Args:
        community_sizes: 1d array, shape (n) sizes of community
        prob_mat: 2d array, shape (n,n) probability of edges for each community
        symmetric: boolean, if true, the function will output a symmetric matrix
    Returns:
        stochastic block matrix, 2d array, shape depending on community sizes
    '''
    community_sizes = np.array(community_sizes)
    prob_mat = np.array(prob_mat)
    if prob_mat.shape[0] != prob_mat.shape[1]:
        raise ValueError("prob_mat needs to be square matrix.")
    if not np.all(community_sizes > 0):
        raise ValueError("Each community size in community_sizes "
                         "needs to be greater than 1.")
    if len(community_sizes) != len(prob_mat):
        raise ValueError("community_sizes needs to be of size n if "
                         "prob_mat is nxn")
    if not (np.all(prob_mat >= 0) and np.all(prob_mat <= 1)):
        raise ValueError("Needs to be a valid probability matrix.")

    n = np.sum(community_sizes)
    sbm = np.zeros((n, n))
    for i, size_i in enumerate(community_sizes):
        for j, size_j in enumerate(community_sizes):
            prev_sum_i = np.sum(community_sizes[:i])
            prev_sum_j = np.sum(community_sizes[:j])
            prob = prob_mat[i][j]
            sample = np.random.choice(
                2, size=size_i * size_j, p=[1 - prob, prob]).reshape((size_i,
                                                                      size_j))
            sbm[prev_sum_i:prev_sum_i + size_i, prev_sum_j:
                prev_sum_j + size_j] = sample

    # Discard lower triangular and reflect upper triangular
    if symmetric:
        sbm = make_symmetric_triu(sbm)
    return sbm


def sbm_prob(community_sizes, prob_mat):
    '''
    Generates a matrix indicating the underlying probability
    that gives rise to a stochastic block matrix

    Args:
        community_sizes: 1d array, shape (n) sizes of community
        prob_mat: 2d array, shape (n,n) probability of edges for each community
    Returns:
        probabilities of a stochastic block matrix,
        2d array, shape depending on community sizes
    '''
    community_sizes = np.array(community_sizes)
    prob_mat = np.array(prob_mat)
    if prob_mat.shape[0] != prob_mat.shape[1]:
        raise ValueError("prob_mat needs to be square matrix.")
    if not np.all(community_sizes > 0):
        raise ValueError("Each community size in community_sizes "
                         "needs to be greater than 1.")
    if len(community_sizes) != len(prob_mat):
        raise ValueError("community_sizes needs to be of size n if "
                         "prob_mat is nxn")
    if not (np.all(prob_mat >= 0) and np.all(prob_mat <= 1)):
        raise ValueError("Needs to be a valid probability matrix.")

    n = np.sum(community_sizes)
    sbm = np.zeros((n, n))
    for i, size_i in enumerate(community_sizes):
        for j, size_j in enumerate(community_sizes):
            prev_sum_i = np.sum(community_sizes[:i])
            prev_sum_j = np.sum(community_sizes[:j])
            prob = prob_mat[i][j]
            prob_block = prob * np.ones((size_i, size_j))
            sbm[prev_sum_i:prev_sum_i + size_i, prev_sum_j:
                prev_sum_j + size_j] = prob_block

    return sbm


def sbm_autoregressive(community_sizes, prob_list, symmetric=True):
    '''
    Generates an autoregressive SBM

    An autoregressive SBM has edge probability according to the prob_list
    on the diagonal but (prob_list[i] * prob_list[j])**(abs(i - j))
    for the off-diagonal blocks entries.

    This idea is similar to the autoregressive models

    Args:
        community_sizes: 1d array, shape (n) sizes of community
        prob_list: 1d array, shape (n), where n is the number of diagonal blocks
        symmetric: boolean, if true, the function will output a symmetric matrix
    Returns:
        An autoregressive SBM, 2d array, shape depending on community sizes
    '''
    prob_matrix = _sbm_autoregressive_gen_prob_matrix(prob_list)
    return sbm(community_sizes, prob_matrix, symmetric)


def sbm_autoregressive_prob(community_sizes, prob_list):
    '''
    Generates the underlying probability matrix thatgives
    rise to the autoregressive SBM

    Args:
        community_sizes: 1d array, shape (n) sizes of community
        prob_list: 1d array, shape (n), where n is the number of diagonal blocks
    Returns:
        A probability matrix for an autoregressive SBM,
        2d array, shape depending on community sizes
    '''
    prob_matrix = _sbm_autoregressive_gen_prob_matrix(prob_list)
    return sbm_prob(community_sizes, prob_matrix)


def _sbm_autoregressive_gen_prob_matrix(prob_list):
    '''
    Generates the probability block matrix for an autoregressive SBM

    Each element of the probaiblity block matrix represents the
    probability of an edge within the block. This is of shape (n,n)

    This is different than sbm_autoregressive_prob in that this is
    shape (n,n) whereas the shape of sbm_autoregressive_prob depends
    on the community sizes.

    Args:
        prob_list: 1d array, shape (n), where n is the number of diagonal blocks
    Returns:
        A probability matrix for an autoregressive SBM,
        2d array, shape (n,n)
    '''
    n_probs = len(prob_list)
    prob_matrix = np.zeros((n_probs, n_probs))
    for i in range(n_probs):
        for j in range(n_probs):
            if i == j:
                prob_matrix[i, i] = prob_list[i]
            else:
                prob_matrix[i, j] = (prob_list[i] * prob_list[j])**(abs(i - j))
    return prob_matrix


def make_symmetric_triu(mat):
    '''
    Makes the matrix symmetric upper triangular

    Args:
        mat: 2d array, shape (n,n)
    Returns:
        upper triangular symmetric matrix of the input
        2d array, shape (n,n)
    '''
    mat = np.triu(mat)
    mat = np.maximum(mat, mat.T)
    return mat


def erdos_renyi(n, p, symmetric=True):
    '''
    Generates Erdos Renyi random graph size n with
    probability p

    Args:
        n: int, size of the output matrix
        p: float, edge probability
        symmetric: boolean, if true, the function will output a symmetric matrix
    Returns:
        Erdos Renyi random graph matrix
        2d array, shape (n,n)
    '''
    return sbm([n], [[p]], symmetric)

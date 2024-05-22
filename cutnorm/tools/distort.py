"""
This package adds distortion to matrices.
"""
import numpy as np


def add_gaussian_noise(mat, mean, std):
    '''
    Adds gaussian noise to the matrix

    Args:
        mat: 2d array, shape (n,n)
        mean: gaussian mean
        std: gaussian std
    Returns:
        Processed matrix
    '''
    noise_filter = std * np.random.randn(mat.shape[0], mat.shape[1]) + mean
    return mat + noise_filter


def shift(mat, n_shift):
    '''
    Shifts the matrix by rolling it along the diagonal

    Args:
        mat: 2d array, shape (n,n)
        n_shift: number to roll
    Returns:
        Shifted matrix
    '''
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("mat needs to be square matrix.")
    shifted_mat = np.roll(mat, n_shift, axis=0)
    shifted_mat = np.roll(shifted_mat, n_shift, axis=1)
    return shifted_mat

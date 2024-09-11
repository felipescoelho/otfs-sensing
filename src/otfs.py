"""otfs.py

Script with methods for the OTFS implementation.

luizfelipe.coelho@smt.ufrj.br
Mar 19, 2024
"""


import numpy as np


def idft_mat(N: int) -> np.ndarray:
    """
    Method to generate the unitary inverse discrete Fourier transform
    (IDFT).
    
    Parameters
    ----------
    N : int
        Number of bins in the IDFT.
    
    Returns
    -------
    W_i : np.ndarray
        Matrix that performs the IDFT.
    """

    W_i = np.zeros((N, N), dtype=np.complex128)
    for k in range(N):
        for l in range(N):
            W_i[k, l] = (1/N)*np.exp(1j*2*np.pi*k*l/N)
    
    return W_i


def dft_mat(M: int) -> np.ndarray:
    """
    Method to generate the unitary discrete Fourier transform (DFT).
    
    Parameters
    ----------
    M : int
        Number of bins in the DFT.
    
    Returns
    -------
    W : np.ndarray
        Matrix that performs the DFT.
    """

    W = np.zeros((M, M), dtype=np.complex128)
    for k in range(M):
        for l in range(M):
            W[k, l] = np.exp(-1j*2*np.pi*k*l/M)

    return W

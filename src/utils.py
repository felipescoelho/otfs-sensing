"""utils.py

Script with utilitary methods for ISAC using OTFS.

luizfelipe.coelho@smt.ufrj.br
Mar 19, 2024
"""


import numpy as np
from math import gcd
from numba import njit
from scipy.constants import speed_of_light


def gen_symbols(mod_size: int, N: int, M: int) -> np.ndarray:
    """Method to generate QAM symbols.
    
    Parameters
    ----------
    mod_size : int
        Size of the constellation.
    M : int
        Number of subcarriers or number of elements in fast-time.
    N : int
        Number of elements in slow-time.
    
    Returns
    -------
    dig_symbols : np.ndarray
        Symbols of information.
    """

    match mod_size:
        case 4:
            symbols = (-1-1j, -1+1j, 1-1j, 1+1j)
        case 16:
            symbols = (-3-3j, -3-1j, -3+1j, -3+3j, -1-3j, -1-1j, -1+1j,
                        -1+3j, 1-3j, 1-1j, 1+1j, 1+3j, 3-3j, 3-1j, 3+1j,
                        3+3j)
    dig_symbols = np.random.choice(symbols, size=(N, M), replace=True)

    return dig_symbols


def barker_code(N: int, dtype=np.complex128):
    """Method to generate a Barker code sequence.
    
    Parameters
    ----------
    N : int
        Number of elements in the sequence choice in the set
        N = {2, 3, 4, 5, 7, 11, 13}
    
    Returns
    -------
    x : np.ndarray
        Array containing Barker code sequence.
    """

    match N:
        case 2:
            return np.array((1, -1), dtype=dtype)  # Can be +- or ++
        case 3:
            return np.array((1, 1, -1), dtype=dtype)
        case 4:
            return np.array((1, -1, 1, 1), dtype=dtype)  # Can be +-++ or +---
        case 5:
            return np.array((1, 1, 1, -1, 1), dtype=dtype)
        case 7:
            return np.array((1, 1, 1, -1, -1, 1, -1), dtype=dtype)
        case 11:
            return np.array((1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1),
                            dtype=dtype)
        case 13:
            return np.array((1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1),
                            dtype=dtype)
        case _:
            raise ValueError('N should be in {2, 3, 4, 5, 7, 11, 13}.')


def chirp(T: float, Ts: float, BW: float, chirp_factor: float, M: int):
    """Method to formulate a chirp pulse

    Parameters
    ----------
    T : float
        Pulse duration in seconds
    Ts : float
        Sampling period
    BW : float
        Bandwidth of the chirp signal.
    chirp_factor : float
        Ratio between chirp duration and pulse duration
    M : int
        Total number of delay bins, or number of samples in pulse.

    Returns
    -------
    pulse : np.ndarray
        Pulse containing chirp.
    """

    Tch = T*chirp_factor
    t = np.arange(0, Tch, Ts)
    x = np.exp(-1j*np.pi*(BW/Tch)*t**2)
    pulse = np.hstack((x, np.zeros((M-len(x),), dtype=np.complex128)))

    return pulse


def zadoff_chu(N: int):
    """Method to generate the Zadoff-Chu (ZC) sequence.
    
    Parameters
    ----------
    N : int
        Sequence length.
    

    References
    ----------
    [1] D. Chu, "Polyphase codes with good periodic correlation
    properties (Corresp.)," in IEEE Transactions on Information Theory,
    vol. 18, no. 4, pp. 531-532, July 1972,
    doi: 10.1109/TIT.1972.1054840.
    """

    k = np.arange(N)
    M = N
    for _ in range(N):
        M -= 1
        if gcd(M, N) == 1:
            break
        elif M == 0:
            raise(ValueError('choose another value for N.'))
    if N % 2 == 0:  # The sequence is even
        a = np.exp(1j*M*np.pi*k**2 / N)
    else:  # The sequence is odd
        a = np.exp(1j*M*np.pi*k*(k+1) / N)

    return a


@njit()
def corr_2d(X, Y):
    """"""

    N, M = X.shape
    V = np.zeros((N, M), dtype=np.complex128)
    for k in range(N):
        for l in range(M):
            for m in range(N):
                for n in range(M):
                    if m-l >= 0:
                        V[k, l] += np.conj(Y[n, m]) * X[(n-k)%N, (m-l)%M] \
                            * np.exp(1j*2*np.pi*(m-l)*k/(M*N))
                    else:
                        V[k, l] += np.conj(Y[n, m]) * X[(n-k)%N, (m-l)%M] \
                            * np.exp(-1j*2*np.pi*k/N) \
                            * np.exp(1j*2*np.pi*(m-l)*k/(M*N))

    
    return V


def permute_dft(N: int):
    """
    Method to generate permutation matrix to center the zero frequency.
    
    Parameters
    ----------
    N : int
        Length of the DFT.
    
    Returns
    -------
    P : np.ndarray
        Permutation matrix for DFT.
    """

    n = int(N/2)
    P = np.vstack((
        np.hstack((np.zeros((n, n)), np.eye(n))),
        np.hstack((np.eye(n), np.zeros((n, n))))
    ))

    return P


def print_info(targets, **kwargs):
    """Method to print information relative to the simulation."""

    M = kwargs['M']  # Number of delay bins
    N = kwargs['N']  # Number of Doppler bins
    P = kwargs['P']  # Number of targets
    delta_f = kwargs['Df'] * 1e3  # Distance between subcarriers
    fc = kwargs['fc'] * 1e9  # Carrier frequency
    T = 1/delta_f  # Block duration
    BW = M*delta_f  # Bandwidth
    resolution_Doppler_hz = 1/(T*N)
    resolution_Doppler_ms = speed_of_light/(N*T*fc)
    delay_res = 1/(delta_f*M)
    delay_res_m = speed_of_light/(delta_f*M)
    max_Doppler = resolution_Doppler_hz*N/2
    max_speed = resolution_Doppler_ms*N/2  # It goes both ways (pos and neg)
    max_delay = delay_res*M
    max_distance = delay_res_m*M/2

    print('\n\nRADAR-OTFS SIMULATION:')
    print('------------------------\n')
    print('-> Definitions:')
    print('----------------')
    print(f'N = {N} subcarriers')
    print(f'M = {M} blocks')
    print(f'fc = {fc*1e-9:.2f} GHz')
    print(f'delta_f = {delta_f*1e-3:.2f} kHz')
    print(f'BW = {BW*1e-6:.4f} MHz')
    print(f'T = {T*1e6:.4f} µs')
    print(f'Ts = {(1/BW)*1e6:.4f} µs')
    print(f'Doppler Res.: {resolution_Doppler_hz:.2f} Hz'
          + f' | {resolution_Doppler_ms:.2f} m/s'
          + f' | {resolution_Doppler_ms*3.6:.2f} km/h')
    print(f'Max. Doppler: {max_Doppler*1e-3:.2f} kHz | {max_speed:.2f} m/s'
          + f' | {max_speed*3.6:.2f} km/h')
    print(f'delay Res.: {delay_res*1e6:.2f} µs | {delay_res_m:.2f} m')
    print(f'Max. delay: {max_delay*1e6:.2f} µs | {max_distance:.2f} m')
    print('\n-> Targets:')
    print('----------------')
    for i in range(P):
        coeff_ref = targets[i, 0]
        dist_i = targets[i, 1]
        speed_i = targets[i, 2]
        print(f'--> Target i={i}:')
        print(f'  - Reflection Coeff.: {coeff_ref:.2f}')
        print(f'  - delay: {(2*dist_i/speed_of_light)*1e6:.2f} µs '
              + f'| {dist_i:.2f} m')
        print(f'  - Doppler: {(speed_i/3.6)*fc/speed_of_light:.2f} Hz | '
              + f'{speed_i/3.6:.2f} m/s | {speed_i:.2f} km/h')
    print('\n')


@njit
def correlation_periodic(X: np.ndarray, x_i: np.ndarray):
    """
    Method to recover delay infroamtion using a periodic/cyclic
    correlation.

    Parameters
    ----------
    X : np.ndarray
        Matrix in delay-Doppler domain with inforamtion we want to
        retrieve
    x_i : np.ndarray
        Transmitted sequence
    
    Returns
    -------
    Y : np.ndarray
        Matrix in delay-Doppler with retrieved information
    """

    M, N = X.shape
    seq_len = len(x_i)
    if seq_len % 2 == 0:
        cyc_len = int(seq_len/2)
    else:
        cyc_len = int((seq_len-1)/2)
    X_extended = np.vstack((X[-cyc_len:, :], X, X[:cyc_len, :]))
    Y = np.zeros((M, N), dtype=np.complex128)
    for n in range(N):
        for m in range(M):
            tdl = X_extended[m:m+seq_len, n]
            Y[m, n] = np.dot(tdl, np.conj(x_i))
    
    return Y


def mf_mat(x: np.ndarray):
    """"""
    M = len(x)
    X = np.zeros((M, M), dtype=np.complex128)
    for m in range(M):
        X[m, :] = np.roll(np.conj(x), m)
    
    return X


def min_max_norm(x: np.ndarray):
    return (x - np.min(x.flatten()))/np.max(x.flatten() - np.min(x.flatten()))


def gen_approx_mat(X: np.ndarray, K: int, method='rnd-l1', seed=None):
    """
    Parameters
    ----------
    X : np.ndarray
        Array containing matrix of information.
    K : int
        Number of elements to be considered in matrix product.
    method : str
        String with method of approximation (rnd, det1, det2).
        `rnd` -> 
    """

    M, _ = X.shape

    match method:
        case 'rnd':
            rng = np.random.default_rng(seed=seed)
            Q = np.zeros((K, M))
            m = rng.choice(M, (K,))
            Q[range(K), m] = np.sqrt(M/K)
        case 'rnd-l1':
            rng = np.random.default_rng(seed=seed)
            l1_norm = np.linalg.norm(X, ord=1, axis=1)
            p_l1 = l1_norm / np.sum(l1_norm)
            Q = np.zeros((K, M))
            m = rng.choice(M, (K,), p=p_l1)
            Q[range(K), m] = 1/np.sqrt(K*p_l1[m])
        case 'rnd-l2':
            rng = np.random.default_rng(seed=seed)
            l2_norm = np.linalg.norm(X, ord=2, axis=1)**2
            p_l2 = l2_norm / np.sum(l2_norm)
            Q = np.zeros((K, M))
            m = rng.choice(M, (K,), p=p_l2)
            Q[range(K), m] = 1/np.sqrt(K*p_l2[m])
        case 'det-l1':
            Q = np.zeros((K, M))
            l1_norm = np.linalg.norm(X, ord=1, axis=1)
            p_l1 = l1_norm / np.sum(l1_norm)
            ind = np.argsort(-l1_norm)[:K]
            Q[range(K), ind[range(K)]] = 1/np.sqrt(K*p_l1[ind[range(K)]])
        case 'det-l2':
            Q = np.zeros((K, M))
            l2_norm = np.linalg.norm(X, ord=2, axis=1)**2
            p_l2 = l2_norm / np.sum(l2_norm)
            ind = np.argsort(-l2_norm)[:K]
            Q[range(K), ind[range(K)]] = 1/np.sqrt(K*p_l2[ind[range(K)]])
    
    return Q


def find_targets(X: np.ndarray, threshold: float):
    """
    Method to estimate target position in DD domain according to a given
    threshold.
    
    Parameters
    ----------
    X : np.ndarray
        Input array with normalized post matched filter resulting
        matrix.
    threshold : float
        Threshold to test for targets.
    """
    
    return np.nonzero(X >= threshold)


def adjust_delay_position(Y_data: np.ndarray, seq_len: int):
    """Method to adjust the delay position.
    
    Parameters
    ----------
    Y_data : np.ndarray
    seq_len : int
    
    Returns
    -------
    Y_data_adjusted : np.ndarray
    """

    Y_data_adjusted = np.roll(Y_data, -int(seq_len/2), axis=0) \
        if seq_len % 2 == 0 else np.roll(Y_data, -int((seq_len-1)/2), axis=0)

    return Y_data_adjusted


def classify_targets(idx_data_pred: tuple, idx_data_true: tuple, M: int, N: int,
                     P: int):
    """Method to classify if targets were well classified or not.
    
    Parameters
    ----------
    idx_data_pred : tuple
    idx_data_true : tuple
    M : int
    N : int
    P : int
    
    Returns
    -------
    num_tp : int
    num_fp : int
    num_tn : int
    num_fn : int
    """

    idx_delay_pred, idx_Doppler_pred = idx_data_pred
    idx_delay_true, idx_Doppler_true = idx_data_true
    total_num_targets = len(idx_delay_pred)
    # All negative:
    num_tp = 0
    num_fp = 0
    num_fn = P
    num_tn = M*N - P
    if total_num_targets == 0:

        return np.array((num_tp, num_fp, num_tn, num_fn))

    true_points = list(zip(idx_delay_true, idx_Doppler_true))
    pred_points = list(zip(idx_delay_pred, idx_Doppler_pred))
    num_tp = sum([1 if point in pred_points else 0 for point in true_points])
    num_fn -= num_tp
    num_fp = len(pred_points) - num_tp
    num_tn -= num_fp

    return np.array((num_tp, num_fp, num_tn, num_fn))


def estimate_pd_pf(res: np.ndarray):
    """Method to estimate the probability of detection and false alarm.
    
    Parameters
    ----------
    res : np.ndarray
    
    Returns
    -------
    pd : float
        Probability of detection, #TP/(#TP + #FN)
    pf : float
        Probability of false alarm, #FP/(#FP + #TN)
    """

    pd = res[0]/(res[0]+res[3])
    pf = res[1]/(res[1]+res[2])

    return (pd, pf)


def estimate_f_g1(res: np.ndarray, M: int, N: int, P: int):
    """
    Method to estimate the F1 considering the unbalance between classes.
    """
    
    beta = P/(M*N)
    precision = res[0]/(res[0]+res[1])
    recall = res[0]/P
    precision_g = (precision - beta)/((1-beta)*precision) if precision >= beta \
        else 0
    recall_g = (recall - beta)/((1-beta)*recall) if recall >= beta else 0
    f_g1 = 2*precision_g*recall_g/(precision_g+recall_g) \
        if precision_g+recall_g != 0 else 0

    return f_g1


def periodogram(x, N):
    """Method to estimate the PSD."""

    num_winds = len(x) // N
    XX = np.zeros((N,), dtype=np.float64)
    for it in range(num_winds):
        XX += np.abs(np.fft.fftshift(np.fft.fft(x[int(it*N):int((it+1)*N)], N)))**2

    return XX


def add_offset(M, offset):
    """Matrix to add offset in frequency"""

    half_M = int(M/2)
    offset_mat = np.block([
        [np.eye(half_M), np.zeros((half_M, half_M))],
        [np.zeros((offset, M))],
        [np.zeros((half_M, half_M)), np.eye(half_M)]
    ])

    return offset_mat


def interp1d_curve(pf: np.ndarray, pd: np.ndarray, pf_axis: np.ndarray):
    """Linear interpolation of (pf, pd) along idx."""
    bacon = np.argsort(pf)
    spam = np.take_along_axis(pf, bacon, axis=0)
    eggs = np.take_along_axis(pd, bacon, axis=0)
    spam, sausage = np.unique(spam, return_index=True)

    return np.interp(pf_axis, spam, eggs[sausage])

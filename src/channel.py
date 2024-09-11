"""channel.py

Script with methods to generate the channel.

luizfelipe.coelho@smt.ufrj.br
Mar 19, 2024
"""


import numpy as np
from scipy.constants import speed_of_light


def awgn(s: np.ndarray, snr_dB: float, seed=42) -> np.ndarray:
    """Add white Gaussian noise with adjusted power according to SNR.
    
    Parameters
    ----------
    s : np.ndarray
        Input signal
    snr_dB : float
        SNR in dB to adjuste noise power.
    
    Returns
    -------
    y : np.ndarray
        Output singal. Input signal with AWGN.
    """

    rng = np.random.default_rng(seed)
    Ps = np.dot(np.conj(s), s).real / len(s)
    Pn = Ps * 10 ** (-.1*snr_dB)
    n = rng.standard_normal(len(s)) + 1j*rng.standard_normal(len(s))
    n *= np.sqrt(Pn / (np.dot(np.conj(n), n).real/len(n)))

    return s+n


def tx2rx_passive(s: np.ndarray, targets: np.ndarray, M: int, N: int, T: float,
          fc: float, snr_dB: float, seed=42) -> np.ndarray:
    """From Tx to Rx considering the delays, and passive 
    
    Parameters
    ----------
    s : np.ndarray
        Transmitted signal.
    targets : np.ndarray
        Array with targets information. The following information should
        be distributed over axis=1. Reflection coefficient, target
        range in meters, and relative speed in km/h.
    M : int
        Number of elements in fast-time axis (delay bins).
    N : int
        Number of elements in slow-times axis (Doppler bins).
    T : float
        Block/Pulse duration in seconds.
    fc : float
        Carrier frequency in Hz.
    snr_dB : float
        SNR for AWGN adjustment.
    
    Returns
    -------
    r : np.ndarray
        Received signal.
    """
    
    rng = np.random.default_rng(seed)
    # Defitions:
    kmh2ms = lambda x: x/3.6  # Function to change km/h -> m/s
    P = targets.shape[0]  # Number of targets (FR:cibles)
    h_i = targets[:, 0]  # Reflection coefficient
    tau_i = 2*targets[:, 1]/speed_of_light  # Delay in seconds
    nu_i = kmh2ms(targets[:, 2])*fc/speed_of_light  # Doppler shift in Hz
    Ts = T/M
    t = np.arange(0, T*N, Ts)  # Time axis for the whole Tx
    # Memory Allocation:
    r = np.zeros((M*N,), dtype=np.complex128)
    # Computation:
    for i in range(P):
        delay_samples = round(tau_i[i]/Ts)
        if delay_samples > 0:
            s_delayed = np.hstack((np.zeros((delay_samples,)),
                                   s[:-delay_samples]))
        else:
            s_delayed = s
        r += np.exp(1j*2*np.pi*rng.standard_normal(1)) \
            * np.exp(2*1j*np.pi*nu_i[i]*(t - tau_i[i])) * s_delayed

    return awgn(r, snr_dB, rng.integers(0, 100000))


def tx2rx(s: np.ndarray, targets: np.ndarray, M: int, N: int, T: float,
          fc: float, snr_dB: float, seed=42) -> np.ndarray:
    """From Tx to Rx considering the delays.
    
    Parameters
    ----------
    s : np.ndarray
        Transmitted signal.
    targets : np.ndarray
        Array with targets information. The following information should
        be distributed over axis=1. Reflection coefficient, target
        range in meters, and relative speed in km/h.
    M : int
        Number of elements in fast-time axis (delay bins).
    N : int
        Number of elements in slow-times axis (Doppler bins).
    T : float
        Block/Pulse duration in seconds.
    fc : float
        Carrier frequency in Hz.
    snr_dB : float
        SNR for AWGN adjustment.
    
    Returns
    -------
    r : np.ndarray
        Received signal.
    """
    
    rng = np.random.default_rng(seed)
    # Defitions:
    kmh2ms = lambda x: x/3.6  # Function to change km/h -> m/s
    P = targets.shape[0]  # Number of targets (FR:cibles)
    h_i = targets[:, 0]  # Reflection coefficient
    tau_i = 2*targets[:, 1]/speed_of_light  # Delay in seconds
    nu_i = kmh2ms(targets[:, 2])*fc/speed_of_light  # Doppler shift in Hz
    Ts = T/M
    t = np.arange(0, T*N, Ts)  # Time axis for the whole Tx
    # Memory Allocation:
    r = np.zeros((M*N,), dtype=np.complex128)
    # Computation:
    for i in range(P):
        delay_samples = round(tau_i[i]/Ts)
        if delay_samples > 0:
            s_delayed = np.hstack((np.zeros((delay_samples,)),
                                   s[:-delay_samples]))
        else:
            s_delayed = s
        r += 1 * (rng.standard_normal(1) + 1j*rng.standard_normal(1)) \
            * np.exp(2j*np.pi*nu_i[i]*(t - tau_i[i])) * s_delayed

    return awgn(r, snr_dB, rng.integers(0, 100000))


def gen_targets(seed=42, **kwargs):
    """Method to generate our targets.
    
    Paramereters
    ------------
    seed : int
        Number to set RNG seed.
    **kwarg : dict
        Dictionary that maps keyword arguments.
        - keys:
            M : int
                Number of delay bins (fast-time)
            N : int
                Number of Doppler bins (slow-time)
            P : int
                Number of targets
            Df : float
                Subcarrier spacing in kHz
            fc : float
                Carrier frequency in GHz
    
    Returns
    -------
    targets : np.ndarray
        2D array containing target information, as in
            targets[target_id, 0] = reflection coefficient of target_id
            targets[target_id, 1] = range of target_id, in m
            targets[target_id, 2] = velocity of target_id, in km/h
    """

    # Read keyword arguments:
    num_delay_bins = kwargs['M']  # fast-time
    num_Doppler_bins = kwargs['N']  # slow-time
    num_targets = kwargs['P']
    subcarr_spacing = kwargs['Df'] * 1e3
    carrier_freq = kwargs['fc'] * 1e9

    # Calculate system parameters:
    # bandwidth = num_delay_bins*subcarr_spacing
    # sampling_period = 1/bandwidth
    block_duration = 1/subcarr_spacing
    max_delay = block_duration
    max_range = speed_of_light*max_delay/2
    max_Doppler_negative = subcarr_spacing/2
    max_Doppler_positive = subcarr_spacing/2 - subcarr_spacing/num_Doppler_bins
    min_velocity = -3.6*speed_of_light*max_Doppler_negative/carrier_freq
    max_velocity = 3.6*speed_of_light*max_Doppler_positive/carrier_freq

    # Generate random target parameters:
    rng = np.random.default_rng(seed=seed)  # Set seed
    reflection_coefficients = rng.random((num_targets, 1))  # [0, 1)
    reflection_coefficients /= max(reflection_coefficients)  # Normalize
    target_range = max_range*rng.random((num_targets, 1))
    target_velocity = (max_velocity-min_velocity)*rng.random((num_targets, 1)) \
        + min_velocity
    # target_velocity = max_velocity*3.6*(2*(rng.random((num_targets, 1))-.5))
    targets = np.hstack((reflection_coefficients, target_range,
                         target_velocity))

    # Get position in delay-Doppler matrix:
    target_delay_idx = np.zeros((num_targets,))
    target_Doppler_idx = np.zeros((num_targets,))
    
    def find_Doppler_idx(targ_speed, min_speed, max_speed, N):

        speed_axis = np.linspace(min_speed, max_speed, N)
        euc_dist = np.sqrt((speed_axis - targ_speed)**2)
        idx = np.argmin(euc_dist)

        return idx
    
    for i in range(num_targets):
        target_delay_idx[i] = np.floor(target_range[i, 0]
                                       / (max_range/num_delay_bins))
        # target_Doppler_idx[i] = int(num_Doppler_bins/2) \
        #     + np.floor(target_velocity[i, 0]
        #                / (3.6*max_velocity/(num_Doppler_bins/2)))
        target_Doppler_idx[i] = find_Doppler_idx(
            target_velocity[i, 0], min_velocity, max_velocity, num_Doppler_bins
        )

    return targets, (target_delay_idx, target_Doppler_idx)

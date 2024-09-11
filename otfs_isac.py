"""randomized_sampling_otfs_isac.py

luizfelipe.coelho@smt.ufrj.br
may 20, 2024
"""


import argparse
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from matplotlib import cm
from scipy.constants import speed_of_light as c
from tqdm import tqdm
from src.channel import gen_targets, tx2rx, awgn
from src.otfs import idft_mat, dft_mat
from src.utils import (mf_mat, min_max_norm, zadoff_chu, periodogram,
                       add_offset, gen_approx_mat, print_info, permute_dft,
                       find_targets, classify_targets, estimate_pd_pf,
                       interp1d_curve, chirp, barker_code)


plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'mathptmx',
    'font.size': 8
})


def arg_parser():
    parser = argparse.ArgumentParser()
    help_M = 'Number of elements in fast-time. It defines the number of bins' \
        + ' in the range axis of the radar system and the number of bins in ' \
        + 'the delay axis of the OTFS (number of sub-carriers)'
    parser.add_argument('--M', type=int, default=256, help=help_M)
    help_N = 'Number of elements in slow-time. It defines the number of bins' \
        + ' in the Doppler axis of the radar system and the OTFS (number of ' \
        + 'time slots)'
    parser.add_argument('--N', type=int, default=64, help=help_N)
    parser.add_argument('--P', type=int, default=1, help='Number of targets')
    parser.add_argument('--L', type=int, default=16,
                        help='Number of selected data')
    parser.add_argument('--fc', type=float, default=24,
                        help='Carrier frequency in GHz.')
    parser.add_argument('--Df', type=float, default=3,
                        help='Subcarrier spacing in kHz.')
    parser.add_argument('--snr_lim', type=str, default='-15,21',
                        help='SNR limits for simulation.')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--ensemble', type=int, default=1000)
    parser.add_argument('-m', '--mode', type=str, default='sim',
                        choices=['sim', 'psd', 'test'])
    parser.add_argument('--default_seed', action=argparse.BooleanOptionalAction,
                        help='Sets a default seed value (42).')
    parser.add_argument('--figures', action=argparse.BooleanOptionalAction,
                        help='Will skip simulation and generate figures.')
    args = parser.parse_args()

    return args


def worker(setup: tuple):
    """A funtion to handle parallel processing."""

    idx, snr, threshold, ensemble, M, N, P, Df, fc, seed = setup

    rng = np.random.default_rng(seed=seed)

    # Transmitter:
    x_zc = zadoff_chu(M)
    x_chirp = chirp(1/Df, 1/(Df*M), M*Df, 1, M)
    bc_len = 13
    x_bc = barker_code(bc_len)
    X_zc = np.zeros((M, N), dtype=np.complex128)
    X_chirp = np.zeros((M, N), dtype=np.complex128)
    X_bc  = np.zeros((M, N), dtype=np.complex128)
    X_zc[:, 0] = x_zc
    X_chirp[:, 0] = x_chirp
    X_bc[:bc_len, 0] = x_bc
    X_zc_dt = X_zc @ idft_mat(N)
    X_chirp_dt = X_chirp @ idft_mat(N)
    X_bc_dt = X_bc @ idft_mat(N)
    x_zc_t = X_zc_dt.T.flatten()
    x_chirp_t = X_chirp_dt.T.flatten()
    x_bc_t = X_bc_dt.T.flatten()
    X_zc_corr = mf_mat(X_zc[:, 0])
    X_chirp_corr = mf_mat(X_chirp[:, 0])
    X_bc_corr = mf_mat(X_bc[:, 0])
    # Memory allocation
    pd = np.zeros((ensemble, 3))
    pf = np.zeros((ensemble, 3))
    for it in range(ensemble):
        # Channel:
        spamseed = rng.integers(99999999)
        targets, targets_pos = gen_targets(seed=spamseed, M=M, N=N, P=P,
                                           Df=Df*1e-3, fc=fc*1e-9)
        r_zc_t = tx2rx(x_zc_t, targets, M, N, T, fc, snr, spamseed)
        r_chirp_t = tx2rx(x_chirp_t, targets, M, N, T, fc, snr, spamseed)
        r_bc_t = tx2rx(x_bc_t, targets, M, N, T, fc, snr, spamseed)
        # Receiver:
        Y_zc_dt = r_zc_t.reshape((N, M)).T
        Y_chirp_dt = r_chirp_t.reshape((N, M)).T
        Y_bc_dt = r_bc_t.reshape((N, M)).T
        Y_zc_dd = Y_zc_dt @ dft_mat(N)
        Y_chirp_dd = Y_chirp_dt @ dft_mat(N)
        Y_bc_dd = Y_bc_dt @ dft_mat(N)
        # Correlation and normalization:
        Y_zc = np.abs(X_zc_corr @ Y_zc_dd) @ permute_dft(N) / M
        Y_chirp = np.abs(X_chirp_corr @ Y_chirp_dd) @ permute_dft(N) / M
        Y_bc = np.abs(X_bc_corr @ Y_bc_dd) @ permute_dft(N) / bc_len
        # Classify Targets
        res_zc = classify_targets(find_targets(Y_zc, threshold),
                                  targets_pos, M, N, P)
        res_chirp = classify_targets(find_targets(Y_chirp, threshold),
                                     targets_pos, M, N, P)
        res_bc = classify_targets(find_targets(Y_bc, threshold),
                                  targets_pos, M, N, P)
        # Estimate Probabilities
        pd[it, 0], pf[it, 0] = estimate_pd_pf(res_zc)
        pd[it, 1], pf[it, 1] = estimate_pd_pf(res_chirp)
        pd[it, 2], pf[it, 2] = estimate_pd_pf(res_bc)
    # Average Resutls
    pd_avg = np.mean(pd, axis=0)
    pf_avg = np.mean(pf, axis=0)

    return (idx, pd_avg, pf_avg)


if __name__ == '__main__':
    
    args = arg_parser()

    Df = args.Df * 1e3  # Subcarrier spacing
    fc = args.fc * 1e9  # Carrier frequency
    M = args.M  # delay axis
    N = args.N  # Doppler axis
    P = args.P  # Number of targets
    T = 1/Df  # Block duration
    Ts = T/M  # Sampling period
    if args.default_seed:
        rng = np.random.default_rng(seed=42)
    else:
        rng = np.random.default_rng()

    match args.mode:
        case 'test':

            X0 = np.zeros((M, N), dtype=np.complex128)
            X0[:, 2] = zadoff_chu(M)
            X0[:, 25] = np.roll(zadoff_chu(M), 50)
            X0_corr = mf_mat(X0[:, 2])
            Y0 = np.abs(X0_corr @ X0)
            Q0_rand = gen_approx_mat(X0, L, 'rnd')
            Y0_rand = np.abs(X0_corr @ Q0_rand.T @ Q0_rand @ X0)
            Q0_rand_l1 = gen_approx_mat(X0, L, 'rnd-l1')
            Y0_rand_l1 = np.abs(X0_corr @ Q0_rand_l1.T @ Q0_rand_l1 @ X0)
            Q0_rand_l2 = gen_approx_mat(X0, L, 'rnd-l2')
            Y0_rand_l2 = np.abs(X0_corr @ Q0_rand_l2.T @ Q0_rand_l2 @ X0)
            Q0_det_l1 = gen_approx_mat(X0, L, 'det-l1')
            Y0_det_l1 = np.abs(X0_corr @ Q0_det_l1.T @ Q0_det_l1 @ X0)
            Q0_det_l2 = gen_approx_mat(X0, L, 'det-l2')
            Y0_det_l2 = np.abs(X0_corr @ Q0_det_l2.T @ Q0_det_l2 @ X0)

            print('Full dynamic')
            print(f'full: {np.max(Y0)}')
            print(f'rand: {np.max(Y0_rand)}')
            print(f'rand-l1: {np.max(Y0_rand_l1)}')
            print(f'rand-l2: {np.max(Y0_rand_l2)}')
            print(f'det-l1: {np.max(Y0_det_l1)}')
            print(f'det-l2: {np.max(Y0_det_l2)}')

            seq_len = 32
            X1 = np.zeros((M, N), dtype=np.complex128)
            X1[:seq_len, 2] = zadoff_chu(seq_len)
            X1[50:50+seq_len, 25] = zadoff_chu(seq_len)
            X1_corr = mf_mat(X1[:, 2])
            Y1 = np.abs(X1_corr @ X1)
            Q1_rand = gen_approx_mat(X1, L, 'rnd')
            Y1_rand = np.abs(X1_corr @ Q1_rand.T @ Q1_rand @ X1)
            Q1_rand_l1 = gen_approx_mat(X1, L, 'rnd-l1')
            Y1_rand_l1 = np.abs(X1_corr @ Q1_rand_l1.T @ Q1_rand_l1 @ X1)
            Q1_rand_l2 = gen_approx_mat(X1, L, 'rnd-l2')
            Y1_rand_l2 = np.abs(X1_corr @ Q1_rand_l2.T @ Q1_rand_l2 @ X1)
            Q1_det_l1 = gen_approx_mat(X1, L, 'det-l1')
            Y1_det_l1 = np.abs(X1_corr @ Q1_det_l1.T @ Q1_det_l1 @ X1)
            Q1_det_l2 = gen_approx_mat(X1, L, 'det-l2')
            Y1_det_l2 = np.abs(X1_corr @ Q1_det_l2.T @ Q1_det_l2 @ X1)

            print(f'{seq_len}/{M} dynamic : {M/seq_len}')
            print(f'full: {np.max(Y1)}')
            print(f'rand: {np.max(Y1_rand)}')
            print(f'rand-l1: {np.max(Y1_rand_l1)}')
            print(f'rand-l2: {np.max(Y1_rand_l2)}')
            print(f'det-l1: {np.max(Y1_det_l1)}')
            print(f'det-l2: {np.max(Y1_det_l2)}')

        case 'sim':
            
            # Make folders:
            fig_folder = 'figures/'
            os.makedirs(fig_folder, exist_ok=True)
            res_folder = 'results/'
            os.makedirs(res_folder, exist_ok=True)
            res_path = os.path.join(res_folder, f'results_{P}_targets.npz')
            
            # Generate the axes
            min_Doppler = -Df/2  # Hz
            max_Doppler = Df/2 - Df/N
            min_velocity = c*min_Doppler/fc  # m/s
            max_velocity = c*max_Doppler/fc
            max_delay = T  # s
            max_range = c*max_delay/2  # m
            mesh_Doppler, mesh_delay = np.meshgrid(
                3.6*np.linspace(min_velocity, max_velocity, N),
                np.linspace(0, max_range, M)
            )

            sausage = [int(val) for val in args.snr_lim.split(',')]
            snrs = np.arange(*sausage, 5)
            thresholds = np.logspace(-5, np.log10(3), 50)
            pd = np.zeros((3, len(thresholds), len(snrs)))
            pf = np.zeros((3, len(thresholds), len(snrs)))
            if not args.figures:
                idx_list = list(itertools.product(range(len(snrs)),
                                                  range(len(thresholds))))
                setup_list = [(idx, snrs[idx[0]], thresholds[idx[1]],
                               args.ensemble, M, N, P, Df, fc,
                               rng.integers(99999999)) for idx in idx_list]
                with Pool(cpu_count()) as p:
                    for data in tqdm(p.imap_unordered(worker, setup_list),
                                     total=len(setup_list)):
                        idx, pd_avg, pf_avg = data
                        pd[:, idx[1], idx[0]] = pd_avg
                        pf[:, idx[1], idx[0]] = pf_avg

                np.savez(res_path, pd, pf)

            else:

                res_file = np.load(res_path)
                pd = res_file['arr_0']
                pf = res_file['arr_1']
                # Sort for increasing false alarm rate, drop duplicates, and
                # interpolate data.
                pf_axis = np.linspace(0, 1, 1000)
                pd_interp = np.zeros((3, len(pf_axis), len(snrs)))
                for idx in range(len(snrs)):
                    pd_interp[0, :, idx] = interp1d_curve(pf[0, :, idx],
                                                          pd[0, :, idx],
                                                          pf_axis)
                    pd_interp[1, :, idx] = interp1d_curve(pf[1, :, idx],
                                                          pd[1, :, idx],
                                                          pf_axis)
                    pd_interp[2, :, idx] = interp1d_curve(pf[2, :, idx],
                                                          pd[2, :, idx],
                                                          pf_axis)
                idx_pf = 5
                idx_snr = 3
                golden_ratio = (1 + 5**.5)/2
                width = 3.5
                height = width/golden_ratio
                print(f'Pf = {pf_axis[idx_pf]}')
                print(f'SNR = {snrs[idx_snr]}')

                # Figure 1 -- ROC
                fig1 = plt.figure(figsize=(width, height))
                ax = fig1.add_subplot(111)
                ax.plot(pf_axis, pd_interp[0, :, idx_snr], label='ZC')
                ax.plot(pf_axis, pd_interp[1, :, idx_snr], label='Chirp')
                ax.plot(pf_axis, pd_interp[2, :, idx_snr], label='Barker')
                ax.set_xlabel('$P_f$')
                ax.set_ylabel('$P_d$')
                ax.legend()
                fig1.tight_layout()

                # Figure 2 -- Pd vs. SNR
                fig2 = plt.figure(figsize=(width, height))
                ax = fig2.add_subplot(111)
                ax.plot(snrs, pd_interp[0, idx_pf, :], label='ZC')
                ax.plot(snrs, pd_interp[1, idx_pf, :], label='Chirp')
                ax.plot(snrs, pd_interp[2, idx_pf, :], label='Barker')
                ax.set_xlabel('SNR, dB')
                ax.set_ylabel('$P_d$')
                ax.legend()
                fig2.tight_layout()

                plt.show()

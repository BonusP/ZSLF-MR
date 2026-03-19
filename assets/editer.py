"""
assets/editer.py
EDITER k-space noise correction.
Exact Python port of the MATLAB EDITER algorithm.
"""

import numpy as np


def _padarray_2d(A: np.ndarray, pad_col: int, pad_lin: int) -> np.ndarray:
    return np.pad(A, ((pad_col, pad_col), (pad_lin, pad_lin)), mode="constant")


def editer_kspace_correction(
    datafft: np.ndarray,
    datanoise_fft_list: list,
    Nc: int = 5,
    ksz_col_init: int = 0,
    ksz_lin_init: int = 0,
    corr_thresh: float = 0.5,
    ksz_col_final: int = 7,
    ksz_lin_final: int = 0,
):
    """
    EDITER k-space correction.

    Args:
        datafft:            k-space data (ncol, nlin), complex128
        datanoise_fft_list: list of Nc noise k-space arrays, each (ncol, nlin)
        Nc:                 number of noise channels
        ksz_col_init:       initial kernel half-size along columns
        ksz_lin_init:       initial kernel half-size along lines
        corr_thresh:        correlation threshold for window grouping
        ksz_col_final:      final kernel half-size along columns
        ksz_lin_final:      final kernel half-size along lines

    Returns:
        gksp:       corrected k-space (ncol, nlin)
        kern_pe:    estimated kernels per line
        win_stack:  grouped line windows
        kcor:       kernel correlation matrix
        kcor_thresh: thresholded correlation matrix
    """
    ncol, nlin = datafft.shape
    assert len(datanoise_fft_list) == Nc, "Nc must match len(datanoise_fft_list)."

    # ── Initial pass: estimate kernels per PE line ────────────────────────────
    ksz_col = int(ksz_col_init)
    ksz_lin = int(ksz_lin_init)
    K = Nc * (2 * ksz_col + 1) * (2 * ksz_lin + 1)
    kern_pe = np.zeros((K, nlin), dtype=np.complex128)

    for clin in range(nlin):
        pe_rng = [clin]
        padded = [_padarray_2d(df[:, pe_rng], ksz_col, ksz_lin) for df in datanoise_fft_list]

        noise_slices = []
        for col_shift in range(-ksz_col, ksz_col + 1):
            for lin_shift in range(-ksz_lin, ksz_lin + 1):
                for ch in range(Nc):
                    dftmp = np.roll(padded[ch], shift=(col_shift, lin_shift), axis=(0, 1))
                    cropped = dftmp[
                        ksz_col : dftmp.shape[0] - ksz_col,
                        ksz_lin : dftmp.shape[1] - ksz_lin,
                    ]
                    noise_slices.append(cropped)

        noise_mat = np.stack(noise_slices, axis=2)
        gmat = noise_mat.reshape(-1, noise_mat.shape[2])
        b = datafft[:, pe_rng].reshape(-1)
        kern = np.linalg.lstsq(gmat, b, rcond=None)[0]
        kern_pe[:, clin] = kern

    # ── Kernel correlation ─────────────────────────────────────────────────────
    kern_pe_normalized = np.zeros_like(kern_pe)
    for clin in range(nlin):
        kern_pe_normalized[:, clin] = kern_pe[:, clin] / (np.linalg.norm(kern_pe[:, clin]) + 1e-12)

    kcor = kern_pe_normalized.conj().T @ kern_pe_normalized
    kcor_thresh = np.abs(kcor) > corr_thresh

    # ── Window stacking ────────────────────────────────────────────────────────
    aval_lins = list(range(nlin))
    win_stack = []
    while aval_lins:
        clin = min(aval_lins)
        row = kcor_thresh[clin, clin:]
        idx_true = np.where(row)[0]
        end_idx = clin + int(idx_true.max()) if idx_true.size > 0 else clin
        pe_rng = list(range(clin, end_idx + 1))
        win_stack.append(pe_rng)
        aval_lins = sorted(set(aval_lins) - set(pe_rng))

    # ── Final solve per window ─────────────────────────────────────────────────
    ksz_col = int(ksz_col_final)
    ksz_lin = int(ksz_lin_final)
    gksp = np.zeros((ncol, nlin), dtype=np.complex128)

    for pe_rng in win_stack:
        padded = [_padarray_2d(df[:, pe_rng], ksz_col, ksz_lin) for df in datanoise_fft_list]

        noise_slices = []
        for col_shift in range(-ksz_col, ksz_col + 1):
            for lin_shift in range(-ksz_lin, ksz_lin + 1):
                for ch in range(Nc):
                    dftmp = np.roll(padded[ch], shift=(col_shift, lin_shift), axis=(0, 1))
                    cropped = dftmp[
                        ksz_col : dftmp.shape[0] - ksz_col,
                        ksz_lin : dftmp.shape[1] - ksz_lin,
                    ]
                    noise_slices.append(cropped)

        noise_mat = np.stack(noise_slices, axis=2)
        gmat = noise_mat.reshape(-1, noise_mat.shape[2])
        init_mat_sub = datafft[:, pe_rng]
        b = init_mat_sub.reshape(-1)
        kern = np.linalg.lstsq(gmat, b, rcond=None)[0]
        tosub = (gmat @ kern).reshape(ncol, len(pe_rng))
        gksp[:, pe_rng] = init_mat_sub - tosub

    return gksp, kern_pe, win_stack, kcor, kcor_thresh

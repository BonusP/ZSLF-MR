"""
assets/utils.py
Shared math utilities: Fourier transforms, vectorize, view transform, image save.
Includes smart .mat loader that auto-detects k-space and noise channels.
"""

import numpy as np
from scipy.io import loadmat as _loadmat
import cv2
import os
from typing import Optional


# ─── Fourier Transforms ────────────────────────────────────────────────────────

def ft2(kdata: np.ndarray) -> np.ndarray:
    """2D DFT (MATLAB-style: centered, normalized)."""
    kdata = np.asarray(kdata)
    nx, ny = kdata.shape[0], kdata.shape[1]
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(kdata, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    ) / np.sqrt(nx * ny)


def ift2(data: np.ndarray) -> np.ndarray:
    """2D IDFT (MATLAB-style: centered, normalized)."""
    data = np.asarray(data)
    nx, ny = data.shape[0], data.shape[1]
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(data, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    ) * np.sqrt(nx * ny)


# ─── Array utilities ───────────────────────────────────────────────────────────

def vec(a: np.ndarray) -> np.ndarray:
    """Column-major vectorize (MATLAB vec)."""
    return np.asarray(a).reshape(-1, order="F")


def even(n: int) -> bool:
    return (n % 2) == 0


def rmse(in_arr: np.ndarray, true_arr: np.ndarray, use_abs: int = 0) -> float:
    in_arr = np.asarray(in_arr)
    true_arr = np.asarray(true_arr)
    if use_abs == 1:
        num = np.linalg.norm(np.abs(in_arr).ravel(order="F") - np.abs(true_arr).ravel(order="F"))
        den = np.linalg.norm(np.abs(true_arr).ravel(order="F")) + 1e-12
    else:
        num = np.linalg.norm(in_arr.ravel(order="F") - true_arr.ravel(order="F"))
        den = np.linalg.norm(true_arr.ravel(order="F")) + 1e-12
    return 100.0 * (num / den)


# ─── Display / Crop ────────────────────────────────────────────────────────────

def legacy_view(arr: np.ndarray, col_start: int = 165, col_end: int = 366) -> np.ndarray:
    """
    Crop + rotate to standard display orientation.
    Matches: abs(flip(rot90(arr, 1), 1))[:, col_start:col_end]
    """
    return (np.abs(np.flip(np.rot90(arr, 1), 1)))[:, col_start:col_end]


# ─── Image Save ────────────────────────────────────────────────────────────────

def save_image(arr: np.ndarray, path: str, is_ddnm: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if np.max(arr) > 0:
        if not is_ddnm:
            img = (arr / np.max(arr) * 255)
        else:
            img = arr
    else:
        img = np.zeros_like(arr, dtype=np.uint8)
    cv2.imwrite(path, img)


# ─── Smart .mat Loader ─────────────────────────────────────────────────────────

def load_mat_mri(path: str) -> tuple:
    """
    Load an MRI .mat file and auto-detect the k-space signal and noise channels.

    Detection rules:
      - Signal key:  the single key whose name does NOT contain 'noise'
                     (among the real data keys, i.e. not starting with '__')
      - Noise keys:  all remaining keys, sorted alphabetically

    Returns:
        datafft    (np.ndarray): k-space data, complex128
        noise_list (list):       list of noise channel arrays, complex128
    """
    mat = _loadmat(path)
    data_keys = sorted(k for k in mat.keys() if not k.startswith("__"))

    signal_keys = [k for k in data_keys if "noise" not in k.lower()]
    noise_keys  = sorted(k for k in data_keys if "noise" in k.lower())

    if len(signal_keys) != 1:
        raise ValueError(
            f"Expected exactly 1 signal key (no 'noise' in name), "
            f"found: {signal_keys}  (all keys: {data_keys})"
        )

    datafft    = np.asarray(mat[signal_keys[0]], dtype=np.complex128)
    noise_list = [np.asarray(mat[k], dtype=np.complex128) for k in noise_keys]

    print(f"  Loaded '{path}'")
    print(f"    signal key : '{signal_keys[0]}'  shape={datafft.shape}")
    print(f"    noise keys : {noise_keys}  ({len(noise_list)} channels)")

    return datafft, noise_list


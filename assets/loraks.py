"""
assets/loraks.py
Single-channel SENSE-LORAKS MRI reconstruction.
Combines LORAKS operators (cell 0) and CG solver + main loop (cell 1)
from EDITER+LORAKS.ipynb.
"""

import numpy as np
from typing import Optional
from scipy.sparse.linalg import cg, LinearOperator

from assets.utils import ft2, ift2, vec, even


# ─── SVD helper ───────────────────────────────────────────────────────────────

def svd_left(A: np.ndarray, r: Optional[int] = None) -> np.ndarray:
    """Compute left singular vectors via eigendecomposition of A @ A^H."""
    A = np.asarray(A)
    AhA = A @ A.conj().T
    if r is None:
        w, U = np.linalg.eigh(AhA)
        idx = np.argsort(np.abs(w))[::-1]
        return U[:, idx].astype(A.dtype, copy=False)
    else:
        try:
            from scipy.sparse.linalg import eigsh
            w, U = eigsh(AhA.astype(np.complex128), k=r, which="LM")
            idx = np.argsort(np.abs(w))[::-1]
            return U[:, idx].astype(A.dtype, copy=False)
        except Exception:
            w, U = np.linalg.eigh(AhA)
            idx = np.argsort(np.abs(w))[::-1][:r]
            return U[:, idx].astype(A.dtype, copy=False)


# ─── LORAKS fast filtering ─────────────────────────────────────────────────────

def filtfilt_loraks(ncc: np.ndarray, opt: str, N1: int, N2: int, Nc: int, R: int) -> np.ndarray:
    """Zero-phase LORAKS filter — exact port of MATLAB filtfilt.m."""
    ncc = np.asarray(ncc)
    fltlen = ncc.shape[1] // Nc
    numflt = ncc.shape[0]

    in1, in2 = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1), indexing="xy")
    mask = (in1**2 + in2**2) <= R**2
    in1v = in1[mask].astype(int).ravel()
    in2v = in2[mask].astype(int).ravel()

    H = 2 * R + 1
    rr0 = (R + in1v)
    cc0 = (R + in2v)
    ind0 = rr0 + cc0 * H

    F = np.zeros((H * H, Nc, numflt), dtype=ncc.dtype)
    tmp = ncc.T.reshape((fltlen, Nc, numflt), order="F")
    F[ind0, :, :] = tmp
    F = F.reshape((H, H, Nc, numflt), order="F")

    cfilt = np.conj(F)
    ffilt = np.conj(F) if opt == "S" else np.flip(np.flip(F, axis=0), axis=1)

    P = 4 * R + 1
    ccfilt = np.fft.fft2(cfilt, s=(P, P), axes=(0, 1))
    fffilt = np.fft.fft2(ffilt, s=(P, P), axes=(0, 1))

    a = ccfilt.reshape((P, P, 1, Nc, numflt), order="F")
    b = fffilt.reshape((P, P, Nc, 1, numflt), order="F")
    patch = np.fft.ifft2((a * b).sum(axis=4), axes=(0, 1))

    pad1 = N1 - 1 - 2 * R
    pad2 = N2 - 1 - 2 * R
    if pad1 < 0 or pad2 < 0:
        raise ValueError(f"Invalid padding: pad1={pad1}, pad2={pad2}. Check N1,N2,R.")

    patch_pad = np.pad(patch, ((0, pad1), (0, pad2), (0, 0), (0, 0)), mode="constant")

    if opt == "S":
        sh1 = -4 * R - (N1 % 2)
        sh2 = -4 * R - (N2 % 2)
    else:
        sh1 = -2 * R
        sh2 = -2 * R

    shifted = np.roll(patch_pad, shift=(sh1, sh2), axis=(0, 1))
    return np.fft.fft2(shifted, axes=(0, 1))


# ─── LORAKS operators ──────────────────────────────────────────────────────────

def _sub2ind_colmajor(N1: int, r0: np.ndarray, c0: np.ndarray) -> np.ndarray:
    return r0 + c0 * N1


def LORAKS_operators(x: np.ndarray, N1: int, N2: int, Nc: int,
                     R: int, LORAKS_type: int, weights=None) -> np.ndarray:
    """S-matrix LORAKS forward (+1) and adjoint (-1) operators."""
    x = np.asarray(x)

    in1, in2 = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1), indexing="xy")
    mask = (in1**2 + in2**2) <= R**2
    in1v = in1[mask].astype(int).ravel()
    in2v = in2[mask].astype(int).ravel()
    patchSize = in1v.size

    eN1 = int(even(N1))
    eN2 = int(even(N2))
    nPatch = (N1 - 2 * R - eN1) * (N2 - 2 * R - eN2)

    i0_start   = R + eN1
    i0_end_excl = N1 - R
    j0_start   = R + eN2
    j0_end_excl = N2 - R

    c1 = 2 * int(np.ceil((N1 - 1) / 2.0)) + 2
    c2 = 2 * int(np.ceil((N2 - 1) / 2.0)) + 2

    if LORAKS_type == 1:
        X = x.reshape((N1 * N2, Nc), order="F")
        out = np.zeros((patchSize, 2, Nc, nPatch * 2), dtype=X.dtype)
        k = 0
        for i0 in range(i0_start, i0_end_excl):
            for j0 in range(j0_start, j0_end_excl):
                rr = i0 + in1v
                cc = j0 + in2v
                Ind  = _sub2ind_colmajor(N1, rr, cc)
                rp0  = (-i0) + in1v + (c1 - 2)
                cp0  = (-j0) + in2v + (c2 - 2)
                Indp = _sub2ind_colmajor(N1, rp0, cp0)

                tmp = X[Ind, :] - X[Indp, :]
                out[:, 0, :, k]         = np.real(tmp)
                out[:, 1, :, k]         = -np.imag(tmp)

                tmp = X[Ind, :] + X[Indp, :]
                out[:, 0, :, k + nPatch] = np.imag(tmp)
                out[:, 1, :, k + nPatch] = np.real(tmp)
                k += 1

        return out.reshape((patchSize * Nc * 2, nPatch * 2), order="F")

    elif LORAKS_type == -1:
        X = x.reshape((patchSize * 2, Nc, nPatch * 2), order="F")
        res = np.zeros((N1 * N2, Nc), dtype=X.dtype)
        k = 0
        for i0 in range(i0_start, i0_end_excl):
            for j0 in range(j0_start, j0_end_excl):
                rr   = i0 + in1v
                cc   = j0 + in2v
                Ind  = _sub2ind_colmajor(N1, rr, cc)
                rp0  = (-i0) + in1v + (c1 - 2)
                cp0  = (-j0) + in2v + (c2 - 2)
                Indp = _sub2ind_colmajor(N1, rp0, cp0)

                a = X[0:patchSize, :, k]
                b = X[patchSize:2 * patchSize, :, nPatch + k]
                c = X[0:patchSize, :, nPatch + k]
                d = X[patchSize:2 * patchSize, :, k]

                res[Ind,  :] += (a + b) + 1j * (c - d)
                res[Indp, :] += (-a + b) + 1j * (c + d)
                k += 1
        return vec(res)

    else:
        raise ValueError("LORAKS_type must be +1 or -1.")


# ─── CG solver ────────────────────────────────────────────────────────────────

def cg_solve(A_mul, b, x0=None, maxiter: int = 200, tol: float = 1e-5):
    """Conjugate Gradient solver (scipy CG with numpy fallback)."""
    b = np.asarray(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    try:
        n = b.size
        Aop = LinearOperator((n, n), matvec=A_mul, dtype=b.dtype)
        x, info = cg(Aop, b, x0=x0, maxiter=maxiter, tol=tol)
        return x, info
    except Exception:
        x = x0.copy()
        r = b - A_mul(x)
        p = r.copy()
        rsold = np.vdot(r, r)
        for _ in range(maxiter):
            Ap = A_mul(p)
            alpha = rsold / (np.vdot(p, Ap) + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = np.vdot(r, r)
            if np.sqrt(rsnew.real) < tol:
                return x, 0
            p = r + (rsnew / (rsold + 1e-12)) * p
            rsold = rsnew
        return x, 1


# ─── Main LORAKS reconstruction ────────────────────────────────────────────────

def sense_loraks_single_channel(
    corr_img: np.ndarray,
    R: int = 5,
    rank: int = 50,
    lam: float = 1e-2,
    tol: float = 1e-3,
    max_iter: int = 2,
    LORAKS_type: int = 1,
    cg_maxiter: int = 200,
    cg_tol: float = 1e-5,
) -> np.ndarray:
    """
    Single-channel SENSE-LORAKS reconstruction.

    Args:
        corr_img:    complex image (nx, ny), e.g. ift2(gksp) from EDITER
        R:           LORAKS patch radius
        rank:        truncation rank for low-rank approximation
        lam:         regularization weight
        tol:         convergence tolerance (outer loop)
        max_iter:    number of outer iterations
        LORAKS_type: 1 for S-matrix
        cg_maxiter:  CG max iterations
        cg_tol:      CG convergence tolerance

    Returns:
        recon: (nx, ny) complex reconstructed image
    """
    corr_img = np.asarray(corr_img)
    nx, ny = corr_img.shape
    Nc = 1  # single-channel

    z    = vec(corr_img)
    Ahd  = vec(corr_img)

    def AhA(x): return x

    def B(x):
        return vec(ft2(np.asarray(x).reshape((nx, ny), order="F")))

    def Bh(x):
        return vec(ift2(np.asarray(x).reshape((nx, ny), order="F")))

    def ZD(x):
        X = np.asarray(x).reshape((nx, ny, 1), order="F")
        return np.pad(X, ((0, 2*R), (0, 2*R), (0, 0)), mode="constant")

    def ZD_H(x):
        return x[:nx, :ny, ...]

    in1, in2 = np.meshgrid(np.arange(-R, R+1), np.arange(-R, R+1), indexing="xy")
    patchSize = np.where(in1**2 + in2**2 <= R**2)[0].size

    for _ in range(max_iter):
        z_prev = z.copy()

        MM  = LORAKS_operators(B(z), nx, ny, Nc, R, LORAKS_type)
        Um  = svd_left(MM)
        nmm = Um[:, rank:].conj().T

        nf = nmm.shape[0]
        nmm_r = nmm.reshape((nf, patchSize, 2 * Nc), order="F")
        nss_h = (nmm_r[:, :, 0::2] + 1j * nmm_r[:, :, 1::2]).reshape((nf, patchSize * Nc), order="F")

        Nis  = filtfilt_loraks(nss_h, "C", nx, ny, Nc, R)
        Nis2 = filtfilt_loraks(nss_h, "S", nx, ny, Nc, R)

        def LhL(x):
            bx  = B(x)
            Z   = ZD(bx)
            FZ  = np.fft.fft2(Z, axes=(0, 1))
            FZ_rep = FZ[:, :, :, None]
            s1  = np.sum(Nis  * FZ_rep, axis=2)
            s2  = np.sum(Nis2 * np.conj(FZ_rep), axis=2)
            t1  = np.fft.ifft2(s1, axes=(0, 1))
            t2  = np.fft.ifft2(s2, axes=(0, 1))
            d   = ZD_H(t1) - ZD_H(t2)
            return 2.0 * Bh(vec(d[:, :, 0]))

        def M_mul(x):
            return AhA(x) + lam * LhL(x)

        z, _ = cg_solve(M_mul, Ahd, x0=z, maxiter=cg_maxiter, tol=cg_tol)

        if np.linalg.norm(z_prev - z) / (np.linalg.norm(z) + 1e-12) < tol:
            break

    return np.asarray(z).reshape((nx, ny), order="F")

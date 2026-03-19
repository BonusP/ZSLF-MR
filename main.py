import os
import sys
import numpy as np
import yaml
import cv2
import argparse
from scipy.io import loadmat

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from assets.utils    import ft2, ift2, legacy_view, save_image, load_mat_mri
from assets.editer   import editer_kspace_correction
from assets.loraks   import sense_loraks_single_channel
from assets.ddnm.diffusion import DDNMDiffusion
from assets.Evaluation import gt_eval

# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_mat_data(mat_path: str) -> tuple:
    """Auto-detect k-space and noise channels from .mat file."""
    return load_mat_mri(mat_path)


def run_editer(datafft: np.ndarray, noise_list: list, cfg: dict) -> np.ndarray:
    """Run EDITER k-space correction. Returns corrected image (complex)."""
    ec = cfg["editer"]
    gksp, *_ = editer_kspace_correction(
        datafft=datafft,
        datanoise_fft_list=noise_list,
        Nc=ec["Nc"],
        ksz_col_init=ec["ksz_col_init"],
        ksz_lin_init=ec["ksz_lin_init"],
        corr_thresh=ec["corr_thresh"],
        ksz_col_final=ec["ksz_col_final"],
        ksz_lin_final=ec["ksz_lin_final"],
    )
    return ift2(gksp)   # complex image


def run_loraks(editer_img: np.ndarray, cfg: dict) -> np.ndarray:
    """Run LORAKS on EDITER-corrected image. Returns reconstructed image (complex)."""
    lc = cfg["loraks"]
    return sense_loraks_single_channel(
        corr_img=editer_img,
        R=lc["R"],
        rank=lc["rank"],
        lam=lc["lam"],
        tol=lc["tol"],
        max_iter=lc["max_iter"],
        LORAKS_type=lc["LORAKS_type"],
        cg_maxiter=lc["cg_maxiter"],
        cg_tol=lc["cg_tol"],
    )


def run_ddnm(loraks_img: np.ndarray, cfg: dict) -> np.ndarray:
    """Run DDNM diffusion on LORAKS output. Returns result in [0,1] float."""
    ddnm = DDNMDiffusion(cfg["ddnm"])
    return ddnm.run(loraks_img)


def save_outputs(stage: str, img: np.ndarray, data_id: int, cfg: dict, is_ddnm: bool = False) -> None:
    """Save stage output: PNG in 'image' folder, NPY in 'npy' folder."""
    base_dir = os.path.join(cfg["output"]["results_dir"], f"data{data_id}")
    img_dir = os.path.join(base_dir, "image")
    npy_dir = os.path.join(base_dir, "npy")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    # Save NPY
    if cfg["output"].get("save_npy", True):
        np.save(os.path.join(npy_dir, f"{stage}.npy"), img)

    # Save PNG (only the legacy_view for non-ddnm results)
    if not is_ddnm:
        vc  = cfg.get("view", {})
        col_start = vc.get("col_start", 165)
        col_end   = vc.get("col_end", 366)
        view_img = legacy_view(img, col_start, col_end)
        save_image(view_img, os.path.join(img_dir, f"{stage}.png"))
    else:
        img = img[77:-78, 27:-28]
        save_image(img, os.path.join(img_dir, f"{stage}.png"), is_ddnm)
    
    print(f"  [{stage}] saved to {img_dir}/ and {npy_dir}/")


# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(mat_path: str, data_id: int, cfg: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Dataset {data_id}: {mat_path}")
    print(f"{'='*60}")

    # 1) Load
    datafft, noise_list = load_mat_data(mat_path)
    uncorr = ift2(datafft)
    save_outputs("uncorr", uncorr, data_id, cfg)

    # 2) EDITER
    print("\n[1/3] Running EDITER...")
    editer_img = run_editer(datafft, noise_list, cfg)
    save_outputs("editer", editer_img, data_id, cfg)

    # 3) LORAKS (on EDITER result)
    print("[2/3] Running LORAKS...")
    loraks_img = run_loraks(editer_img, cfg)
    save_outputs("loraks", loraks_img, data_id, cfg)

    # 4) DDNM (on LORAKS result)
    print("[3/3] Running DDNM...") # Start with LORAKS result in image folder
    loraks_img = cv2.imread(f'./results/data{data_id}/image/loraks.png',0)
    ddnm_result = run_ddnm(loraks_img, cfg)
    save_outputs("ddnm", ddnm_result, data_id, cfg, is_ddnm=True)

    print(f"\n  ✓ Dataset {data_id} complete.")


def main():
    cfg = load_config(os.path.join(ROOT, "config.yml"))

    mat_paths = {
        1: cfg["data"]["mat_path_1"],
        2: cfg["data"]["mat_path_2"],
    }

    selected = cfg.get("run", {}).get("datasets", [1, 2])
    print(f"Running datasets: {selected}")

    for data_id in selected:
        process_dataset(mat_paths[data_id], data_id=data_id, cfg=cfg)

    print("\n✓ All selected datasets processed.")

    for data_id in selected:
        gt_eval(data_id, os.path.join(cfg["output"]["results_dir"], f"data{data_id}", "image"))


if __name__ == "__main__":
    main()

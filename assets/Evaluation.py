import os
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np


def find_images(folder):
    fold_path = glob.glob(os.path.join(folder,'*.png'))
    paths = sorted(fold_path)
    if len(paths) == 0:
        raise ValueError(f"`{folder}`에서 이미지를 찾지 못했습니다.")
    return paths


def load_image_rgb(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def gt_eval(
    data,
    result_dir
):

    if data == 1:
        ref_path = './dataset/GT/gt1.png'
        ref_pil = load_image_rgb(ref_path)
    else:
        ref_path = './dataset/GT/gt2.png'
        ref_pil = load_image_rgb(ref_path)
    
    edit_paths = find_images(result_dir)

    results = []

    for p in edit_paths:
        edit_pil = load_image_rgb(p)
        if edit_pil.size != ref_pil.size:
            edit_pil = edit_pil.resize((ref_pil.size[0], ref_pil.size[1]), resample= Image.BICUBIC)
        edit_pil = np.array(edit_pil)
        # psnr_value = psnr(ref_pil / np.max(ref_pil), edit_pil / np.max(edit_pil), data_range=1.0)
        psnr_value = psnr(np.array(ref_pil), edit_pil, data_range=255.0)
        # ssim_value = ssim(ref_pil / np.max(ref_pil), edit_pil / np.max(edit_pil), data_range=1.0, channel_axis=-1)
        ssim_value = ssim(np.array(ref_pil), edit_pil, data_range=255.0, channel_axis=-1)
        results.append((os.path.basename(p), psnr_value, ssim_value))

    print(f"Reference: {ref_path}")
    print("-" * 72)
    print(f"{'image':40s} | {'PSNR':>10s} | {'SSIM':>10s}")
    print("-" * 72)
    for name, psnr_, ssim_ in results:
        print(f"{name:40s} | {psnr_:10.6f} | {ssim_:10.6f}")
    print("-" * 72)
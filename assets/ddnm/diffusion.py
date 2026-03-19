import os
import sys
import numpy as np
import torch
import tqdm
import cv2
import torchvision.transforms as transforms

# Make guided_diffusion importable relative to this file
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from guided_diffusion.script_util import create_model
from functions.ckpt_util import download
import matplotlib.pyplot as plt


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        betas = (np.linspace(beta_start**0.5, beta_end**0.5,
                             num_diffusion_timesteps, dtype=np.float64)) ** 2
    else:
        raise NotImplementedError(beta_schedule)
    return betas


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)


def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1
    t = T_sampling
    ts = []
    while t >= 1:
        t -= 1
        ts.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(travel_length):
                t += 1
                ts.append(t)
    ts.append(-1)
    return ts


def data_transform(config_data, x):
    """Normalize image tensor to [-1, 1]."""
    x = 2 * x - 1.0
    return x


def inverse_data_transform(config_data, x):
    if config_data.get("logit_transform", False):
        return torch.sigmoid(x)
    if config_data.get("rescaled", True):
        x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


# ─── Main DDNM class ──────────────────────────────────────────────────────────

class DDNMDiffusion:
    """
    Simplified DDNM diffusion for single-channel MRI.
    Takes LORAKS output as numpy array, returns final image as numpy array.
    """

    def __init__(self, cfg: dict, device=None):
        """
        cfg: the 'ddnm' block from config.yml (already loaded as dict)
        """
        self.cfg = cfg
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._build_betas()

    def _build_betas(self):
        betas = get_beta_schedule(
            self.cfg.get("beta_schedule", "linear"),
            beta_start=self.cfg.get("beta_start", 1e-4),
            beta_end=self.cfg.get("beta_end", 0.02),
            num_diffusion_timesteps=self.cfg.get("num_diffusion_timesteps", 1000),
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = alphas.cumprod(dim=0)

    def _load_model(self):
        """Load the diffusion U-Net from checkpoint."""
        image_size = self.cfg.get("image_size", 256)
        model = create_model(
            image_size=image_size,
            num_channels=256,
            num_res_blocks=2,
            channel_mult="",
            learn_sigma=True,
            class_cond=False,
            use_checkpoint=False,
            attention_resolutions="32,16,8",
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0.0,
            resblock_updown=True,
            use_fp16=self.cfg.get("use_fp16", True),
            use_new_attention_order=False,
        )
        ckpt = self.cfg.get("checkpoint_path", "assets/ddnm/exp/logs/imagenet/256x256_diffusion_uncond.pt")
        if not os.path.exists(ckpt):
            download(
                "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
                ckpt,
            )
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        if self.cfg.get("use_fp16", True):
            model.convert_to_fp16()
        model.to(self.device)
        model.eval()
        return torch.nn.DataParallel(model)

    def run(self, loraks_img: np.ndarray) -> np.ndarray:
        """
        Run DDNM on a single LORAKS output image.

        Args:
            loraks_img: 2D numpy array (H, W), magnitude image from LORAKS

        Returns:
            result: 2D numpy array (H, W), diffusion-reconstructed image
        """
        cfg = self.cfg
        image_size   = cfg.get("image_size", 256)
        sigma_y      = cfg.get("sigma_y", 0.3)
        eta          = cfg.get("eta", 0.85)
        T_sampling   = cfg.get("T_sampling", 100)
        travel_len   = cfg.get("travel_length", 1)
        travel_rep   = cfg.get("travel_repeat", 1)
        crop_top     = cfg.get("crop_top", 27)
        crop_bottom  = cfg.get("crop_bottom", 28)
        crop_left    = cfg.get("crop_left", 77)
        crop_right   = cfg.get("crop_right", 78)
        num_ts       = cfg.get("num_diffusion_timesteps", 1000)
        cfg_data     = {"rescaled": True, "logit_transform": False}

        model = self._load_model()

        # ── Prepare Apy (guide image) ─────────────────────────────────────────
        tt = transforms.ToTensor()
        apy_np = loraks_img
        apy_np = apy_np.T
        apy_t = tt(apy_np.copy())
        apy_t = data_transform(cfg_data, apy_t)

        sigma_y_s = 2 * sigma_y            # account for [-1,1] scaling

        # ── Random starting noise ─────────────────────────────────────────────
        x = torch.randn(1, cfg.get("channels", 3), image_size, image_size,
                        device=self.device)

        skip   = num_ts // T_sampling
        n      = x.size(0)
        xs     = [x]
        x0_preds = []

        times      = get_schedule_jump(T_sampling, travel_len, travel_rep)
        time_pairs = list(zip(times[:-1], times[1:]))

        with torch.no_grad():
            for i, j in tqdm.tqdm(time_pairs, desc="DDNM sampling"):
                i, j = i * skip, j * skip
                if j < 0:
                    j = -1

                if j < i:  # forward denoising step
                    t      = (torch.ones(n) * i).to(self.device)
                    next_t = (torch.ones(n) * j).to(self.device)
                    at      = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    sigma_t = (1 - at_next**2).sqrt()

                    xt = xs[-1].to(self.device)
                    et = model(xt, t)
                    if et.size(1) == 6:
                        et = et[:, :3]

                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                    if sigma_t >= at_next * sigma_y_s:
                        lambda_t = 1.0
                        gamma_t  = (sigma_t**2 - (at_next * sigma_y_s)**2).sqrt()
                    else:
                        lambda_t = (sigma_t / (at_next * sigma_y_s)).clone().detach().cpu()
                        gamma_t  = 0.0

                    # Crop diffusion estimate to LORAKS image size, blend with Apy
                    x0_t1  = x0_t.clone().detach().cpu().mean(dim=1).unsqueeze(1)
                    re_x0_t = x0_t1[:, :, crop_top:-crop_bottom, crop_left:-crop_right]
                    x0_t_hat = lambda_t * apy_t + (1 - lambda_t) * re_x0_t
                    x0_t_hat = np.pad(
                        x0_t_hat.detach().squeeze(0).squeeze(0).numpy(),
                        ((crop_top, crop_bottom), (crop_left, crop_right)),
                    )
                    x0_t_hat = np.expand_dims(np.expand_dims(x0_t_hat, 0), 0)
                    x0_t_hat = torch.tensor(np.repeat(x0_t_hat, 3, 1),
                                            dtype=torch.float32).to(self.device)

                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta**2) ** 0.5)
                    xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (
                        c1 * torch.randn_like(x0_t) + c2 * et
                    )

                    x0_preds.append(x0_t.to("cpu"))
                    xs.append(xt_next.to("cpu"))

                else:  # time-travel back
                    next_t  = (torch.ones(n) * j).to(self.device)
                    at_next = compute_alpha(self.betas, next_t.long())
                    x0_t    = x0_preds[-1].to(self.device)
                    xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()
                    xs.append(xt_next.to("cpu"))

        # ── Decode final result ───────────────────────────────────────────────
        final = xs[-1]
        result_t = inverse_data_transform(cfg_data, final)
        result_np = result_t.squeeze(0).permute(2,1,0).detach().cpu().numpy()*255

        return result_np

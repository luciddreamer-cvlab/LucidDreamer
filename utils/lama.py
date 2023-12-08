import os
import sys
import hashlib
import logging
from typing import Union
from urllib.parse import urlparse

import numpy as np
import torch
from torch.hub import download_url_to_file, get_dir


LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",)
LAMA_MODEL_MD5 = os.environ.get(
    "LAMA_MODEL_MD5",
    "e3aa4aaa15225a33ec84f9f4bc47e500")


def md5sum(filename: str) -> str:
    md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def handle_error(model_path: str, model_md5: str, e: str) -> None:
    _md5 = md5sum(model_path)
    if _md5 != model_md5:
        try:
            os.remove(model_path)
            logging.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, wrong model "
                f"deleted. Please restart lama-cleaner. If you still have "
                f"errors, please try download model manually first https://"
                f"lama-cleaner-docs.vercel.app/install/download_model_"
                f"manually.\n")
        except:
            logging.error(
                f"Model md5: {_md5}, expected md5: {model_md5}, please delete"
                f" {model_path} and restart lama-cleaner.")
    else:
        logging.error(
            f"Failed to load model {model_path}, please submit an issue at "
            f"https://github.com/ironjr/simple-lama/issues and include a "
            f"screenshot of the error:\n{e}")
    exit(-1)


def get_cache_path_by_url(url: str) -> str:
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url: str, model_md5: str = None) -> str:
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
        if model_md5:
            _md5 = md5sum(cached_file)
            if model_md5 == _md5:
                logging.info(f"Download model success, md5: {_md5}")
            else:
                try:
                    os.remove(cached_file)
                    logging.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, wrong"
                        f" model deleted. Please restart lama-cleaner. If you"
                        f" still have errors, please try download model "
                        f"manually first https://lama-cleaner-docs.vercel"
                        f".app/install/download_model_manually.\n")
                except:
                    logging.error(
                        f"Model md5: {_md5}, expected md5: {model_md5}, "
                        f"please delete {cached_file} and restart lama-"
                        f"cleaner.")
                exit(-1)
    return cached_file


def load_jit_model(
    url_or_path: str,
    device: Union[torch.device, str],
    model_md5: str,
) -> torch.jit._script.RecursiveScriptModule:
    if os.path.exists(url_or_path):
        model_path = url_or_path
    else:
        model_path = download_model(url_or_path, model_md5)

    logging.info(f"Loading model from: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu").to(device)
    except Exception as e:
        handle_error(model_path, model_md5, e)
    model.eval()
    return model


def norm_img(np_img: np.ndarray) -> np.ndarray:
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def ceil_modulo(x: int, mod: int) -> int:
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img: np.ndarray, mod: int) -> np.ndarray:
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


class LaMa:
    name = "lama"
    pad_mod = 8

    def __init__(self, device: Union[torch.device, str], **kwargs) -> None:
        self.device = device
        self.model = load_jit_model(
            LAMA_MODEL_URL, device, LAMA_MODEL_MD5).eval()

    @staticmethod
    def is_downloaded() -> bool:
        return os.path.exists(get_cache_path_by_url(LAMA_MODEL_URL))

    def forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: RGB IMAGE
        """
        dtype = image.dtype
        image = norm_img(image)
        mask = norm_img(mask if np.max(mask) > 1.0 else mask * 2)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255)
        return cur_res.astype(dtype)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        images: [H, W, C] RGB, not normalized
        masks: [H, W]
        return: RGB IMAGE
        """
        dtype = image.dtype
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(image, mod=self.pad_mod)
        pad_mask = pad_img_to_modulo(mask, mod=self.pad_mod)

        result = self.forward(pad_image, pad_mask)
        result = result[0:origin_height, 0:origin_width, :]

        mask = mask[:, :, np.newaxis]
        mask = mask / 255 if np.max(mask) > 1.0 else mask
        result = result * mask + image * (1 - mask)
        return result.astype(dtype)

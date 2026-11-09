import os
import cv2
import numpy as np
from pathlib import Path

import mod_img
import metrics
import stretching_mthd

IMAGE_NAMES = ["arena", "firenze", "ippopotamo", "panda", "penna"]

BASE_DIR = Path(__file__).resolve().parent.parent   # project folder (above src)
INPUT_DIR = BASE_DIR / "data" / "original_img"
RESIZED_DIR = BASE_DIR / "data" / "resized_img"
DEGRADED_DIR = BASE_DIR / "data" / "degraded_img"
OUTPUT_DIR_PROCESSED = BASE_DIR / "output" / "processed_img"
STRETCHING_DIR = OUTPUT_DIR_PROCESSED / "stretching_img"


def main():
    np.random.seed(42)

    os.makedirs(RESIZED_DIR, exist_ok=True)
    os.makedirs(DEGRADED_DIR, exist_ok=True)
    os.makedirs(STRETCHING_DIR, exist_ok=True)

    results = []   # one entry per (image, method): later written to metrics.csv

    for name in IMAGE_NAMES:
        input_path = INPUT_DIR / f"{name}.png"
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Error: could not read '{input_path}'")
            continue

        # --- Data preparation ---
        resized_image = mod_img.resize_image(image)
        degraded_image = mod_img.modify_image(resized_image)
        cv2.imwrite(str(RESIZED_DIR / f"{name}_resized.png"), resized_image)
        cv2.imwrite(str(DEGRADED_DIR / f"{name}_degraded.png"), degraded_image)

        # --- Method 1: baseline (degraded image, no processing) ---
        psnr_baseline = metrics.metric_psnr(resized_image, degraded_image)
        results.append({"image": name, "method": "baseline", "psnr": psnr_baseline})

        # --- Method 2: linear stretching (per channel) ---
        # on the degraded image -> evaluated with metrics
        stretched_deg = stretching_mthd.stretching_canalRGB(degraded_image, 0, 255)
        cv2.imwrite(str(STRETCHING_DIR / f"{name}_degraded_stretching.png"), stretched_deg)
        psnr_stretched = metrics.metric_psnr(resized_image, stretched_deg)
        results.append({"image": name, "method": "stretching", "psnr": psnr_stretched})

        # on the real (resized) image -> qualitative evaluation only
        stretched_real = stretching_mthd.stretching_canalRGB(resized_image, 0, 255)
        cv2.imwrite(str(STRETCHING_DIR / f"{name}_real_stretching.png"), stretched_real)

        print(f"{name}: PSNR baseline = {psnr_baseline:.2f} dB, stretching = {psnr_stretched:.2f} dB")


if __name__ == "__main__":
    main()
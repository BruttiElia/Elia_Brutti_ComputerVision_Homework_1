import os
import cv2
import numpy as np
from pathlib import Path

import mod_img
import metrics
import stretching_mthd
import equalization_mthd
import denoising_mthd

IMAGE_NAMES = ["arena", "firenze", "ippopotamo", "panda", "penna"]

BASE_DIR = Path(__file__).resolve().parent.parent   # project folder (above src)
INPUT_DIR = BASE_DIR / "data" / "original_img"
RESIZED_DIR = BASE_DIR / "data" / "resized_img"
DEGRADED_DIR = BASE_DIR / "data" / "degraded_img"
OUTPUT_DIR_PROCESSED = BASE_DIR / "output" / "processed_img"
STRETCHING_DIR = OUTPUT_DIR_PROCESSED / "stretching_img"
EQUALIZATION_DIR = OUTPUT_DIR_PROCESSED / "equalization_img"
DENOISING_DIR = OUTPUT_DIR_PROCESSED / "denoising_img"

def main():
    np.random.seed(42)

    os.makedirs(RESIZED_DIR, exist_ok=True)
    os.makedirs(DEGRADED_DIR, exist_ok=True)
    os.makedirs(STRETCHING_DIR, exist_ok=True)
    os.makedirs(EQUALIZATION_DIR, exist_ok=True)
    os.makedirs(DENOISING_DIR, exist_ok=True)

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

        # --- Method 3: histogram equalization (HSV) ---
        equalized_hsv = equalization_mthd.equalization_HSV(degraded_image)
        cv2.imwrite(str(EQUALIZATION_DIR / f"{name}_equalized_hsv.png"), equalized_hsv)
        psnr_equalized = metrics.metric_psnr(resized_image, equalized_hsv)
        results.append({"image": name, "method": "equalization_hsv", "psnr": psnr_equalized})

        #on the real (resized) image -> qualitative evaluation only
        equalized_real = equalization_mthd.equalization_HSV(resized_image)
        cv2.imwrite(str(EQUALIZATION_DIR / f"{name}_real_equalized_hsv.png"), equalized_real)

        # --- Method 4: denoising (bilateral) ---
        denoised_bilateral = denoising_mthd.denoising_bilateral(degraded_image, d=9, sigma_color=75, sigma_space=75)
        cv2.imwrite(str(DENOISING_DIR / f"{name}_denoised_bilateral.png"), denoised_bilateral)
        psnr_denoised_bilateral = metrics.metric_psnr(resized_image, denoised_bilateral)
        results.append({"image": name, "method": "denoising_bilateral", "psnr": psnr_denoised_bilateral})

        #on the real (resized) image -> qualitative evaluation only
        denoised_bilateral_real = denoising_mthd.denoising_bilateral(resized_image, d=9, sigma_color=75, sigma_space=75)
        cv2.imwrite(str(DENOISING_DIR / f"{name}_real_denoised_bilateral.png"), denoised_bilateral_real)    

        # --- Method 4 : denoising (non-local means) ---
        denoised_nlm = denoising_mthd.denoising_non_local_means(degraded_image, h=10, template_window_size=7, search_window_size=21)
        cv2.imwrite(str(DENOISING_DIR / f"{name}_denoised_nlm.png"), denoised_nlm)
        psnr_denoised_nlm = metrics.metric_psnr(resized_image, denoised_nlm)
        results.append({"image": name, "method": "denoising_nlm", "psnr": psnr_denoised_nlm})

        #on the real (resized) image -> qualitative evaluation only
        denoised_nlm_real = denoising_mthd.denoising_non_local_means(resized_image, h=10, template_window_size=7, search_window_size=21)
        cv2.imwrite(str(DENOISING_DIR / f"{name}_real_denoised_nlm.png"), denoised_nlm_real)    


        print(f"{name}: PSNR baseline = {psnr_baseline:.2f} dB, stretching = {psnr_stretched:.2f} dB , equalization HSV = {psnr_equalized:.2f} dB, denoising bilateral = {psnr_denoised_bilateral:.2f} dB, denoising NLM = {psnr_denoised_nlm:.2f} dB")


if __name__ == "__main__":
    main()
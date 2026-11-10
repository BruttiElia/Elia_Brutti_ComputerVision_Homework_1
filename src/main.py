import os
import cv2
import numpy as np
import csv
from pathlib import Path

import mod_img
import metrics
import stretching_mthd
import equalization_mthd
import denoising_mthd
import visualization

# ---------------- Experiment parameters (documented in README and report) ----------------
# Data preparation
RESIZE_SIZE = 256        # output size (pixels) of the resized images
ALPHA = 0.7              # contrast reduction factor (1 = unchanged)
SIGMA = 10               # std of the Gaussian noise (0-255 scale)
SEED = 42                # random seed for noise generation

# Method 2: linear stretching
PERCENTILE_LOW = 0.5     # lower percentile for contrast stretching
PERCENTILE_HIGH = 99.5   # upper percentile for contrast stretching

# Method 4: denoising
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
NLM_H = 10
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21
# -------------------------------------------------------------------------------------------

IMAGE_NAMES = ["arena", "firenze", "ippopotamo", "panda", "penna"]

BASE_DIR = Path(__file__).resolve().parent.parent   # project folder (above src)
INPUT_DIR = BASE_DIR / "data" / "original_img"
RESIZED_DIR = BASE_DIR / "data" / "resized_img"
DEGRADED_DIR = BASE_DIR / "data" / "degraded_img"
OUTPUT_DIR_PROCESSED = BASE_DIR / "output" / "processed_img"
STRETCHING_DIR = OUTPUT_DIR_PROCESSED / "stretching_img"
EQUALIZATION_DIR = OUTPUT_DIR_PROCESSED / "equalization_img"
DENOISING_DIR = OUTPUT_DIR_PROCESSED / "denoising_img"
COMPARISON_DIR = BASE_DIR / "output" / "comparison_img"
METRICS_CSV = BASE_DIR / "output" / "metrics.csv"

# File naming scheme for processed images: <image>_<degraded|real>_<method>.png
# where <method> is the same name used in metrics.csv


def main():
    np.random.seed(SEED)  # for reproducibility of the noise

    os.makedirs(RESIZED_DIR, exist_ok=True)
    os.makedirs(DEGRADED_DIR, exist_ok=True)
    os.makedirs(STRETCHING_DIR, exist_ok=True)
    os.makedirs(EQUALIZATION_DIR, exist_ok=True)
    os.makedirs(DENOISING_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    results = []   # one entry per (image, method): later written to metrics.csv

    for name in IMAGE_NAMES:
        input_path = INPUT_DIR / f"{name}.png"
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Error: could not read '{input_path}'")
            continue

        # --- Data preparation ---
        resized_image = mod_img.resize_image(image, RESIZE_SIZE)
        degraded_image = mod_img.modify_image(resized_image, ALPHA, SIGMA)
        cv2.imwrite(str(RESIZED_DIR / f"{name}_resized.png"), resized_image)
        cv2.imwrite(str(DEGRADED_DIR / f"{name}_degraded.png"), degraded_image)

        # --- Method 1: baseline (degraded image, no processing) ---
        psnr_baseline = metrics.metric_psnr(resized_image, degraded_image)
        ssim_baseline = metrics.metric_ssim(resized_image, degraded_image)
        results.append({"image": name, "method": "baseline", "psnr": psnr_baseline, "ssim": ssim_baseline})

        # Collect results of this image for the comparison figure
        real_results = {}
        degraded_results = {}
        metrics_by_method = {"baseline": (psnr_baseline, ssim_baseline)}

        # --- Method 2: linear stretching (per channel) ---
        # on the degraded image -> evaluated with metrics
        stretched_deg = stretching_mthd.stretching_canalBGR(degraded_image, 0, 255, PERCENTILE_LOW, PERCENTILE_HIGH)
        cv2.imwrite(str(STRETCHING_DIR / f"{name}_degraded_stretching.png"), stretched_deg)
        psnr_stretched = metrics.metric_psnr(resized_image, stretched_deg)
        ssim_stretched = metrics.metric_ssim(resized_image, stretched_deg)
        results.append({"image": name, "method": "stretching", "psnr": psnr_stretched, "ssim": ssim_stretched})

        # on the real (resized) image -> qualitative evaluation only
        stretched_real = stretching_mthd.stretching_canalBGR(resized_image, 0, 255, PERCENTILE_LOW, PERCENTILE_HIGH)
        cv2.imwrite(str(STRETCHING_DIR / f"{name}_real_stretching.png"), stretched_real)

        degraded_results["stretching"] = stretched_deg
        real_results["stretching"] = stretched_real
        metrics_by_method["stretching"] = (psnr_stretched, ssim_stretched)

        # --- Method 3: histogram equalization (HSV) ---
        equalized_hsv = equalization_mthd.equalization_HSV(degraded_image)
        cv2.imwrite(str(EQUALIZATION_DIR / f"{name}_degraded_equalization_hsv.png"), equalized_hsv)      # CHANGED
        psnr_equalized = metrics.metric_psnr(resized_image, equalized_hsv)
        ssim_equalized = metrics.metric_ssim(resized_image, equalized_hsv)
        results.append({"image": name, "method": "equalization_hsv", "psnr": psnr_equalized, "ssim": ssim_equalized})

        # on the real (resized) image -> qualitative evaluation only
        equalized_real = equalization_mthd.equalization_HSV(resized_image)
        cv2.imwrite(str(EQUALIZATION_DIR / f"{name}_real_equalization_hsv.png"), equalized_real)         # CHANGED

        degraded_results["equalization HSV"] = equalized_hsv
        real_results["equalization HSV"] = equalized_real
        metrics_by_method["equalization HSV"] = (psnr_equalized, ssim_equalized)

        # --- Method 4: denoising (bilateral) ---
        denoised_bilateral = denoising_mthd.denoising_bilateral(degraded_image, d=BILATERAL_D, sigma_color=BILATERAL_SIGMA_COLOR, sigma_space=BILATERAL_SIGMA_SPACE)
        cv2.imwrite(str(DENOISING_DIR / f"{name}_degraded_denoising_bilateral.png"), denoised_bilateral)  # CHANGED
        psnr_denoised_bilateral = metrics.metric_psnr(resized_image, denoised_bilateral)
        ssim_denoised_bilateral = metrics.metric_ssim(resized_image, denoised_bilateral)
        results.append({"image": name, "method": "denoising_bilateral", "psnr": psnr_denoised_bilateral, "ssim": ssim_denoised_bilateral})

        # on the real (resized) image -> qualitative evaluation only
        denoised_bilateral_real = denoising_mthd.denoising_bilateral(resized_image, d=BILATERAL_D, sigma_color=BILATERAL_SIGMA_COLOR, sigma_space=BILATERAL_SIGMA_SPACE)
        cv2.imwrite(str(DENOISING_DIR / f"{name}_real_denoising_bilateral.png"), denoised_bilateral_real)  # CHANGED

        degraded_results["bilateral"] = denoised_bilateral
        real_results["bilateral"] = denoised_bilateral_real
        metrics_by_method["bilateral"] = (psnr_denoised_bilateral, ssim_denoised_bilateral)

        # --- Method 4: denoising (non-local means) ---
        denoised_nlm = denoising_mthd.denoising_non_local_means(degraded_image, h=NLM_H, template_window_size=NLM_TEMPLATE_WINDOW, search_window_size=NLM_SEARCH_WINDOW)
        cv2.imwrite(str(DENOISING_DIR / f"{name}_degraded_denoising_nlm.png"), denoised_nlm)              # CHANGED
        psnr_denoised_nlm = metrics.metric_psnr(resized_image, denoised_nlm)
        ssim_denoised_nlm = metrics.metric_ssim(resized_image, denoised_nlm)
        results.append({"image": name, "method": "denoising_nlm", "psnr": psnr_denoised_nlm, "ssim": ssim_denoised_nlm})

        # on the real (resized) image -> qualitative evaluation only
        denoised_nlm_real = denoising_mthd.denoising_non_local_means(resized_image, h=NLM_H, template_window_size=NLM_TEMPLATE_WINDOW, search_window_size=NLM_SEARCH_WINDOW)
        cv2.imwrite(str(DENOISING_DIR / f"{name}_real_denoising_nlm.png"), denoised_nlm_real)             # CHANGED

        degraded_results["NLM"] = denoised_nlm
        real_results["NLM"] = denoised_nlm_real
        metrics_by_method["NLM"] = (psnr_denoised_nlm, ssim_denoised_nlm)

        print(f"{name}: PSNR baseline = {psnr_baseline:.2f} dB, stretching = {psnr_stretched:.2f} dB, equalization HSV = {psnr_equalized:.2f} dB, denoising bilateral = {psnr_denoised_bilateral:.2f} dB, denoising NLM = {psnr_denoised_nlm:.2f} dB \n\tSSIM baseline = {ssim_baseline:.4f}, stretching = {ssim_stretched:.4f}, equalization HSV = {ssim_equalized:.4f}, denoising bilateral = {ssim_denoised_bilateral:.4f}, denoising NLM = {ssim_denoised_nlm:.4f}")

        # --- Comparison figure for this image ---
        visualization.save_comparison_figure(
            name, resized_image, degraded_image,
            real_results, degraded_results, metrics_by_method,
            COMPARISON_DIR / f"{name}_comparison.png",
        )

    # --- Save all metrics to CSV (written once, after processing every image) ---
    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "method", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Metrics saved to {METRICS_CSV}")


if __name__ == "__main__":
    main()
import os
import cv2
import numpy as np
import mod_img
import metrics
from pathlib import Path

IMAGE_NAMES = ["arena", "firenze", "ippopotamo", "panda", "penna"]

BASE_DIR = Path(__file__).resolve().parent.parent   # cartella del progetto (sopra src)
INPUT_DIR = BASE_DIR / "data" / "original_img" 
RESIZED_DIR = BASE_DIR / "data" / "resized_img"
DEGRADED_DIR = BASE_DIR / "data" / "degraded_img"


def main():
    np.random.seed(42)

    os.makedirs(RESIZED_DIR, exist_ok=True)
    os.makedirs(DEGRADED_DIR, exist_ok=True)

    for name in IMAGE_NAMES:
        input_path = INPUT_DIR / f"{name}.png"
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Error: could not read '{input_path}'")
            continue

        resized_image = mod_img.resize_image(image)
        degraded_image = mod_img.modify_image(resized_image)

        cv2.imwrite(str(RESIZED_DIR / f"{name}_resized.png"), resized_image)
        cv2.imwrite(str(DEGRADED_DIR / f"{name}_degraded.png"), degraded_image)
        print(f"{name}: done {resized_image.shape}")

    metrics_results = []
    for name in IMAGE_NAMES:
        resized_path = RESIZED_DIR / f"{name}_resized.png"
        degraded_path = DEGRADED_DIR / f"{name}_degraded.png"

        resized_image = cv2.imread(str(resized_path))
        degraded_image = cv2.imread(str(degraded_path))

        if resized_image is None or degraded_image is None:
            print(f"Error: could not read images for '{name}'")
            continue

        mse = metrics.metric_mse(resized_image, degraded_image)
        psnr = metrics.metric_psnr(resized_image, degraded_image)

        metrics_results.append((name, mse, psnr))
        print(f"{name}: MSE={mse:.2f}, PSNR={psnr:.2f} dB")


if __name__ == "__main__":
    main()
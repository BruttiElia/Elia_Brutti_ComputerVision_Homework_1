# Comparison figures for the report: one 2-row grid per image.

import cv2
import matplotlib
matplotlib.use("Agg")            # no windows: figures are only saved to file
import matplotlib.pyplot as plt


def to_rgb(image_bgr):
    """OpenCV images are BGR, matplotlib expects RGB."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_comparison_figure(name, resized, degraded, real_results, degraded_results,
                           metrics_by_method, out_path):
    """
    Save a 2-row comparison grid for one image.
    Row 1 (real):      resized original | each method applied to the original
    Row 2 (synthetic): degraded         | each method applied to the degraded image
    Titles in row 2 report PSNR and SSIM against the resized original.

    - real_results / degraded_results: dict  method name -> result image (BGR)
    - metrics_by_method: dict  method name -> (psnr, ssim); must include "baseline"
    """
    n_cols = 1 + len(degraded_results)
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 7.5))

    # Row 1: real image and methods applied to it (qualitative only)
    axes[0, 0].imshow(to_rgb(resized))
    axes[0, 0].set_title("original (real)", fontsize=9)
    for col, (method, img) in enumerate(real_results.items(), start=1):
        axes[0, col].imshow(to_rgb(img))
        axes[0, col].set_title(f"{method} (real)", fontsize=9)

    # Row 2: degraded image and methods applied to it (with metrics)
    psnr_b, ssim_b = metrics_by_method["baseline"]
    axes[1, 0].imshow(to_rgb(degraded))
    axes[1, 0].set_title(f"degraded\nPSNR {psnr_b:.2f} dB | SSIM {ssim_b:.3f}", fontsize=9)
    for col, (method, img) in enumerate(degraded_results.items(), start=1):
        psnr_v, ssim_v = metrics_by_method[method]
        axes[1, col].imshow(to_rgb(img))
        axes[1, col].set_title(f"{method}\nPSNR {psnr_v:.2f} dB | SSIM {ssim_v:.3f}", fontsize=9)

    for ax in axes.ravel():
        ax.axis("off")             # no ticks: these are images, not plots

    fig.suptitle(f"{name} - metrics (row 2) computed against original (real)", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)                 # free memory
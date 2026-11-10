# Image Enhancement Pipeline – Computer Vision Homework 1

**Author:** Elia Brutti mat.VR553705

This project builds and compares a small pipeline of **classical image processing methods** (OpenCV + NumPy) to enhance color images affected by non-uniform illumination, low contrast and noise.

Each method is applied to:

- the **real** image (original photo, resized), evaluated **qualitatively** only, since no clean reference exists for real-world images.
- the **degraded** version of each image (synthetic degradation: reduced contrast + Gaussian noise), evaluated **quantitatively** with PSNR and SSIM against the original;


Implemented methods:

| #  | Method                       | Description                                                                   |
|--- |------------------------------|-------------------------------------------------------------------------------|
| 1  | Baseline                     | Degraded image without processing                                             |
| 2  | Linear contrast stretching   | Per-channel (B, G, R) stretching between two percentiles, mapped to [0, 255]  |
| 3  | Histogram equalization       | Equalization of the V channel in HSV (H and S unchanged)                      |
| 4a | Denoising – bilateral filter | `cv2.bilateralFilter`                                                         |
| 4b | Denoising – non-local means  | `cv2.fastNlMeansDenoisingColored`                                             |

---

## Requirements

- **Python 3.11**
- Dependencies listed in `requirements.txt`:

```
numpy==2.0.2
opencv-python==5.0.0.93
scikit-image==0.24.0
matplotlib==3.9.4
```

`scikit-image` is used only for SSIM; `matplotlib` only for the comparison figures.

## Installation

The project was developed and tested on **macOS** (Python 3.11).
Instructions are given for macOS/Linux and for Windows. Run all commands from the project root folder.

### macOS / Linux

(program tested on macOS)

```bash
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install the dependencies
pip install -r requirements.txt
```

### Windows

(on README of lab )

```bat
:: create and activate a virtual environment (Command Prompt)
python -m venv .venv
.venv\Scripts\activate

:: install the dependencies
pip install -r requirements.txt
```

If you use **PowerShell**, activate the environment with `.venv\Scripts\Activate.ps1` instead.

## How to run

From the project root folder, run:

```bash
python src/main.py
```

This single command runs the **complete pipeline** starting from the images in `data/original_img/`:

1. resizes the original images and creates their degraded versions;
2. applies all methods to both the degraded and the real (resized) images;
3. computes PSNR and SSIM for every method on the degraded images;
4. saves the processed images, the comparison figures and `output/metrics.csv`.

All generated folders are created automatically if they do not exist.

## Output

| Path                                      | Content                                                       |
|-------------------------------------------|---------------------------------------------------------------|
| `data/resized_img/`                       | Resized original images (used as reference for the metrics)   |
| `data/degraded_img/`                      | Synthetically degraded images                                 |
| `output/processed_img/stretching_img/`    | Results of linear stretching                                  |
| `output/processed_img/equalization_img/`  | Results of histogram equalization                             |
| `output/processed_img/denoising_img/`     | Results of bilateral and non-local means denoising            |
| `output/comparison_img/`                  | One comparison figure per image (see below)                   |
| `output/metrics.csv`                      | PSNR and SSIM for every image and method                      |

**File naming scheme** for processed images:

```
<image>_<degraded|real>_<method>.png
```

where `<method>` is the same name used in `metrics.csv` (`stretching`, `equalization_hsv`, `denoising_bilateral`, `denoising_nlm`).
Example: `arena_degraded_denoising_nlm.png` is the non-local means result on the degraded version of `arena`.

**Comparison figures** (`<image>_comparison.png`) are 2-row grids:

- row 1: real image and each method applied to it (qualitative only);
- row 2: degraded image and each method applied to it, with PSNR and SSIM computed **against the resized original**.

**`metrics.csv`** columns: `image`, `method`, `psnr` (dB), `ssim`. For both metrics, higher means closer to the original. The `baseline` rows give the quality of the degraded image before any processing.

## File structure

```
Elia_Brutti_ComputerVision_Homework_1/
├── README.md
├── requirements.txt
├── src/
│   ├── main.py               # entry point: runs the whole pipeline, holds all parameters
│   ├── mod_img.py            # data preparation: resizing and synthetic degradation
│   ├── stretching_mthd.py    # method 2: linear contrast stretching
│   ├── equalization_mthd.py  # method 3: histogram equalization (HSV)
│   ├── denoising_mthd.py     # method 4: bilateral and non-local means denoising
│   ├── metrics.py            # MSE, PSNR and SSIM
│   └── visualization.py      # comparison figures
├── data/
│   ├── original_img/         # input photos (included in the repository)
│   ├── resized_img/          # generated
│   └── degraded_img/         # generated
└── output/
    ├── processed_img/        # generated: one subfolder per method
    ├── comparison_img/       # generated: comparison figures
    └── metrics.csv           # generated: metrics table
```

## Parameters

All parameters are defined in a single block at the top of `src/main.py`.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `RESIZE_SIZE` | 256 | Shorter side resized to 256 px, then center crop to 256 × 256 (aspect ratio preserved, no distortion) |
| `ALPHA` | 0.7 | Contrast reduction: `out = mean + ALPHA · (I − mean)`, with the global mean of the image |
| `SIGMA` | 10 | Standard deviation of the zero-mean Gaussian noise (0–255 scale), independent per channel |
| `SEED` | 42 | Random seed (`np.random.seed`) for reproducible noise |
| `PERCENTILE_LOW` / `PERCENTILE_HIGH` | 0.5 / 99.5 | Percentiles mapped to 0 and 255 in linear stretching |
| `BILATERAL_D` | 9 | Bilateral filter: neighborhood diameter |
| `BILATERAL_SIGMA_COLOR` / `BILATERAL_SIGMA_SPACE` | 75 / 75 | Bilateral filter: range and spatial sigma |
| `NLM_H` | 10 | Non-local means: filter strength (used for both `h` and `hColor`) |
| `NLM_TEMPLATE_WINDOW` / `NLM_SEARCH_WINDOW` | 7 / 21 | Non-local means: patch size and search window size |

SSIM is computed with `skimage.metrics.structural_similarity` using a Gaussian window (`sigma = 1.5`), `data_range = 255` and averaged over the three color channels.

## Images

Five color photographs, each showing at least one of the conditions required by the assignment:

| File              | Content                                       | Main condition                                |  
|-------------------|-----------------------------------------------|-----------------------------------------------|
| `arena.png`       | Arena di Verona at night                      | Low light, strongly non-uniform illumination  |
| `firenze.png`     | Arno river in Florence, backlit by the sun    | Backlight, details in shadow, saturated areas |
| `ippopotamo.png`  | Hippopotamus in a pond                        | Dappled light, low-contrast subject           |
| `panda.png`       | Red panda among pine branches                 | Fine details and textures                     |
| `penna.png`       | White pen on grid paper                       | Low contrast (white on white), slight shading |

**Source and license:** all photographs were taken by the author (Elia Brutti).

**Note on evaluation:** the original (resized) images serve as references only for evaluating the **synthetic** degradations. In real-world scenarios a clean reference is generally not available, which is why the methods are also applied to the real images and evaluated qualitatively.
# AI-Powered Medical Image Analysis System

An educational, reproducible chest X-ray classification project built with Python, TensorFlow/Keras, and the public PneumoniaMNIST benchmark. It trains a small CNN to classify benchmark images as **normal** or **pneumonia**, reports held-out metrics, and saves plots and sample predictions.

> **Research/education only.** This project is not a medical device and must not be used for diagnosis, triage, treatment, or patient care. Benchmark performance does not establish clinical validity. It does not connect to hospital systems or simulate a real clinical workflow.

## Quick start on Windows

Open Command Prompt or PowerShell in this project folder. Run setup once, then train and evaluate:

```powershell
.\setup_windows.bat
.\run_windows.bat train --epochs 15 --batch-size 32
.\run_windows.bat evaluate
```

No environment activation is needed. The batch files select the project's Python 3.12 environment explicitly, even if the system `python` command points to Python 3.14. The first training run needs internet access to download the public dataset.

## What this project demonstrates

Medical image analysis uses software to measure or classify images and help qualified clinicians review information. In hospitals, diagnostic labs, radiology centers, and health technology companies, computer vision can help prioritize worklists, flag regions for review, quantify findings, or support research. A model can make mistakes, inherit dataset bias, and miss disease. A clinician remains responsible for interpretation.

In plain language, this project shows how example X-rays can be converted to numbers, passed through a neural network, and compared with labels that were supplied with the dataset. Technically, it is binary supervised image classification: image decoding and scaling, learned convolutional feature extraction, sigmoid probability, threshold-based class prediction, and evaluation on a pre-defined held-out split.

This is a **classification** demo, not an object detector or segmentation system. It does not draw a tumor boundary, locate a fracture, or diagnose an individual.

## Selected approach and alternatives

| Option | Stack/model | Data | Difficulty / hardware | Output |
| --- | --- | --- | --- | --- |
| A — easiest (selected) | Python, NumPy, Matplotlib, TensorFlow/Keras; small CNN | PneumoniaMNIST, 28×28 chest X-rays | Beginner; CPU works, GPU optional | Normal/pneumonia probability, metrics and plots |
| B — intermediate | TensorFlow/Keras or PyTorch; transfer learning with MobileNetV2 | Larger public chest X-ray dataset with its license and split rules | Intermediate; CPU possible but training slower, GPU useful | Higher-resolution classification and saliency exploration |
| C — advanced | PyTorch, MONAI, 3D CNN/nnU-Net, DICOM tooling | Public CT/MRI volumes with segmentation masks | Advanced; GPU and substantial storage strongly recommended | Volumetric segmentation/detection and 3D visualization |

Option A is selected because it downloads reproducibly, is small enough for a laptop, and exposes the complete supervised-learning workflow. The trade-off is important: 28×28 images have been heavily downsampled and are suitable for a benchmark exercise, not a realistic radiology product. PneumoniaMNIST is derived from 5,856 pediatric chest X-rays; see the [official MedMNIST repository](https://github.com/MedMNIST/MedMNIST), [dataset description](https://medmnist.com/), and [paper](https://doi.org/10.1038/s41597-022-01721-8). The repository documents the official distribution at [Zenodo](https://doi.org/10.5281/zenodo.10519652). Review the source dataset terms and attribution before redistribution.

## Architecture and data flow

```text
MedMNIST download (train / validation / test)
        ↓
grayscale 28×28 arrays + binary labels
        ↓ scale pixels to [0, 1], add channel dimension
small CNN: Conv → Pool → Conv → Pool → Dense → Sigmoid
        ↓
probability + thresholded class (educational output only)
        ↓
test metrics (loss, accuracy, precision, recall, AUC)
        ↓
training curve, confusion matrix, sample prediction image, JSON metrics
```

`src/data.py` downloads and prepares the published splits. `src/model.py` defines the model. `src/train.py` trains on train and selects progress using validation data. `src/evaluate.py` evaluates the test split and writes plots and metrics. `src/predict.py` runs a saved model on a user-supplied image. `main.py` provides simple commands.

## Folder structure

```text
AI-Powered Medical Image Analysis/
├── data/                 # downloaded benchmark (ignored by Git)
├── docs/                 # workflow, limitations, proof checklist
├── images/               # README screenshots and demo assets
├── models/               # saved Keras weights (ignored by Git; publish separately if desired)
├── notebooks/            # space for optional exploration notebooks
├── outputs/              # generated plots and metrics (ignored by Git)
├── src/                  # reusable code
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

## Setup

Use **64-bit Python 3.12** in a virtual environment. This project pins TensorFlow to 2.20.0, a version compatible with Python 3.12. It reads the official MedMNIST NPZ archive directly, so it does not install MedMNIST's PyTorch/torchvision stack. A GPU is optional; CPU is sufficient for the small benchmark.

### Windows PowerShell

Run `setup_windows.bat` by double-clicking it, or from PowerShell:

```powershell
.\setup_windows.bat
```

It reuses `.venv312` if present, otherwise `.venv`, and creates `.venv` using `py -3.12` if needed. It installs packages through that environment's Python executable. **Do not activate the environment or use bare `python`**; Windows may resolve that to another installed version. Run commands below using `run_windows.bat`, which selects the environment explicitly.

### macOS / Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Use `.venv/bin/python` for all project commands so the interpreter and installed libraries always match.

## Run it

From the repository root, download data and train (the first run needs internet access):

```powershell
.\run_windows.bat train --epochs 15 --batch-size 32
```

On macOS/Linux, call `.venv/bin/python main.py train --epochs 15 --batch-size 32`.

Then evaluate on the pre-defined test split and save metrics/plots:

```powershell
.\run_windows.bat evaluate
```

On macOS/Linux, use `.venv/bin/python main.py evaluate`.

### Give the model an image to score

The model does not come with a personal `xray.png`; **you supply the image file**. Copy a permitted, non-identifying chest X-ray image into this project's `images` folder and name it `xray.png`, so the path is `images/xray.png`. PNG, JPG/JPEG, BMP, and TIFF are supported. DICOM files are not supported by this simple demo.

Then run this from the project folder:

```powershell
.\run_windows.bat predict --image "images\xray.png"
```

If the image is elsewhere, pass its full path in quotes, for example `.\run_windows.bat predict --image "C:\Users\YourName\Pictures\xray.png"`. The program converts the image to grayscale and resizes it to 28×28, then prints a pneumonia-class score and thresholded benchmark class. That lossy resize is only for this tiny benchmark; it is not suitable for clinical use. Do not use identifiable patient images.

On macOS/Linux, use `.venv/bin/python main.py predict --image "images/xray.png"`.

The output is a model score, not a diagnosis. The script resizes the image to the benchmark input size and converts it to grayscale. For the demo image output:

```text
outputs/
├── training_history.png
├── confusion_matrix.png
├── test_predictions.png
└── metrics.json
models/pneumonia_cnn.weights.h5
```

The training run prints loss and metrics each epoch. Results depend on software versions, random initialization, and hardware; there is no guaranteed accuracy target. `metrics.json` is the value to cite for your run. Never claim a result before running it.

## Evaluation notes

Accuracy alone can hide poor performance on a less common class. Inspect precision, recall/sensitivity, AUC, and the confusion matrix. The test split must not be used to tune architecture or thresholds. Patient-level independence and source dataset construction matter; this small curated benchmark does not validate generalization to other hospitals, populations, equipment, or image quality.

## Industry relevance and safety boundary

The project mirrors common ML engineering stages—dataset provenance, preprocessing, train/validation/test separation, model serialization, inference, metrics, and artifacts. Real deployment also requires clinical validation, representative multi-site data, privacy and security controls, calibration, workflow and human-factors assessment, monitoring, regulatory review, and governance. None is provided here. There is no real patient information or hospital integration in this repository.

## Learning outcomes

- Prepare image tensors and labels from a published split.
- Build and train a Keras CNN.
- Keep validation and test roles distinct.
- Evaluate beyond accuracy and communicate limitations.
- Save a model, metrics, plots, and a reproducible Git history.

## Dataset and citations

- MedMNIST project: https://github.com/MedMNIST/MedMNIST
- MedMNIST website: https://medmnist.com/
- Dataset distribution: https://doi.org/10.5281/zenodo.10519652
- MedMNIST v2 paper: https://doi.org/10.1038/s41597-022-01721-8

The data loader downloads the official PneumoniaMNIST `.npz` archive from Zenodo directly and checks its published MD5 checksum. It needs internet access on the first run. The dataset is not committed to Git. Confirm current dataset terms and original-source requirements before sharing data or derivative artifacts.

## GitHub publishing

Suggested repository name: `ai-medical-image-analysis`. Description: `Educational chest X-ray classification benchmark using PneumoniaMNIST and TensorFlow/Keras.` Topics: `computer-vision`, `medical-imaging`, `tensorflow`, `keras`, `pneumonia-detection`, `medmnist`, `machine-learning`.

```bash
git init
git add README.md requirements.txt .gitignore main.py src docs
git commit -m "docs: add project overview and workflow"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai-medical-image-analysis.git
git push -u origin main
```

After training, commit only selected reproducibility artifacts (metrics/plots if appropriate) and code. The ignore file excludes downloaded data, model binaries, and generated outputs by default. If you want to publish the model, check repository size and dataset terms first; consider GitHub Releases or an approved model registry. Do not upload patient information, private data, or unlicensed images. Add screenshots under `images/` and link them in this README after you have generated them.

## Seven-day proof plan

| Day | Work and suggested commit | Proof to capture |
| --- | --- | --- |
| 1 | Setup and project skeleton — `chore: create project structure` | Python version, environment and dependency installation |
| 2 | Dataset loader — `feat: add PneumoniaMNIST data loader` | Dataset split sizes and source link, no individual identifiable data |
| 3 | Preprocessing — `feat: normalize image tensors` | Sample grid and tensor shape/range |
| 4 | CNN — `feat: add baseline CNN` | Model summary and architecture diagram |
| 5 | Training — `feat: train baseline model` | Epoch log and training curve |
| 6 | Evaluation/inference — `feat: report held-out metrics` | Confusion matrix, metrics JSON and one demo prediction |
| 7 | Documentation — `docs: publish usage and limitations` | GitHub repository page, README rendered, clean clone instructions |

## Demo asset naming

Save reproducible, non-identifying artifacts as `images/sample_grid.png`, `outputs/training_history.png`, `outputs/confusion_matrix.png`, `outputs/test_predictions.png`, and `outputs/metrics.json`. Generated `outputs/` are ignored until deliberately selected. Put a small curated copy in `images/` only when useful for the README, and disclose how it was generated. Capture the repo preview after publishing; never present illustrative or unrun numbers as results.

For the detailed phase-by-phase runbook and common mistakes, see [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md). For the final proof checklist, see [docs/PROOF_CHECKLIST.md](docs/PROOF_CHECKLIST.md).

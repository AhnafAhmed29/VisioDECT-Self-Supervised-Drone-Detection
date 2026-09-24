# VidectDINO-26: Self-Supervised Representation Learning for Drone Detection

An experimental computer-vision framework for studying **supervised and self-supervised representation learning for multi-class drone detection** using the VisioDECT dataset.

This repository benchmarks conventional supervised object detectors against self-supervised representations learned with **SimCLR, BYOL, I-JEPA, and DINOv3**, followed by controlled downstream fine-tuning using limited labeled data. The project additionally studies **label efficiency, error patterns, explainability, and multi-object tracking** using the resulting drone detectors.

---

## Overview

Modern drone-detection systems generally depend on large quantities of annotated training data. Bounding-box annotation, however, is expensive and becomes particularly difficult when datasets contain multiple UAV models, environmental conditions, object scales, and lighting scenarios.

This project investigates whether self-supervised representation learning can provide useful initialization for drone detection when only a fraction of the available annotations is used.

The experimental pipeline contains four main stages:

1. **Dataset exploration and preprocessing**
2. **Fully supervised detector benchmarking**
3. **Self-supervised representation learning and downstream detection**
4. **Label-efficiency and multi-object tracking experiments**

The repository evaluates:

### Supervised Detectors

- YOLOv10
- YOLOv12
- YOLOv26
- RF-DETR

### Self-Supervised Representation Learning

- SimCLR
- BYOL
- I-JEPA
- DINOv3-based domain-adaptive distillation

### Downstream Analysis

- Reduced-label object detection
- Random-initialization baseline
- COCO-pretrained baseline
- Label-efficiency ablation
- Per-class and per-scenario error analysis
- Qualitative prediction analysis
- EigenCAM and ScoreCAM analysis where applicable
- ByteTrack multi-object tracking
- BoT-SORT with global motion compensation
- BoT-SORT with global motion compensation and ReID

---

## Dataset

Experiments use the **VisioDECT for Scenario-Based Multi-Drone Detection** dataset.

**Dataset:**  
https://www.kaggle.com/datasets/simeonajakwe/visiodect-for-scenario-based-multi-drone-detection

VisioDECT contains six rotary-wing UAV classes captured under multiple environmental conditions.

The classes used in this project are:

- Anafi
- DJI FPV
- DJI Phantom
- EFT E410S
- Mavic Air
- Mavic Enterprise

The dataset contains images from three primary scenarios:

- Sunny
- Cloudy
- Evening

The original VisioDECT release contains approximately 20.9k annotated drone images.

### Experimental Dataset Preparation

The preprocessing pipeline performs annotation validation, missing-label recovery, duplicate detection, class remapping, and leakage-safe dataset splitting.

After annotation recovery and filtering:

- 20,617 usable images were identified before duplicate removal.
- 14 byte-identical duplicate images were detected using MD5 hashes and removed.
- 20,603 unique images entered the final experimental split.

The resulting dataset used by the supervised experiments is:

| Split | Images |
|---|---:|
| Train | 14,422 |
| Validation | 3,090 |
| Test | 3,091 |
| **Total** | **20,603** |

The split is stratified across both **drone class and acquisition scenario**.

The dataset itself is **not redistributed through this repository**. Users should obtain VisioDECT from the original Kaggle source and comply with its original licensing conditions.

---

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── CITATION.cff
│
├── notebooks/
│   ├── 00_data_preparation/
│   │   └── VisioDECTnotebook1-datasetexploration-preprocessing.ipynb
│   │
│   ├── 01_supervised_detection/
│   │   ├── VisioDECTnotebook2-yolo-v10trainevaluate-erroranalysis.ipynb
│   │   ├── VisioDECTnotebook3-yolov12trainevaluate-erroranalysis.ipynb
│   │   ├── VisioDECTnotebook4-yolov26trainevaluate-erroranalysis.ipynb
│   │   └── VisioDECTnotebook5-rf-detrtrainevaluate-erroranalysis.ipynb
│   │
│   ├── 02_self_supervised_learning/
│   │   ├── simclr/
│   │   │   ├── VisioDECTnotebook1a-simclr-self-supervisedpretraining.ipynb
│   │   │   └── VisioDECTnotebook1b-simclr-self-superviseddownstream.ipynb
│   │   ├── byol/
│   │   │   ├── VisioDECTnotebook2a-byol-self-supervisedpretraining.ipynb
│   │   │   └── VisioDECTnotebook2b-byol-self-superviseddownstream.ipynb
│   │   ├── i_jepa/
│   │   │   ├── VisioDECTnotebook3a-i-jepa-self-supervisedpretraining.ipynb
│   │   │   └── VisioDECTnotebook3b-i-jepa-self-superviseddownstream.ipynb
│   │   └── dinov3/
│   │       ├── VisioDECTnotebook4a-dinov3-self-supervisedpretraining.ipynb
│   │       └── VisioDECTnotebook4b-dinov3-self-superviseddownstream.ipynb
│   │
│   ├── 03_ablation/
│   │   └── VisioDECTnotebookbonus-labelefficiency-selfsupervised.ipynb
│   │
│   └── 04_tracking/
│       └── VisioDECTnotebook5-downstreamtracking-self-supervised.ipynb
│
└── assets/
    └── figures/

```

---

## 1. Dataset Exploration and Preprocessing

The first notebook builds the common dataset used by the subsequent experiments.

The preprocessing pipeline includes:

- dataset integrity verification
- missing-label diagnosis
- annotation recovery
- bounding-box validation
- class and scenario analysis
- object-size analysis
- image-resolution analysis
- scenario photometric analysis
- MD5-based duplicate detection
- stratified train/validation/test splitting
- YOLO class-ID remapping
- augmentation verification

The final split is created **before augmentation**, preventing transformed versions of the same image from appearing across multiple dataset partitions.

MD5 hashing is additionally used to prevent byte-identical images from crossing train, validation, and test sets.

---

## 2. Supervised Drone Detection

Four supervised detector families are evaluated using the common train/validation/test protocol.

### YOLOv10

The YOLOv10 experiment includes:

- model training
- validation
- test-set evaluation
- per-class metrics
- error analysis
- confidence-threshold analysis
- inference benchmarking

Observed test performance:

| Metric | Value |
|---|---:|
| Precision | 0.9836 |
| Recall | 0.9780 |
| mAP@50 | 0.9871 |
| mAP@50:95 | 0.6466 |

### YOLOv12

Observed test performance:

| Metric | Value |
|---|---:|
| Precision | 0.9879 |
| Recall | 0.9885 |
| mAP@50 | 0.9905 |
| mAP@50:95 | 0.6534 |
| Single-image FPS | 53.5 |

### YOLOv26

Observed test performance:

| Metric | Value |
|---|---:|
| Precision | 0.9858 |
| Recall | 0.9835 |
| mAP@50 | 0.9898 |
| mAP@50:95 | 0.6489 |
| Single-image FPS | 67.9 |

### RF-DETR

Observed test performance:

| Metric | Value |
|---|---:|
| Precision | 0.9854 |
| Recall | 0.9848 |
| mAP@50 | 0.9797 |
| mAP@50:95 | 0.6298 |
| Single-image FPS | 32.9 |

These experiments establish the supervised reference points used in the later representation-learning experiments.

---

## 3. Self-Supervised Representation Learning

Four representation-learning strategies are studied.

### SimCLR

SimCLR learns representations using a contrastive objective over two augmented views of each image.

The VisioDECT SSL pool contains approximately **16,483 images**.

After self-supervised pretraining, the learned representation is transferred to the YOLOv26-s detector and fine-tuned using only 20% of the labeled training pool.

### BYOL

BYOL trains online and target networks using two augmented image views without requiring explicit negative pairs.

The learned backbone is transferred to the same downstream YOLOv26-s architecture under the shared fine-tuning protocol.

### I-JEPA

I-JEPA learns representations using masked latent prediction.

The resulting representation is transferred to YOLOv26-s and evaluated using the same labeled subset and downstream training protocol as the other SSL methods.

### DINOv3

The DINOv3 experiment uses a frozen DINOv3 ViT-B/16 teacher initialized from released LVD-1689M weights and performs domain-adaptive distillation on the VisioDECT image pool.

This experiment differs from the from-scratch SimCLR, BYOL, and I-JEPA experiments because DINOv3 inherits representations from large-scale external pretraining.

Results involving DINOv3 should therefore be interpreted as **domain-adaptive transfer rather than a strictly from-scratch SSL comparison**.

---

## 4. Reduced-Label Downstream Detection

Self-supervised representations are evaluated using a common **20% labeled-data budget**.

The downstream detector is YOLOv26-s, and the same training configuration, seed, image resolution, augmentation policy, evaluation thresholds, and labeled subset are maintained across initialization strategies.

The principal comparison is:

| Initialization | Test mAP@50:95 |
|---|---:|
| COCO pretrained | **0.6224** |
| I-JEPA | **0.6085** |
| DINOv3 | **0.6083** |
| BYOL | **0.6064** |
| SimCLR | **0.6041** |
| Random initialization | **0.5928** |

At the 20% label budget, all evaluated self-supervised initialization strategies outperform random initialization.

I-JEPA provides the highest mAP@50:95 among the evaluated SSL initialization strategies in this experiment, although I-JEPA and DINOv3 are nearly tied.

The COCO-pretrained detector remains the strongest initialization under the same 20% labeled-data setting.

---

## 5. Label-Efficiency Ablation

A separate experiment studies detector performance as the amount of labeled training data varies from **10% to 50%**.

The experiment compares:

- DINOv3-derived initialization
- COCO-pretrained initialization

using nested labeled subsets:

```text
10% ⊂ 20% ⊂ 30% ⊂ 40% ⊂ 50%
```

The same downstream training protocol is maintained across all label budgets.

### Test mAP@50:95

| Label Fraction | DINOv3 | COCO |
|---:|---:|---:|
| 10% | 0.5876 | 0.6066 |
| 20% | 0.6083 | 0.6224 |
| 30% | 0.6254 | 0.6344 |
| 40% | 0.6326 | 0.6432 |
| 50% | 0.6339 | 0.6456 |

The experiment demonstrates the effect of increasing annotation availability while preserving a controlled training protocol.

---

## 6. Multi-Object Tracking

The downstream tracking experiment applies the selected SSL-based detector to multi-drone video.

Three tracker configurations are evaluated:

- ByteTrack
- BoT-SORT with Global Motion Compensation
- BoT-SORT with Global Motion Compensation + ReID

The experiment measures:

- unique track IDs
- track births
- mean and median track duration
- track continuity
- fragmentation
- track gaps
- detection confidence
- active tracks per frame
- tracking/inference FPS

For the evaluated 264-frame video containing six expected drone objects:

| Tracker | Unique IDs | Mean Track Length | Continuity | FPS |
|---|---:|---:|---:|---:|
| ByteTrack | 7 | 7.58 s | 0.944 | 37.9 |
| BoT-SORT + GMC | 7 | 7.60 s | 0.945 | 16.4 |
| BoT-SORT + GMC + ReID | 13 | 4.20 s | 0.914 | 9.4 |

The tracking notebook also produces annotated videos, track-level CSV files, frame-level statistics, and visual diagnostics.

Because this evaluation uses a controlled downstream video rather than VisioDECT ground-truth tracking annotations, tracking statistics should be interpreted as comparative experimental diagnostics rather than benchmark MOT scores.

---

## Experimental Design

Several controls are used to improve reproducibility and fairness.

### Leakage Prevention

- dataset split before augmentation
- MD5-based duplicate detection
- disjoint train/validation/test partitions
- fixed random seeds
- class × scenario stratification

### Fair SSL Comparison

The downstream SSL experiments maintain the same:

- detector architecture
- label fraction
- random seed
- image resolution
- batch size
- number of epochs
- augmentation configuration
- optimizer policy
- validation split
- test split

Only the initialization strategy changes.

---

## Installation and Running the Notebooks

The experiments in this repository were primarily developed and evaluated using **Kaggle Notebooks with GPU acceleration**. Kaggle is therefore the recommended environment for reproducing the experiments.

The notebooks can also be executed locally using a Python virtual environment.

---

### Option 1 — Run on Kaggle (Recommended)

#### 1. Download or Fork the Notebook

Open the notebook you want to run from the `notebooks/` directory of this repository.

Download the corresponding `.ipynb` file and upload it to Kaggle:

1. Sign in to [Kaggle](https://www.kaggle.com/).
2. Select **Create → New Notebook**.
3. Use **File → Import Notebook** or upload the `.ipynb` file.
4. Open the imported notebook.

---

#### 2. Enable GPU Acceleration

The detection and self-supervised learning experiments are designed for GPU execution.

In the Kaggle notebook:

1. Open **Settings**.
2. Find **Accelerator**.
3. Select an available **GPU** accelerator.

Several of the original experiments were executed on NVIDIA Tesla T4 GPUs. Some self-supervised pretraining experiments used multiple GPUs when available.

CPU execution is possible for some preprocessing and analysis sections but is not recommended for model training.

---

#### 3. Enable Internet Access

Some notebooks install packages directly from PyPI or GitHub.

Enable:

```text
Settings → Internet → On
```

Internet access is required particularly for dependencies such as:

```text
ultralytics
rfdetr
lightly-train
ssl-detection-lab
```

and for downloading model weights when they are not already supplied as Kaggle inputs.

---

#### 4. Add the VisioDECT Dataset

All experiments use the **VisioDECT for Scenario-Based Multi-Drone Detection** dataset.

Dataset:

https://www.kaggle.com/datasets/simeonajakwe/visiodect-for-scenario-based-multi-drone-detection

Inside the Kaggle notebook:

1. Click **Add Input**.
2. Search for:

```text
VisioDECT for Scenario-Based Multi-Drone Detection
```

3. Select the dataset by **simeonajakwe**.
4. Click **Add**.

Kaggle will mount the dataset under:

```text
/kaggle/input/
```

The exact subdirectory name can be inspected using:

```python
from pathlib import Path

for path in Path("/kaggle/input").iterdir():
    print(path)
```

The VisioDECT dataset itself is not included in this GitHub repository.

---

#### 5. Install Dependencies

Most Kaggle notebooks already contain the package-installation commands required for their corresponding experiment.

For example, the YOLO experiments install:

```python
%pip install -q -U ultralytics
```

The RF-DETR experiment requires:

```python
%pip install -q -U "rfdetr[train]" supervision pycocotools torchmetrics faster-coco-eval
```

The DINOv3 experiment additionally uses:

```python
%pip install -q -U "lightly-train[ultralytics]>=0.16.2"
```

The SimCLR, BYOL, I-JEPA, and DINOv3 SSL experiments use:

```python
%pip install -q --no-cache-dir --force-reinstall --no-deps \
    "git+https://github.com/rifat963/ssl-detection-lab.git@main"
```

The tracking notebook also requires:

```python
%pip install -q -U ultralytics lap
```

For local execution, all primary dependencies are collected in the repository-level:

```text
requirements.txt
```

---

## Recommended Kaggle Execution Order

The notebooks are designed as a multi-stage experimental pipeline. Run them in the following order.

### Stage 1 — Dataset Preparation

Start with:

```text
notebooks/
└── 00_data_preparation/
    └── VisioDECTnotebook1-datasetexploration-preprocessing.ipynb
```

This notebook performs:

- dataset exploration
- annotation validation
- missing-label recovery
- duplicate detection
- leakage-safe splitting
- class remapping
- YOLO-format dataset construction

After successfully running the notebook, use:

```text
Save Version
```

in Kaggle so that the processed outputs can be reused by later notebooks.

---

### Stage 2 — Fully Supervised Detection

The supervised experiments can then be executed from:

```text
notebooks/
└── 01_supervised_detection/
    ├── VisioDECTnotebook2-yolo-v10trainevaluate-erroranalysis.ipynb
    ├── VisioDECTnotebook3-yolov12trainevaluate-erroranalysis.ipynb
    ├── VisioDECTnotebook4-yolov26trainevaluate-erroranalysis.ipynb
    └── VisioDECTnotebook5-rf-detrtrainevaluate-erroranalysis.ipynb
```

For each notebook:

1. Upload the notebook to Kaggle.
2. Add the original VisioDECT dataset or the required processed dataset output.
3. Enable GPU acceleration.
4. Enable Internet access.
5. Run all cells from top to bottom.

These notebooks train and evaluate the supervised reference models independently.

---

### Stage 3 — Self-Supervised Pretraining

Each SSL method has a pretraining notebook followed by a downstream detection notebook.

#### SimCLR

Run:

```text
02_self_supervised_learning/
└── simclr/
    ├── VisioDECTnotebook1a-simclr-self-supervisedpretraining.ipynb
    └── VisioDECTnotebook1b-simclr-self-superviseddownstream.ipynb
```

Run `01_pretraining.ipynb` first.

After training completes:

```text
Save Version
```

in Kaggle.

Then open `02_downstream_detection.ipynb` and add the saved output from the SimCLR pretraining notebook as an input.

---

#### BYOL

Run:

```text
02_self_supervised_learning/
└── byol/
    ├── VisioDECTnotebook2a-byol-self-supervisedpretraining.ipynb
    └── VisioDECTnotebook2b-byol-self-superviseddownstream.ipynb
```

The workflow is:

```text
BYOL Pretraining
        ↓
Save Kaggle Version
        ↓
Add Pretraining Output as Input
        ↓
BYOL Downstream Detection
```

---

#### I-JEPA

Run:

```text
02_self_supervised_learning/
└── i_jepa/
    ├── VisioDECTnotebook3a-i-jepa-self-supervisedpretraining.ipynb
    └── VisioDECTnotebook3b-i-jepa-self-superviseddownstream.ipynb
```

Again, save the pretraining notebook output before running downstream detection.

---

#### DINOv3

Run:

```text
02_self_supervised_learning/
└── dinov3/
    ├── VisioDECTnotebook4a-dinov3-self-supervisedpretraining.ipynb
    └── VisioDECTnotebook4b-dinov3-self-superviseddownstream.ipynb
```

The DINOv3 pretraining notebook requires the corresponding DINOv3 teacher weights in addition to the VisioDECT data.

After the domain-adaptive representation-learning stage finishes, save the Kaggle notebook version and attach its generated checkpoint/output to the downstream notebook.

The dependency relationship is:

```text
VisioDECT
    │
    ▼
DINOv3 SSL Pretraining
    │
    │  learned backbone/checkpoint
    ▼
DINOv3 Downstream Detection
    │
    ▼
Evaluation
```

---

### Stage 4 — Label-Efficiency Ablation

Run:

```text
notebooks/
└── 03_ablation/
    └── VisioDECTnotebookbonus-labelefficiency-selfsupervised.ipynb
```

This experiment uses nested labeled subsets ranging from 10% to 50% and compares DINOv3-derived initialization against COCO-pretrained initialization.

The required preprocessing and DINOv3 checkpoint outputs should therefore be added to the Kaggle notebook before execution.

---

### Stage 5 — Multi-Object Tracking

Finally, run:

```text
notebooks/
└── 04_tracking/
    └── VisioDECTnotebook5-downstreamtracking-self-supervised.ipynb
```

The tracking experiment requires:

- the trained downstream detector checkpoint
- the evaluation video
- Ultralytics
- `lap`

The principal workflow is:

```text
DINOv3 Pretraining
        ↓
Downstream Detector Training
        ↓
Detector Checkpoint
        ↓
Multi-Object Tracking
        ↓
ByteTrack / BoT-SORT Evaluation
```

Add the required saved Kaggle notebook output and tracking video as notebook inputs before execution.

---

## Kaggle Input and Output Workflow

Kaggle notebooks cannot directly access the temporary `/kaggle/working/` directory of another notebook.

To reuse a model checkpoint or processed dataset:

1. Run the producer notebook.
2. Ensure the required artifact is written to:

```text
/kaggle/working/
```

3. Click **Save Version**.
4. Wait for that Kaggle version to complete successfully.
5. Open the dependent notebook.
6. Select **Add Input**.
7. Choose **Your Work / Notebook Output Files**.
8. Add the saved notebook output.

The artifact then becomes available under a path similar to:

```text
/kaggle/input/notebooks/<username>/<notebook-name>/
```

This mechanism is used throughout the SSL pretraining → downstream detection pipeline.

---

## Local Installation

For users who want to run the project locally, clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/VisioDECT-SSL-Drone-Detection.git
cd VisioDECT-SSL-Drone-Detection
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Start JupyterLab:

```bash
jupyter lab
```

Then navigate to the appropriate notebook under:

```text
notebooks/
```

### Local GPU Note

PyTorch/CUDA compatibility depends on the installed NVIDIA driver, CUDA environment, operating system, and GPU.

For GPU-based local execution, users may need to install the appropriate PyTorch build for their system before installing the remaining dependencies.

The Kaggle environment is recommended when reproducing the GPU-intensive training experiments because it avoids most local CUDA configuration issues.

---

## Important Reproducibility Note

The notebooks were designed around the Kaggle filesystem:

```text
/kaggle/input/
```

for read-only datasets and notebook outputs, and:

```text
/kaggle/working/
```

for generated artifacts.

When running locally, these paths must be changed to the corresponding local dataset and output directories.

## Kaggle

The original experiments were designed to run in Kaggle GPU environments.

Dataset:

https://www.kaggle.com/datasets/simeonajakwe/visiodect-for-scenario-based-multi-drone-detection

Some computationally intensive notebooks were executed using Kaggle GPU accelerators and save their intermediate artifacts through Kaggle notebook outputs.

Local execution may require adjustments to dataset paths and GPU-dependent batch sizes.

---

## Reproducibility

Where applicable, experiments use fixed seeds and retain:

- dataset split manifests
- experiment configurations
- evaluation metrics
- model comparison tables
- training curves
- error-analysis outputs
- label-efficiency results
- tracking summaries

Large datasets, pretrained weights, trained checkpoints, and generated videos are intentionally excluded from the Git repository.

---

## Third-Party Models and Licensing

This repository relies on multiple third-party frameworks and model families.

The repository license applies only to original code and materials authored for this project.

Third-party software, pretrained models, model weights, and datasets remain governed by their respective licenses.

In particular:

- VisioDECT remains governed by the license specified by its original authors and dataset provider.
- Ultralytics YOLO software and associated models are subject to the applicable Ultralytics licensing terms.
- RF-DETR remains subject to its respective license.
- DINOv3 software and model weights remain subject to the DINOv3 License Agreement.

The VisioDECT dataset, third-party pretrained weights, and external model repositories are **not redistributed** as part of this repository.

---

## License

Original code and notebooks in this repository are released under the **GNU Affero General Public License v3.0 (AGPL-3.0)** unless otherwise noted.

Third-party dependencies, datasets, pretrained weights, and external assets retain their original licenses.

See [LICENSE](LICENSE) for details.

---

## Citation

If you use this repository in academic work, please cite both this repository and the original VisioDECT dataset/publication.

A `CITATION.cff` file is provided for repository citation.

---

## Acknowledgements

This project builds upon open research and software from the computer-vision community, including work on:

- VisioDECT
- Ultralytics YOLO
- RF-DETR
- SimCLR
- BYOL
- I-JEPA
- DINOv3
- ByteTrack
- BoT-SORT

Please refer to the corresponding original publications and repositories when using these methods.

---

## Disclaimer

This repository is intended for **academic research, benchmarking, and educational experimentation in computer vision**.

Results reported here correspond to the specific dataset partitions, hardware, training configurations, and evaluation protocols used in the included notebooks and should not be interpreted as universal performance guarantees.

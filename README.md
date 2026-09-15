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
│   │   └── 01_dataset_exploration_preprocessing.ipynb
│   │
│   ├── 01_supervised_detection/
│   │   ├── 01_yolov10_training_evaluation.ipynb
│   │   ├── 02_yolov12_training_evaluation.ipynb
│   │   ├── 03_yolov26_training_evaluation.ipynb
│   │   └── 04_rf_detr_training_evaluation.ipynb
│   │
│   ├── 02_self_supervised_learning/
│   │   ├── simclr/
│   │   │   ├── 01_pretraining.ipynb
│   │   │   └── 02_downstream_detection.ipynb
│   │   ├── byol/
│   │   │   ├── 01_pretraining.ipynb
│   │   │   └── 02_downstream_detection.ipynb
│   │   ├── i_jepa/
│   │   │   ├── 01_pretraining.ipynb
│   │   │   └── 02_downstream_detection.ipynb
│   │   └── dinov3/
│   │       ├── 01_pretraining.ipynb
│   │       └── 02_downstream_detection.ipynb
│   │
│   ├── 03_ablation/
│   │   └── 01_label_efficiency_10_to_50_percent.ipynb
│   │
│   └── 04_tracking/
│       └── 01_multi_object_tracking.ipynb
│
├── results/
│   ├── supervised/
│   ├── self_supervised/
│   ├── ablation/
│   └── tracking/
│
├── assets/
│   └── figures/
│
└── data/
    └── README.md
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

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/VisioDECT-SSL-Drone-Detection.git
cd VisioDECT-SSL-Drone-Detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Some experiments use additional packages installed directly inside their corresponding notebooks because particular model families require different environments.

---

## Main Dependencies

The project uses tools including:

- Python
- PyTorch
- Torchvision
- Ultralytics
- RF-DETR
- Albumentations
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow
- tqdm
- LightlyTrain
- pycocotools
- Supervision

Exact environment requirements may vary between the supervised, SSL, and RF-DETR experiments.

---

## Running the Experiments

The notebooks are intended to be executed in numerical order.

### Stage 1 — Prepare the Dataset

Run:

```text
notebooks/00_data_preparation/
    01_dataset_exploration_preprocessing.ipynb
```

This produces the cleaned train/validation/test partitions used by the detector experiments.

### Stage 2 — Supervised Detection

Run notebooks in:

```text
notebooks/01_supervised_detection/
```

### Stage 3 — Self-Supervised Learning

For each SSL method, execute the pretraining notebook before the corresponding downstream notebook.

Example:

```text
02_self_supervised_learning/dinov3/
├── 01_pretraining.ipynb
└── 02_downstream_detection.ipynb
```

### Stage 4 — Label-Efficiency Study

Run:

```text
notebooks/03_ablation/
    01_label_efficiency_10_to_50_percent.ipynb
```

### Stage 5 — Multi-Object Tracking

Run:

```text
notebooks/04_tracking/
    01_multi_object_tracking.ipynb
```

---

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

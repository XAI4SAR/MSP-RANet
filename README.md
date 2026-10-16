# MSP-RANet

English | [简体中文](README.zh-CN.md)

**Multi-Source Scattering Prior Injection and Routing for Fine-Grained SAR Aircraft Detection**

MSP-RANet targets fine-grained oriented aircraft detection in SAR images, extending YOLOv11 with multi-source scattering priors and feature routing. This repository provides the customized model, SAR input enhancement, and training and evaluation implementations. The instructions below use the 12-class OBB configuration for the FAIR-CSAR aircraft subset.

## Overview

- **PPIM**: Combines SAR input enhancement and DualStreamStem to fuse scattering keypoint and structure tensor priors with image features through residual attention.
- **SAR-Conv**: Uses RAPC2f to adaptively route features from orthogonal and diagonal branches, capturing directional scattering structures.
- **HNR**: Increases the margin between the correct class and the most easily confused incorrect class during training.

![MSP-RANet framework](docs/assets/msp-ranet-framework.jpg)

| Component | File |
| --- | --- |
| Model configuration (`s` scale, OBB head) | [`yolo11-obb-rapconv.yaml`](ultralytics/cfg/models/11/yolo11-obb-rapconv.yaml) |
| Dataset and SAR enhancement settings | [`coco_ence.yaml`](ultralytics/cfg/datasets/coco_ence.yaml) |
| DualStreamStem and RAPC2f implementations | [`block.py`](ultralytics/nn/modules/block.py) |
| HNR loss implementation | [`loss.py`](ultralytics/utils/loss.py) |

## Experimental Setup

The original experiments were conducted on a single NVIDIA GeForce RTX 4080 with the following hardware and software configuration.

| Item | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080, 16 GB VRAM |
| CPU | Intel Xeon Gold 6230R, 26 cores / 52 threads |
| Operating system | Ubuntu 24.04.3 LTS (WSL2) |
| NVIDIA driver | 572.70 |
| Python | 3.11.13 |
| PyTorch | 2.5.1+cu124 |
| torchvision / torchaudio | 0.20.1+cu124 / 2.5.1+cu124 |
| CUDA / cuDNN | 12.4 / 9.1.0 |
| Ultralytics | 8.3.111 (customized source in this repository) |

## Installation

After installing Conda and Git, run the following in a Linux or WSL2 Ubuntu terminal:

```bash
conda create -n msp-ranet python=3.11.13 pip=25.2 -y
conda activate msp-ranet
cd /path/to/yolov11_DualStreamStem
python -m pip install --no-deps -r requirement.txt
```

Replace the project path with your actual directory. Run all subsequent commands from the **project root**.

## Dataset Preparation

Download: [fair-csar-craft.zip (Baidu Netdisk)](https://pan.baidu.com/s/1qg6gXwKw8ufjgGo9tfBt0g?pwd=4ajt). Extraction code: `4ajt`.

Organize the dataset as follows, keeping the original split: **2,008** training images, **465** validation images, and **404** test images.

```text
datasets/sar_aircraft/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Image and label files must have matching stems. For example, `images/train/0001.jpg` corresponds to `labels/train/0001.txt`. Each annotation line describes an oriented bounding box with a class ID and four corner coordinates:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Class IDs start at `0`. Normalize horizontal coordinates by the image width and vertical coordinates by the image height to `[0, 1]`, with the four corners ordered along the box boundary.

Edit [`ultralytics/cfg/datasets/coco_ence.yaml`](ultralytics/cfg/datasets/coco_ence.yaml), replacing the existing `path`, `train`, `val`, and `test` entries with the following structure. Set `path` to the absolute path of your dataset root; the split paths are relative to it:

```yaml
path: /absolute/path/to/datasets/sar_aircraft
train: images/train
val: images/val
test: images/test

nc: 12
names:
  0: Airbus_A220
  1: Airbus_A320
  2: Airbus_A330
  3: Boeing737
  4: Boeing747
  5: Boeing767
  6: Boeing777
  7: Fokker-50
  8: Gulfstream
  9: Helicopter
  10: Airfreighter
  11: Other_Aircraft

sar_enhancement: true
sar_sigma_keypoint: 1.5
sar_sigma_tensor: 1.5
sar_clip_ratio: 2
```

## Training

After configuring the dataset paths, run [`train.py`](train.py):

```bash
python train.py
```

Weights directory: `run/fair-csar/v11/weights/`.

## Testing

Place the checkpoint to evaluate at `run/fair-csar/v11/weights/best.pt`, then run [`test.py`](test.py):

```bash
python test.py
```

Test results directory: `run/fair-csar/v11/test/`.

## Reference Results

Reference metrics from the original experiment:

| Split | mAP50 | mAP50–95 |
| --- | ---: | ---: |
| Validation | 80.053% | 63.566% |
| Test | 47.753% | 35.457% |

## Acknowledgements and License

This project builds on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). We thank the contributors for their open-source work. The project's AGPL-3.0 license declaration is provided in [`pyproject.toml`](pyproject.toml).

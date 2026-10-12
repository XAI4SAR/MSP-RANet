# MSP-RANet

**English** | [简体中文](README.zh-CN.md)

Implementation of **Multi-Source Scattering Prior Injection and Routing for Fine-Grained SAR Aircraft Detection**.

## 1. Introduction

MSP-RANet is a physics-guided framework for fine-grained aircraft detection in SAR imagery, built on YOLO11. It combines three components:

- **PPIM:** injects SAR intensity, Harris point cues and structure-tensor priors through residual attention.
- **SAR-Conv:** adaptively routes orthogonal and diagonal features to capture directional scattering structures.
- **HNR:** separates the ground-truth class from the most confusing incorrect class during training.

![MSP-RANet framework](docs/assets/msp-ranet-framework.jpg)

Results reported in the manuscript (**mAP@0.5**):

| Dataset | YOLOv11 | MSP-RANetv11 |
|---|---:|---:|
| SAR-AIRcraft-1.0 | 75.1% | **76.2%** |
| FAIR-CSAR-V1.0 aircraft subset | 41.5% | **47.7%** |

## 2. Getting Started

### 2.1 Installation

Install [anaconda](https://docs.conda.io/projects/miniconda/en/latest/), then run the following from a terminal. Replace the repository path with your own.

```bash
conda create -n msp-ranet python=3.10 -y
conda activate msp-ranet
git clone https://github.com/ikang2001/MSP-RANet.git
cd MSP-RANet
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The command above installs CUDA 12.1 wheels for an NVIDIA GPU with a compatible driver. For CPU use, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

Use the repository's customized `ultralytics/` source and run commands from the repository root. Do not install the PyPI `ultralytics` package. Check the environment:

```bash
python -c "import torch, ultralytics; print('CUDA:', torch.cuda.is_available()); print(ultralytics.__file__)"
```

The printed source path should point to this repository. The PyTorch version pair follows the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v251).

### 2.2 Data Preparation

**Dataset download:** [Baidu Netdisk](https://pan.baidu.com/s/19y-nOR7pdOqplnVipMYwfg?pwd=dirq), **access code: `dirq`**. Download and extract the data, arrange the images and labels as below, and update the dataset paths.

Datasets: [SAR-AIRcraft-1.0](https://www.radars.ac.cn/web/data/getData?newsColumnId=81053ef7-8327-4f09-88c0-ea061f64f9ba&pageType=en) and [FAIR-CSAR-V1.0](https://radars.ac.cn/web/data/getData?newsColumnId=59937a70-2a40-4027-a32d-df36753bf6ae&pageType=en). For FAIR-CSAR, use the 12-class aircraft subset of the Spotlight (SL) data.

Organize each dataset as follows; it may also be stored outside this repository:

```text
datasets/FAIR-CSAR/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Images and labels must have matching names, e.g. `images/train/001.png` and `labels/train/001.txt`. The default model uses **oriented bounding boxes (OBB)**, with one object per line:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
0 0.400000 0.350000 0.600000 0.350000 0.600000 0.650000 0.400000 0.650000
```

Coordinates are normalized to `[0, 1]`, and class IDs start at `0`. List the four corners in order around the rectangle. Convert original annotations to this nine-column OBB format first. Use raw grayscale SAR images; priors are generated online with `sar_enhancement: true`.

The default FAIR-CSAR dataset configuration is [coco_ence_csar.yaml](ultralytics/cfg/datasets/coco_ence_csar.yaml). Replace its existing machine-specific `path`, `train`, `val` and `test` entries with your own paths, for example:

```yaml
path: D:/datasets/FAIR-CSAR  # Linux/WSL example: /data/FAIR-CSAR
train: images/train
val: images/val
test: images/test
```

Keep the 12-class `names` mapping and SAR settings in this YAML, and verify that the class IDs match your OBB labels. Use the paper's training, validation and test splits for reproduction. Do not randomly re-split the data or include test images in training.

### 2.3 Training

The default model is [yolo11-obb-rapconv.yaml](ultralytics/cfg/models/11/yolo11-obb-rapconv.yaml), with an `OBB` head for oriented-box detection. The default dataset is [coco_ence_csar.yaml](ultralytics/cfg/datasets/coco_ence_csar.yaml) for the 12-class FAIR-CSAR aircraft subset.

**FAIR-CSAR:**

```bash
python train.py --model ultralytics/cfg/models/11/yolo11-obb-rapconv.yaml --data ultralytics/cfg/datasets/coco_ence_csar.yaml --epochs 200 --imgsz 1024 --batch 12 --device 0 --name msp_ranet_fair_csar
```

`--model` and `--data` may be omitted to use the defaults above. To use another dataset, pass its YAML through `--data` and provide matching OBB labels.

Training defaults to SGD and initialization from scratch; the default batch size is `12` for both training and testing. Checkpoints are saved to `runs/<name>/weights/best.pt` and `last.pt`. Repeated run names receive a numeric suffix.

Reduce `--batch` if GPU memory is insufficient. Use `--device cpu` for CPU execution, or `--workers 0` for Windows data-loader problems. To resume an interrupted run:

```bash
python train.py --data ultralytics/cfg/datasets/coco_ence_csar.yaml --model runs/msp_ranet_fair_csar/weights/last.pt --resume --device 0
```

Resume restores the saved training settings; use the same dataset as the original run.

### 2.4 Testing and Prediction Export

Evaluate the best checkpoint on the independent test split:

```bash
python test.py --data ultralytics/cfg/datasets/coco_ence_csar.yaml --weights runs/msp_ranet_fair_csar/weights/best.pt --split test --imgsz 1024 --batch 12 --device 0 --save-txt --save-conf --name msp_ranet_fair_test
```

Use `--split val` for validation. The console reports precision, recall, mAP@0.5 and mAP@0.5:0.95; results are saved to `runs/<name>/`. The command also exports predictions to `runs/msp_ranet_fair_test/labels/`, with each line formatted as `class_id x1 y1 x2 y2 x3 y3 x4 y4 confidence`. The default `--conf 0.001` is intended for AP evaluation. The run folder also contains confusion matrices and selected prediction-batch visualizations.

Run `python train.py --help` or `python test.py --help` for all options. Paper results above are manuscript-reported; see [implementation notes](docs/implementation-notes.md) for channel conventions, augmentation and HNR settings that need alignment for strict reproduction.

## 3. Citation

If this work is useful for your research, please cite the manuscript. Update the publication metadata when available.

```bibtex
@misc{chen_msp_ranet,
  title  = {Multi-Source Scattering Prior Injection and Routing for Fine-Grained SAR Aircraft Detection},
  author = {Chen, Jian and Jian, Weikang and Zhou, Xiaoding and Yuan, Jiayi and Liang, Ying and Qin, Yuxiao and Zong, Yali and Huang, Zhongling},
  note   = {Manuscript}
}
```

## 4. Acknowledgements

This implementation builds on [Ultralytics](https://github.com/ultralytics/ultralytics). See [LICENSE](LICENSE) for the code license.

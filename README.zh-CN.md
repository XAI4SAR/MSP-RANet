# MSP-RANet

[English](README.md) | **简体中文**

论文 **Multi-Source Scattering Prior Injection and Routing for Fine-Grained SAR Aircraft Detection** 的代码实现。

## 1. 简介

MSP-RANet 是基于 YOLO11 的物理先验引导细粒度 SAR 飞机检测框架，包含三个主要模块：

- **PPIM：** 通过残差注意力注入 SAR 强度、Harris 点状线索和结构张量先验。
- **SAR-Conv：** 自适应路由正交与对角特征，提取不同方向的散射结构。
- **HNR：** 在训练中拉开真实类别与最易混淆错误类别的间隔。

![MSP-RANet 框架图](docs/assets/msp-ranet-framework.jpg)

论文报告的实验结果（**mAP@0.5**）：

| 数据集 | YOLOv11 | MSP-RANetv11 |
|---|---:|---:|
| SAR-AIRcraft-1.0 | 75.1% | **76.2%** |
| FAIR-CSAR-V1.0 飞机子集 | 41.5% | **47.7%** |

## 2. 开始使用

### 2.1 环境安装

先安装 [anaconda](https://docs.conda.io/projects/miniconda/en/latest/)，再在终端执行以下命令。请将项目路径替换为自己的实际路径。

```bash
conda create -n msp-ranet python=3.10 -y
conda activate msp-ranet
git clone https://github.com/ikang2001/MSP-RANet.git
cd MSP-RANet
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

上述命令安装 CUDA 12.1 版 PyTorch，适用于具有兼容驱动的 NVIDIA 显卡。仅使用 CPU 时，将 PyTorch 安装命令替换为：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

本项目使用仓库内自定义的 `ultralytics/` 源码，请在项目根目录运行命令，不要安装 PyPI 上的 `ultralytics` 包。检查环境：

```bash
python -c "import torch, ultralytics; print('CUDA:', torch.cuda.is_available()); print(ultralytics.__file__)"
```

输出的源码路径应指向当前项目。PyTorch 版本组合参考[官方安装说明](https://pytorch.org/get-started/previous-versions/#v251)。

### 2.2 数据准备

**数据集下载：** [百度网盘](https://pan.baidu.com/s/19y-nOR7pdOqplnVipMYwfg?pwd=dirq)，**提取码：`dirq`**。下载并解压后，按下方结构放置图像与标签，再修改数据配置中的路径。

数据集：[SAR-AIRcraft-1.0](https://www.radars.ac.cn/web/data/getData?newsColumnId=81053ef7-8327-4f09-88c0-ea061f64f9ba&pageType=en)、[FAIR-CSAR-V1.0](https://radars.ac.cn/web/data/getData?newsColumnId=59937a70-2a40-4027-a32d-df36753bf6ae&pageType=en)。FAIR-CSAR 使用 Spotlight（SL）部分的 12 类飞机子集。

每个数据集按下面结构摆放，也可以存放在项目之外：

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

图像与标签必须同名，例如 `images/train/001.png` 对应 `labels/train/001.txt`。默认模型使用**旋转框（OBB）**，每行标注一个目标：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
0 0.400000 0.350000 0.600000 0.350000 0.600000 0.650000 0.400000 0.650000
```

坐标归一化到 `[0, 1]`，类别编号从 `0` 开始。四个顶点应沿矩形边界依次排列，原始标注应先转换为上述九列 OBB 格式。输入使用原始灰度 SAR 图像，开启 `sar_enhancement: true` 后会在线生成先验。

默认使用 FAIR-CSAR 数据配置 [coco_ence_csar.yaml](ultralytics/cfg/datasets/coco_ence_csar.yaml)。将其中原机器的 `path`、`train`、`val`、`test` 路径改为自己的路径，例如：

```yaml
path: D:/datasets/FAIR-CSAR  # Linux/WSL example: /data/FAIR-CSAR
train: images/train
val: images/val
test: images/test
```

保留该 YAML 中的 12 类 `names` 和 SAR 配置，并确认类别编号与 OBB 标签对应。复现时保持与论文一致的训练、验证、测试划分，避免重新随机切分或将测试图像用于训练。

### 2.3 模型训练

模型配置为 [yolo11-obb-rapconv.yaml](ultralytics/cfg/models/11/yolo11-obb-rapconv.yaml)，FAIR-CSAR 数据配置为 [coco_ence_csar.yaml](ultralytics/cfg/datasets/coco_ence_csar.yaml)。

在项目根目录直接运行原始训练脚本：

```bash
python train.py
```

`train.py` 中的训练参数为：`epochs=200`、`batch=12`、`imgsz=1024`、`optimizer="SGD"`、`amp=False`、`pretrained=False`、`patience=200`。需要调整时直接编辑脚本中的对应值。数据增强沿用原始源码默认设置。

权重保存在 `runs/train/weights/best.pt` 和 `runs/train/weights/last.pt`。重复实验名称会自动添加数字后缀，请以训练日志中的实际输出目录为准。

### 2.4 模型测试

原始 `test.py` 加载 `weight/fair-csar/v11/weights/best.pt`。测试上一步训练得到的模型时，将其中的模型路径改为：

```python
model = YOLO("runs/train/weights/best.pt")
```

然后在项目根目录执行：

```bash
python test.py
```

原始测试参数为：`split="test"`、`imgsz=1024`、`batch=8`、`device="cuda:0"`、`conf=0.2`、`save=True`、`save_txt=True`。需要调整权重、数据路径或其他参数时直接编辑 `test.py`；验证集评估将 `split` 改为 `"val"`。

控制台输出 Precision、Recall、mAP@0.5 和 mAP@0.5:0.95。结果保存在 `runs/test/`，预测标签保存在 `runs/test/labels/`，不包含置信度数值；同时保存指标图和部分 batch 的预测可视化。重复实验名称会自动添加数字后缀。

数据集下载和论文精度见上文，严格复现涉及的实现细节见[实现说明](docs/implementation-notes.md)。

## 3. 引用

如果本工作对你的研究有帮助，请引用论文。正式发表信息确定后更新引用条目。

```bibtex
@misc{chen_msp_ranet,
  title  = {Multi-Source Scattering Prior Injection and Routing for Fine-Grained SAR Aircraft Detection},
  author = {Chen, Jian and Jian, Weikang and Zhou, Xiaoding and Yuan, Jiayi and Liang, Ying and Qin, Yuxiao and Zong, Yali and Huang, Zhongling},
  note   = {Manuscript}
}
```

## 4. 致谢

本实现基于 [Ultralytics](https://github.com/ultralytics/ultralytics)。代码许可参见 [LICENSE](LICENSE)。

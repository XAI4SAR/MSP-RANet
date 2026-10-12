# models/PGD_yolo.py

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import initialize_weights
import torch
import torch.nn as nn

class PGDModel(DetectionModel):
    """YOLOv11 OBB 模型 + PGIP 分支（用于位置感知）"""

    def __init__(self, cfg='yolo11-obb.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg=cfg, ch=ch, nc=nc, anchors=anchors)

        # 新增 PGIP 分支（用于位置感知）
        self.pgip_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 1, kernel_size=1)  # 输出单通道位置感知图
            ) for _ in range(len(self.backbone.out_channels))
        ])

        # 初始化权重
        initialize_weights(self.pgip_branch)

    def forward(self, x):
        # 前向传播返回双分支输出 (all_pre, loc_pre)
        all_pre = super().forward(x)  # 标准 OBB 检测输出

        # 获取 backbone 多层级特征图
        backbone_feats = self.backbone(x)

        # 通过 PGIP 分支生成位置感知特征
        loc_pre = [self.pgip_branch[i](f) for i, f in enumerate(backbone_feats)]

        return all_pre, loc_pre  # 返回双分支输出
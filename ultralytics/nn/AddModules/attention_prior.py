# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Physical Prior Attention Module for SAR Image Enhancement

This module provides lightweight attention branches that convert
physical feature priors (Harris corners + Structure tensor) into
attention maps for fusion with the main backbone features.

Architecture:
    Physical Prior (2ch: Harris + StructureTensor)
        → Lightweight Conv Branch
        → Attention Map (0-1)
        → Feature Fusion (Multiply or Concat)
"""

import torch
import torch.nn as nn


class PhysicalPriorBranch(nn.Module):
    """
    Lightweight Physical Prior Branch for Attention Prior.

    Converts physical feature prior map (2 channels: Harris response + Structure tensor)
    into a spatial attention map that highlights strong scattering regions.

    This branch is designed to be extremely lightweight (< 5% of main backbone params)
    to minimize computational overhead while providing meaningful prior guidance.
    """

    def __init__(self, in_channels=2, mid_channels=16, out_channels=1):
        """
        Args:
            in_channels: Number of input channels (2 for Harris + StructureTensor)
            mid_channels: Number of middle layer channels (kept small for lightweight design)
            out_channels: Number of output channels (1 for attention map)
        """
        super().__init__()
        # Extremely lightweight: only 2 conv layers
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Physical prior tensor, shape (B, 2, H, W)

        Returns:
            Attention map, shape (B, 1, H, W), values in [0, 1]
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.sigmoid(self.conv2(x))
        return x


class PriorFusion(nn.Module):
    """
    Prior-guided Feature Fusion Module.

    Fuses backbone features with physical prior attention map.
    Supports two fusion modes:
        - 'multiply': Feature * Attention (soft gating)
        - 'concat': Concat(Feature, Attention) along channel dim
    """

    def __init__(self, channels, prior_channels=2, fusion_mode='multiply'):
        """
        Args:
            channels: Number of backbone feature channels
            prior_channels: Number of prior input channels
            fusion_mode: 'multiply' or 'concat'
        """
        super().__init__()
        self.fusion_mode = fusion_mode
        self.prior_branch = PhysicalPriorBranch(in_channels=prior_channels, mid_channels=16, out_channels=1)

        if fusion_mode == 'concat':
            # For concat mode, we need to match dimensions
            self.conv_fuse = nn.Conv2d(channels + 1, channels, 1, bias=False)
            self.bn_fuse = nn.BatchNorm2d(channels)

    def forward(self, x, prior_map):
        """
        Args:
            x: Backbone features, shape (B, C, H, W)
            prior_map: Physical prior map, shape (B, 2, H, W)

        Returns:
            Fused features, shape (B, C, H, W)
        """
        # Generate attention from prior
        attention = self.prior_branch(prior_map)  # (B, 1, H, W)

        # Ensure attention matches feature spatial size
        if attention.shape[2:] != x.shape[2:]:
            attention = nn.functional.interpolate(
                attention, size=x.shape[2:], mode='bilinear', align_corners=False
            )

        if self.fusion_mode == 'multiply':
            # Soft gating: multiply features with attention
            return x * attention
        elif self.fusion_mode == 'concat':
            # Concat along channel dimension
            x = torch.cat([x, attention], dim=1)  # (B, C+1, H, W)
            x = self.relu(self.bn_fuse(self.conv_fuse(x)))
            return x
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")


class MultiScalePriorFusion(nn.Module):
    """
    Multi-scale Prior Fusion for handling features at different resolutions.

    Interpolates prior attention map to match multiple feature scales.
    """

    def __init__(self, channels_list, prior_channels=2, fusion_mode='multiply'):
        """
        Args:
            channels_list: List of channel numbers for each feature level
            prior_channels: Number of prior input channels
            fusion_mode: 'multiply' or 'concat'
        """
        super().__init__()
        self.fusion_layers = nn.ModuleList([
            PriorFusion(c, prior_channels, fusion_mode) for c in channels_list
        ])

    def forward(self, features, prior_map):
        """
        Args:
            features: List of feature tensors at different scales
            prior_map: Physical prior map, shape (B, 2, H, W)

        Returns:
            List of fused features
        """
        return [fusion_layer(feat, prior_map) for fusion_layer, feat in zip(self.fusion_layers, features)]

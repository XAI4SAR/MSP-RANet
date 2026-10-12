# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
SAR Image Enhancement Module

This module provides SAR (Synthetic Aperture Radar) image enhancement through 
three-channel physical feature extraction:

- R Channel: Scattering Keypoint Heatmap (Harris corner detection)
- G Channel: Structure Tensor Coherence Map (anisotropy)
- B Channel: Adaptive Log-Amplitude Map (dynamic range compression)

Based on the original implementation from /mnt/e/jwk/yuchuli/ronghe1_0.py
"""

import cv2
import numpy as np


def normalize_to_uint8(img):
    """
    Normalize an image to uint8 range [0, 255].
    
    Args:
        img: Input image (any range)
        
    Returns:
        np.ndarray: Normalized image in uint8
    """
    # img_min = img.min()
    # img_max = img.max() + 1e-8  # Prevent division by zero
    # norm = (img - img_min) / (img_max - img_min)
    # return (norm * 255).astype(np.uint8)

    # 修复RuntimeWarning 问题；原因：np.power() 对负数或 NaN 值进行幂运算时会触发警告。
    img_min = img.min()
    img_max = img.max() + 1e-8
    norm = (img - img_min) / (img_max - img_min)
    norm = np.clip(norm, 0, 1)  # 添加：确保在 [0, 1] 范围内
    return (norm * 255).astype(np.uint8)

# def generate_scattering_heatmap(sar_img, block_size=2, ksize=3, k=0.04, sigma=15):
#     """
#     Generate scattering keypoint heatmap using Harris corner detection.
    
#     R Channel: Highlights strong scattering points (nose, wingtips, etc.)
    
#     Args:
#         sar_img: Input SAR image (grayscale, single channel)
#         block_size: Harris corner detector block size
#         ksize: Sobel kernel size
#         k: Harris corner detector free parameter
#         sigma: Gaussian blur radius for heatmap diffusion
        
#     Returns:
#         np.ndarray: Heatmap as float32 array
#     """
#     # Preprocess: normalize and convert to uint8 for cornerHarris
#     src = normalize_to_uint8(sar_img)
    
#     # Harris corner detection
#     # dst returns corner response - higher values indicate stronger corners
#     dst = cv2.cornerHarris(src, block_size, ksize, k)
    
#     # Threshold selection and binarization
#     # Keep only top 1% response values as strong scattering keypoints
#     threshold = 0.01 * dst.max()
#     keypoints_mask = np.zeros_like(sar_img, dtype=np.float32)
#     keypoints_mask[dst > threshold] = 255
    
#     # Generate Gaussian heatmap
#     # Spread discrete points into continuous heat regions
#     heatmap = cv2.GaussianBlur(keypoints_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
#     return heatmap


def generate_scattering_heatmap(sar_img, block_size=2, ksize=3, k=0.04, sigma=2):
    """
    Generate scattering keypoint heatmap using Harris corner detection.
    
    R Channel: Highlights strong scattering points (nose, wingtips, etc.)
    
    Args:
        sar_img: Input SAR image (grayscale, single channel)
        block_size: Harris corner detector block size
        ksize: Sobel kernel size
        k: Harris corner detector free parameter
        sigma: Gaussian blur radius for heatmap diffusion (reduced from 15 to 2)
        
    Returns:
        np.ndarray: Heatmap as float32 array
    """
    # Preprocess: normalize and convert to uint8 for cornerHarris
    src = normalize_to_uint8(sar_img)
    
    # Harris corner detection
    dst = cv2.cornerHarris(src, block_size, ksize, k)
    
    # 修改1：去掉1%阈值截断，保留完整响应分布
    keypoints_mask = dst  # 直接使用原始响应值
    
    # 修改2：降低sigma，保留锐利结构
    heatmap = cv2.GaussianBlur(keypoints_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    return heatmap

# 修改1：相干度，但是会把飞机淹没无法形成伦奎，合成的RGB图像不理想，
# def generate_structure_tensor_map(sar_img, sigma_grad=1.0, sigma_tensor=3.0):
#     """
#     Generate structure tensor coherence map using anisotropy analysis.
    
#     G Channel: Distinguishes artificial edges (linear structures) from natural noise
    
#     Args:
#         sar_img: Input SAR image (grayscale, single channel)
#         sigma_grad: Gaussian smoothing scale for gradient computation
#         sigma_tensor: Gaussian smoothing scale for structure tensor integration
        
#     Returns:
#         np.ndarray: Coherence map as float32 array (values near 1 = strong linear structure)
#     """
#     img = sar_img.astype(np.float32)
    
#     # Compute gradients (Ix, Iy) using Sobel operator
#     grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
#     # Compute structure tensor components (J = gradient * gradient^T)
#     Ixx = grad_x ** 2
#     Iyy = grad_y ** 2
#     Ixy = grad_x * grad_y
    
#     # Gaussian weighted smoothing of components
#     # This is the core of structure tensor - each pixel's tensor represents neighborhood statistics
#     # 单个像素的梯度容易受噪声影响。高斯平滑相当于对每个像素周围的 sigma_tensor 大小邻域 取加权平均。
#     Txx = cv2.GaussianBlur(Ixx, (0, 0), sigmaX=sigma_tensor, sigmaY=sigma_tensor)
#     Tyy = cv2.GaussianBlur(Iyy, (0, 0), sigmaX=sigma_tensor, sigmaY=sigma_tensor)
#     Txy = cv2.GaussianBlur(Ixy, (0, 0), sigmaX=sigma_tensor, sigmaY=sigma_tensor)
    
#     # Compute eigenvalues (lambda1, lambda2)
#     # For 2x2 matrix [[a, b], [b, c]], eigenvalues have analytical solution
#     # Trace = Txx + Tyy
#     # Det = Txx * Tyy - Txy^2
#     # Delta = sqrt(Trace^2 - 4*Det) = sqrt((Txx - Tyy)^2 + 4*Txy^2)

#     # 计算特征值  J = | Txx  Txy |
#     #              | Txy  Tyy |
#     # 对于 2×2 矩阵，特征值的解析解为：λ = (trace ± sqrt(trace² - 4×det)) / 2
#     # 其中，trace² - 4×det
#     #       = (Txx + Tyy)² - 4×(Txx×Tyy - Txy²)
#     #       = Txx² + 2×Txx×Tyy + Tyy² - 4×Txx×Tyy + 4×Txy²
#     #       = Txx² - 2×Txx×Tyy + Tyy² + 4×Txy²
#     #       = (Txx - Tyy)² + 4×Txy²
#     delta = np.sqrt((Txx - Tyy) ** 2 + 4 * Txy ** 2)
#     trace = Txx + Tyy
#     lambda1 = (trace + delta) / 2
#     lambda2 = (trace - delta) / 2
    
#     # Compute coherence / anisotropy
#     # C = (lambda1 - lambda2) / (lambda1 + lambda2 + epsilon)
#     # C near 1 = strong linear structure (edges, fuselage)
#     # C near 0 = isotropic (noise, smooth regions)
#     coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
    
#     return coherence

# 修改2：相干度，但加上能量权重，可以更好的形成轮廓
def generate_structure_tensor_map(sar_img, sigma_grad=1.0, sigma_tensor=3.0):
    """
    G 通道: 能量加权的结构张量相干图
    引入总能量 (Trace) 压制背景噪声的虚假边缘
    
    Args:
        sar_img: 输入SAR图像（灰度单通道）
        sigma_grad: 梯度计算的高斯平滑尺度
        sigma_tensor: 结构张量的高斯平滑尺度
        
    Returns:
        np.ndarray: 能量加权相干图，shape为(H, W)
    """
    img = sar_img.astype(np.float32)
    
    # 计算梯度
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    # 结构张量分量
    Ixx = grad_x ** 2
    Iyy = grad_y ** 2
    Ixy = grad_x * grad_y
    
    # 高斯平滑
    Txx = cv2.GaussianBlur(Ixx, (0, 0), sigmaX=sigma_tensor, sigmaY=sigma_tensor)
    Tyy = cv2.GaussianBlur(Iyy, (0, 0), sigmaX=sigma_tensor, sigmaY=sigma_tensor)
    Txy = cv2.GaussianBlur(Ixy, (0, 0), sigmaX=sigma_tensor, sigmaY=sigma_tensor)
    
    # 计算特征值
    delta = np.sqrt((Txx - Tyy) ** 2 + 4 * Txy ** 2)
    trace = Txx + Tyy  # Trace 代表局部总梯度能量
    
    lambda1 = (trace + delta) / 2
    lambda2 = (trace - delta) / 2
    
    # 1. 纯几何相干度 (0~1)
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
    
    # 2. 能量掩膜 (Energy Mask) - 核心救命操作，压制背景噪声
    # trace_norm = cv2.normalize(trace, None, 0, 1, cv2.NORM_MINMAX)
    # energy_weight = np.power(trace_norm, 1.5) 
    ############
    ##########
    ############
    #########
    # 修复RuntimeWarning 问题；原因：np.power() 对负数或 NaN 值进行幂运算时会触发警告。
    trace_norm = cv2.normalize(trace, None, 0, 1, cv2.NORM_MINMAX)
    trace_norm = np.clip(trace_norm, 0, None)  # 添加：确保非负，处理 NaN
    energy_weight = np.power(trace_norm, 1.5)

###########
##########
##############
##############

    # 3. 加权输出：既要是线(coherence大)，又得有能量(energy大)
    weighted_coherence = coherence * energy_weight
    
    return weighted_coherence


##B通道 修改1 log对数变换
# def generate_log_amplitude_map(sar_img, clip_ratio=0.01):
#     """
#     Generate adaptive log-amplitude map with robust stretching.
    
#     B Channel: Preserves original intensity information, suppresses large dynamic range
    
#     Args:
#         sar_img: Input SAR image (grayscale, single channel)
#         clip_ratio: Truncation ratio for removing extreme noise points
        
#     Returns:
#         np.ndarray: Log-amplitude map as float32 array
#     """
#     img = sar_img.astype(np.float32)
    
#     # Optional: Simple denoising (e.g., median filter for salt-and-pepper noise)
#     # img = cv2.medianBlur(img.astype(np.uint8), 3).astype(np.float32)
    
#     # Log transformation
#     # log(1 + x) compresses dynamic range, reveals texture in dark areas
#     img_log = np.log1p(img)
    
#     # Robust stretching
#     # Truncate extremely bright and dark outliers to prevent contrast reduction
#     v_min = np.percentile(img_log, clip_ratio * 100)
#     v_max = np.percentile(img_log, (1 - clip_ratio) * 100)
#     img_clipped = np.clip(img_log, v_min, v_max)
    
#     return img_clipped


def generate_log_amplitude_map(sar_img, clip_ratio=0.01):
    """
    Generate adaptive log-amplitude map with robust stretching.
    
    B Channel: Preserves original intensity information, suppresses large dynamic range
    
    Args:
        sar_img: Input SAR image (grayscale, single channel)
        clip_ratio: Truncation ratio for removing extreme noise points
        
    Returns:
        np.ndarray: Log-amplitude map as float32 array
    """
    img = sar_img.astype(np.float32)
    
    # Optional: Simple denoising (e.g., median filter for salt-and-pepper noise)
    # img = cv2.medianBlur(img.astype(np.uint8), 3).astype(np.float32)
    
    # Log transformation
    # log(1 + x) compresses dynamic range, reveals texture in dark areas
    img_log = np.log1p(img)
    
    # Robust stretching
    # Truncate extremely bright and dark outliers to prevent contrast reduction
    v_min = np.percentile(img_log, clip_ratio * 100)
    v_max = np.percentile(img_log, (1 - clip_ratio) * 100)
    img_clipped = np.clip(img_log, v_min, v_max)
    
    return img_clipped


def enhance_sar_image(sar_img, sigma_keypoint=2, sigma_tensor=0.5, clip_ratio=0.01):
    """
    One-stop SAR image enhancement interface.
    
    Converts single-channel SAR image to three-channel RGB enhanced image:
    - R: Scattering keypoint heatmap (强散射点/角点)
    - G: Structure tensor coherence map (结构张量相干图/轮廓)
    - B: Original SAR Image (原始单通道SAR底图)
    
    Args:
        sar_img: Single-channel SAR image with shape (H, W) or (H, W, 1)
        sigma_keypoint: Gaussian diffusion radius for keypoint heatmap
        sigma_tensor: Smoothing scale for structure tensor
        clip_ratio: (这里原用于B通道，如果不用Gamma，这个参数可以保留占位或删除)
        
    Returns:
        np.ndarray: Three-channel enhanced RGB/BGR image with shape (H, W, 3)
                    Note: Uses BGR order for OpenCV compatibility
    """
    # Ensure single channel
    if len(sar_img.shape) == 3:
        if sar_img.shape[2] == 3:
            # Convert RGB/BGR to grayscale
            sar_img_gray = cv2.cvtColor(sar_img, cv2.COLOR_BGR2GRAY)
        else:
            sar_img_gray = sar_img[:, :, 0]
    else:
        sar_img_gray = sar_img
    
    # Generate R and G channels
    channel_r = generate_scattering_heatmap(sar_img_gray, sigma=sigma_keypoint)
    channel_g = generate_structure_tensor_map(sar_img_gray, sigma_tensor=sigma_tensor)

    # ==========================================
    # B 通道核心修改：直接使用提取出的原始灰度图
    # ==========================================
    channel_b = sar_img_gray 

    # Normalize each channel to uint8
    R = normalize_to_uint8(channel_r)
    G = normalize_to_uint8(channel_g)
    B = normalize_to_uint8(channel_b)  # 这里会自动将原始SAR图像拉伸到 0-255，防止合并时报错
    
    # Merge to BGR format (OpenCV default)
    enhanced_img = cv2.merge([B, G, R])
    
    return enhanced_img


def is_sar_image(img):
    """
    Detect if an image is a single-channel SAR image or grayscale RGB image.
    
    Args:
        img: Input image array
        
    Returns:
        bool: True if SAR image (single-channel or grayscale RGB)
    """
    if img is None:
        return False
    if len(img.shape) == 2:  # pyright: ignore[reportUnreachable]
        return True  # (H, W) - single channel
    
    if len(img.shape) == 3 and img.shape[2] == 1:
        return True  # (H, W, 1) - single channel
    
    # 新增：识别三通道灰度图 (R=G=B)
    if len(img.shape) == 3 and img.shape[2] == 3:
        # 检查三个通道是否相等（灰度图）
        if img.dtype == np.uint8:
            is_gray = np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,1], img[:,:,2])
        else:
            is_gray = np.allclose(img[:,:,0], img[:,:,1], atol=1e-5) and np.allclose(img[:,:,1], img[:,:,2], atol=1e-5)
        if is_gray:
            return True
    
    return False

def generate_physical_prior(sar_img, sigma_keypoint=2, sigma_tensor=0.5):
    """
    生成物理特征先验图（用于Attention Prior双流架构）

    将Harris角点响应和结构张量相干度融合为2通道先验图:
    - Channel 0: Harris角点响应（散射点/角点）
    - Channel 1: 结构张量相干度（边缘/轮廓）

    Args:
        sar_img: 输入SAR图像（灰度单通道或RGB图像）
        sigma_keypoint: Harris角点检测的高斯模糊半径
        sigma_tensor: 结构张量的高斯平滑尺度

    Returns:
        np.ndarray: 2通道先验图，shape为(2, H, W)，数据类型为uint8
    """
    # Ensure grayscale
    if len(sar_img.shape) == 3:
        if sar_img.shape[2] == 3:
            sar_img_gray = cv2.cvtColor(sar_img, cv2.COLOR_BGR2GRAY)
        else:
            sar_img_gray = sar_img[:, :, 0]
    else:
        sar_img_gray = sar_img

    # R通道: Harris角点响应
    channel_r = generate_scattering_heatmap(sar_img_gray, sigma=sigma_keypoint)
    # G通道: 结构张量相干度
    channel_g = generate_structure_tensor_map(sar_img_gray, sigma_tensor=sigma_tensor)

    # 归一化到uint8
    R = normalize_to_uint8(channel_r)
    G = normalize_to_uint8(channel_g)

    # 堆叠为2通道先验图 (C, H, W)
    prior_map = np.stack([R, G], axis=0)

    return prior_map


def generate_sharp_scattering_heatmap(sar_img, block_size=2, ksize=3, k=0.04, sigma=1.0, threshold_ratio=0.005):
    """
    G通道（先验流1）：提取极其锐利的强散射点（绝不大范围模糊）

    与旧版 generate_scattering_heatmap() 的关键区别：
    1. 阈值更低 (0.005 vs 0.01)，保留更多弱散射点
    2. sigma极小 (1.0 vs 2.0)，仅做微弱扩散，保留尖锐拓扑
    3. 专为 DualStreamStem 的先验引导分支设计

    Args:
        sar_img: 输入SAR图像（灰度单通道）
        block_size: Harris角点检测的块大小
        ksize: Sobel算子核大小
        k: Harris角点自由参数
        sigma: 高斯模糊半径（极小值，仅微弱扩散）
        threshold_ratio: 阈值比例（低于此比例的最大值被忽略）

    Returns:
        np.ndarray: 锐化热图，float32数组
    """
    src = normalize_to_uint8(sar_img)
    dst = cv2.cornerHarris(src, block_size, ksize, k)

    # 降低阈值，保留更多弱散射点（防止细粒度丢失）
    threshold = threshold_ratio * dst.max()
    keypoints_mask = np.zeros_like(sar_img, dtype=np.float32)
    keypoints_mask[dst > threshold] = 255

    # 极小的高斯模糊 (sigma=1.0)，只做微弱扩散，
    # 绝不破坏 RA-PConv 需要的尖锐拓扑
    heatmap = cv2.GaussianBlur(keypoints_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return heatmap


def enhance_sar_image_v2(sar_img, sigma_keypoint=2, sigma_tensor=0.5):
    """
    终极版预处理接口：原图(主干) + 角点(先验) + 张量(先验)

    重构输入为"物理-语义黄金三通道"：
    - R通道（主干流）：原始SAR灰度图（保留100%弱散射细粒度边缘，提供绝对真实的语义基底）
    - G通道（先验流1）：锐化版强散射点（极大减小高斯模糊sigma，只提供十字形骨架的精确坐标）
    - B通道（先验流2）：结构张量图（保留线性边缘轮廓）

    此函数生成的BGR图像专供 DualStreamStem 使用：
    - R通道(索引2) → DualStreamStem.main_stem (独立主干流)
    - B/G通道(索引0,1) → DualStreamStem.prior_stem (物理先验流)

    Args:
        sar_img: 单通道SAR图像，shape为(H, W)或(H, W, 1)
        sigma_keypoint: Harris角点的高斯模糊半径
        sigma_tensor: 结构张量的高斯平滑尺度

    Returns:
        np.ndarray: 三通道BGR增强图像，shape为(H, W, 3)，dtype为uint8
                    OpenCV BGR顺序：[B=G通道(先验), G=角点(先验), R=原图(主干)]
    """
    # 确保单通道
    if len(sar_img.shape) == 3:
        if sar_img.shape[2] == 3:
            sar_img_gray = cv2.cvtColor(sar_img, cv2.COLOR_BGR2GRAY)
        else:
            sar_img_gray = sar_img[:, :, 0]
    else:
        sar_img_gray = sar_img

    # R 通道（主干流）：绝对纯净的原始 SAR 图像
    # —— 拯救细粒度信息的底线！不做任何预处理
    channel_r = sar_img_gray.astype(np.float32)

    # G 通道（先验流1）：锐化的十字/X形强散射点先验
    channel_g = generate_sharp_scattering_heatmap(sar_img_gray, sigma=sigma_keypoint)

    # B 通道（先验流2）：结构张量线性边缘先验
    channel_b = generate_structure_tensor_map(sar_img_gray, sigma_tensor=sigma_tensor)

    # 归一化到uint8
    R = normalize_to_uint8(channel_r)
    G = normalize_to_uint8(channel_g)
    B = normalize_to_uint8(channel_b)

    # BGR顺序：cv2.merge([B, G, R])
    # - x[:, :, 2] → R通道(原图) → DualStreamStem 主干流
    # - x[:, :, 0:2] → B/G通道(先验) → DualStreamStem 先验流
    return cv2.merge([B, G, R])
    
    # if len(img.shape) == 2:  # pyright: ignore[reportUnreachable]
    #     return True  # (H, W) - single channel
    
    # if len(img.shape) == 3 and img.shape[2] == 1:
    #     return True  # (H, W, 1) - single channel
    
    # # 新增：识别三通道灰度图 (R=G=B)
    # if len(img.shape) == 3 and img.shape[2] == 3:
    #     # 检查三个通道是否相等（灰度图）
    #     if img.dtype == np.uint8:
    #         is_gray = np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,1], img[:,:,2])
    #     else:
    #         is_gray = np.allclose(img[:,:,0], img[:,:,1], atol=1e-5) and np.allclose(img[:,:,1], img[:,:,2], atol=1e-5)
    #     if is_gray:
    #         return True
    
    # return False
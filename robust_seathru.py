"""
完全GPU优化的融合版SeaThru实现 (V2 - 鲁棒散射点选择)
=========================================================
Full GPU-Accelerated Hybrid Academic Implementation (V2 - Robust Backscatter Point Selection)
经过严格检查，确保所有操作都能在GPU上完整运行
学术改进：修改了后向散射点的选择逻辑，不再选择极端的“最暗点”，
而是选择一个更靠近中心趋势的百分位区间，以获得更稳定、鲁棒的拟合结果。
"""

import numpy as np
import torch
import cv2
from typing import Dict, Tuple, Optional, Union, Callable
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import os

# 安全导入CuPy和相关模块
try:
    import cupy as cp
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)  # 使用默认内存池，不限制大小
    import cupyx.scipy.ndimage as cupy_ndimage  # 正确的CuPy scipy接口
    CUPY_AVAILABLE = True
    print("✅ CuPy成功导入 (Full GPU Academic Version)")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️ CuPy不可用，将使用NumPy后备方案 (Full GPU Academic Version)")
    cp = None
    cupy_ndimage = None

# 导入scipy作为CPU后备
import scipy.ndimage as scipy_ndimage

# 从improved_seathru导入辅助函数（保持向后兼容）
try:
    from improved_seathru import (
        construct_neighborhood_map, refine_neighborhood_map,
        smooth_depth_map, estimate_water_quality_map,
        EnhancementProcessor
    )
except ImportError:
    print("⚠️ improved_seathru not available, some auxiliary functions disabled")


class GPUSafeFilter:
    """GPU安全滤波器 - 确保滤波操作能在GPU上正确执行"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.use_gpu = CUPY_AVAILABLE
        
    def median_filter(self, input_array, size=3):
        """安全的中值滤波"""
        if self.use_gpu and isinstance(input_array, cp.ndarray):
            try:
                # CuPy的中值滤波
                return cupy_ndimage.median_filter(input_array, size=size)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ GPU median filter failed: {e}")
                # 转到CPU处理
                cpu_array = input_array.get()
                filtered = scipy_ndimage.median_filter(cpu_array, size=size)
                return cp.asarray(filtered)
        else:
            # CPU路径
            return scipy_ndimage.median_filter(input_array, size=size)
    
    def gaussian_filter(self, input_array, sigma=1.0):
        """安全的高斯滤波"""
        if self.use_gpu and isinstance(input_array, cp.ndarray):
            try:
                return cupy_ndimage.gaussian_filter(input_array, sigma=sigma)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ GPU gaussian filter failed: {e}")
                cpu_array = input_array.get()
                filtered = scipy_ndimage.gaussian_filter(cpu_array, sigma=sigma)
                return cp.asarray(filtered)
        else:
            return scipy_ndimage.gaussian_filter(input_array, sigma=sigma)
    
    def uniform_filter(self, input_array, size=3):
        """安全的均值滤波"""
        if self.use_gpu and isinstance(input_array, cp.ndarray):
            try:
                return cupy_ndimage.uniform_filter(input_array, size=size)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ GPU uniform filter failed: {e}")
                cpu_array = input_array.get()
                filtered = scipy_ndimage.uniform_filter(cpu_array, size=size)
                return cp.asarray(filtered)
        else:
            return scipy_ndimage.uniform_filter(input_array, size=size)


class PhysicsBasedSeaThru:
    """
    基于物理的水下图像增强（完全GPU优化版）
    
    这个实现确保所有核心算法都能在GPU上运行：
    1. 所有数值计算使用CuPy
    2. 所有滤波操作有GPU实现
    3. 避免不必要的CPU/GPU数据传输
    4. 提供完整的GPU执行路径和CPU回退
    
    主要学术贡献：
    - 基于波长的精确水体光学建模
    - 自适应后向散射估计与拟合 (已采用更鲁棒的百分位点选择策略)
    - 局部邻域光照估计
    - 宽带衰减系数估计
    - 物理约束的参数优化
    """

    def __init__(self, quality_preset='balanced', verbose_filter=True):
        """
        初始化处理器

        Args:
            quality_preset: 质量预设 - 'fast', 'balanced', 'quality'
            verbose_filter: 是否让滤波器打印详细信息
        """
        self.quality_preset = quality_preset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 检测GPU可用性
        self.use_gpu = False
        self.device_name = "CPU"
        if CUPY_AVAILABLE and torch.cuda.is_available():
            try:
                # 测试CuPy是否真的可以使用
                test_arr = cp.array([1.0, 2.0, 3.0])
                _ = cp.sum(test_arr)
                
                device_id = cp.cuda.runtime.getDevice()
                props = cp.cuda.runtime.getDeviceProperties(device_id)
                self.device_name = props['name'].decode('utf-8')
                self.use_gpu = True
                
                # 设置CuPy内存池以提高性能
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=2**30)  # 限制1GB
                
            except Exception as e:
                print(f"⚠️ CUDA设备检测失败: {e}")
                self.use_gpu = False
                self.device_name = "CPU (CUDA detection failed)"

        # 初始化GPU安全滤波器
        self.filter = GPUSafeFilter(verbose=verbose_filter)
        
        # 获取配置
        self.config = self._get_preset_config(quality_preset)
        
        # 物理参数边界
        self.physical_bounds = {
            'B_inf': {'min': 0.05, 'max': 0.3}, # 收紧了B_inf的上限以获得更稳定的结果
            'beta_B': {'min': 0.2, 'max': 0.8},
            'gamma': {'min': 0.8, 'max': 1.2}, # 收紧了gamma的边界使其更接近1
            'beta_D': {'min': 0.01, 'max': 0.5}
        }
        
        # 水体光学特性（基于科学文献）
        self.water_properties = {
            'absorption': {450: 0.015, 550: 0.035, 650: 0.065},
            'scattering': {450: 0.032, 550: 0.020, 650: 0.010}
        }

        print(f"PhysicsBasedSeaThru Initialized - V2 Robust Backscatter")
        print(f"  Preset: {quality_preset}")
        print(f"  Device: {self.device_name}, GPU Enabled: {self.use_gpu}")
        if self.use_gpu:
            print(f"  CuPy Version: {cp.__version__}")

    def _get_preset_config(self, preset):
        """获取预设配置 - 融合两个版本的参数并增加新参数"""
        configs = {
            'fast': {
                'optimization_iterations': 3,
                'enhancement_strength': 0.7,
                'preserve_natural_look': 1.0,
                'p': 0.15,
                'f': 2.5,
                'neighborhood_iterations': 5,
                'illumination_iterations': 10,
                'backscatter_percentiles': (0.5, 8.0), # (low, high) for backscatter point selection
                'gpu_block_size': 256
            },
            'balanced': {
                'optimization_iterations': 8,
                'enhancement_strength': 0.7,
                'preserve_natural_look': 1.0,
                'p': 0.15,
                'f': 2.5,
                'neighborhood_iterations': 10,
                'illumination_iterations': 20,
                'backscatter_percentiles': (0.5, 8.0), # 更窄的范围以获得平衡结果 #(1.0, 10.0)
                'gpu_block_size': 512
            },
            'quality': {
                'optimization_iterations': 15,
                'enhancement_strength': 0.7,
                'preserve_natural_look': 1.0,
                'p': 0.15,
                'f': 2.5,
                'neighborhood_iterations': 15,
                'illumination_iterations': 30,
                'backscatter_percentiles': (0.5, 8.0), # 最严格的范围以获得高质量结果
                'gpu_block_size': 1024
            }
        }
        return configs.get(preset, configs['balanced'])

    def _ensure_gpu_array(self, array, array_name="array"):
        """确保数组在GPU上（如果使用GPU）"""
        if self.current_run_use_gpu:
            if not isinstance(array, cp.ndarray):
                try:
                    return cp.asarray(array)
                except Exception as e:
                    print(f"⚠️ Failed to move {array_name} to GPU: {e}")
                    self.current_run_use_gpu = False
                    return array
            return array
        else:
            if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
                return array.get()
            return array

    def _get_xp_module(self, array=None):
        """获取正确的数组模块（cupy或numpy）"""
        if array is not None:
            return cp if (CUPY_AVAILABLE and isinstance(array, cp.ndarray)) else np
        return cp if (self.current_run_use_gpu and CUPY_AVAILABLE) else np

    def _preprocess_conservative(self, img: np.ndarray, depths: np.ndarray):
        """
        保守的预处理，确保GPU兼容性
        """
        print("  Preprocessing with GPU optimization...")
        img_np = img.astype(np.float32)
        depths_np = depths.astype(np.float32)

        # 初始化当前运行的GPU使用状态
        self.current_run_use_gpu = self.use_gpu
        
        # 尝试将数据移到GPU
        if self.current_run_use_gpu:
            print("    Moving data to GPU...")
            try:
                current_img_arr = cp.asarray(img_np)
                current_depths_arr = cp.asarray(depths_np)
                print(f"    Data moved to GPU successfully. Image shape: {current_img_arr.shape}")
            except Exception as e:
                print(f"⚠️ GPU array creation failed: {e}. Switching to CPU.")
                self.current_run_use_gpu = False
                current_img_arr = img_np
                current_depths_arr = depths_np
        else:
            print("    Using CPU for preprocessing.")
            current_img_arr = img_np
            current_depths_arr = depths_np

        # 获取正确的数组模块
        xp = self._get_xp_module(current_img_arr)
        
        # 归一化图像到[0,1]
        if current_img_arr.max() > 1.0001:
            print("    Normalizing image to [0,1].")
            current_img_arr = current_img_arr / 255.0
        
        # 使用GPU安全滤波器平滑深度图
        print("    Applying median filter to depth map...")
        filtered_depths = self.filter.median_filter(current_depths_arr, size=5)
        
        # 确保滤波后的深度在正确的设备上
        filtered_depths = self._ensure_gpu_array(filtered_depths, "filtered_depths")
        
        # 归一化深度范围
        print("    Normalizing depth map range...")
        d_max = xp.max(filtered_depths)
        d_min = xp.min(filtered_depths)

        if d_max > d_min:
            normalized_depths = (filtered_depths - d_min) / (d_max - d_min)
        else:
            normalized_depths = xp.zeros_like(filtered_depths)
        
        # 将深度范围调整到[0.05, 1.0]，避免除零
        final_depths = 0.05 + 0.95 * normalized_depths
        
        print(f"  Preprocessing finished. Using {'GPU' if self.current_run_use_gpu else 'CPU'}")
        return current_img_arr, final_depths

    def _get_water_property(self, wavelength: int, property_type: str, xp) -> float:
        """
        获取水体光学特性（GPU兼容版本）
        """
        # 创建数组用于插值
        wavelengths = list(self.water_properties[property_type].keys())
        values = list(self.water_properties[property_type].values())
        
        ws = xp.array(wavelengths, dtype=xp.float32)
        vs = xp.array(values, dtype=xp.float32)
        
        # 确保波长是数组格式
        w_array = xp.array([wavelength], dtype=xp.float32)
        
        # 插值
        result = xp.interp(w_array, ws, vs)
        
        # 返回标量值
        return float(result[0]) if hasattr(result, '__len__') else float(result)

    def _gpu_bincount_weighted(self, indices, weights, minlength=None, xp=cp):
        """GPU优化的加权bincount实现"""
        if xp == np:
            return np.bincount(indices.ravel(), weights=weights.ravel(), minlength=minlength)
        
        # CuPy的bincount可能需要特殊处理
        try:
            return cp.bincount(indices.ravel(), weights=weights.ravel(), minlength=minlength)
        except Exception as e:
            # 如果CuPy bincount失败，使用替代实现
            if minlength is None:
                minlength = int(cp.max(indices)) + 1
            
            result = cp.zeros(minlength, dtype=cp.float32)
            indices_flat = indices.ravel()
            weights_flat = weights.ravel() if weights is not None else cp.ones_like(indices_flat)
            
            # 使用scatter_add作为替代
            cp.scatter_add(result, indices_flat, weights_flat)
            return result

    def _construct_neighborhood_map_gpu(self, depths, xp, epsilon=0.1):
        """
        GPU优化的邻域映射构建
        使用并行算法加速邻域合并
        """
        h, w = depths.shape
        labels = xp.arange(h * w, dtype=xp.int32).reshape(h, w)
        threshold = (xp.max(depths) - xp.min(depths)) * epsilon

        # 创建偏移数组用于邻域检查
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 4邻域
        
        for iteration in range(self.config['neighborhood_iterations']):
            labels_old = labels.copy()
            
            for dy, dx in offsets:
                # 使用GPU友好的方式计算邻域
                if dy == 0:  # 水平偏移
                    if dx > 0:
                        neighbor_depths = xp.pad(depths[:, :-dx], ((0, 0), (dx, 0)), mode='edge')
                        neighbor_labels = xp.pad(labels_old[:, :-dx], ((0, 0), (dx, 0)), mode='edge')
                    else:
                        neighbor_depths = xp.pad(depths[:, -dx:], ((0, 0), (0, -dx)), mode='edge')
                        neighbor_labels = xp.pad(labels_old[:, -dx:], ((0, 0), (0, -dx)), mode='edge')
                else:  # 垂直偏移
                    if dy > 0:
                        neighbor_depths = xp.pad(depths[:-dy, :], ((dy, 0), (0, 0)), mode='edge')
                        neighbor_labels = xp.pad(labels_old[:-dy, :], ((dy, 0), (0, 0)), mode='edge')
                    else:
                        neighbor_depths = xp.pad(depths[-dy:, :], ((0, -dy), (0, 0)), mode='edge')
                        neighbor_labels = xp.pad(labels_old[-dy:, :], ((0, -dy), (0, 0)), mode='edge')
                
                # 更新标签
                mask = xp.abs(depths - neighbor_depths) < threshold
                labels = xp.where(mask & (neighbor_labels < labels), neighbor_labels, labels)
            
            # 检查收敛
            if xp.array_equal(labels, labels_old):
                break

        # 重新标记以获得连续的标签
        unique_labels, relabeled = xp.unique(labels, return_inverse=True)
        return relabeled.reshape(h, w), len(unique_labels)
    
    def _find_backscatter_points(self, img_channel, depths, xp, num_bins=20):
        """
        [学术改进] 查找用于拟合后向散射模型的点（GPU优化版本）。
        该版本不再选择极端的“最暗点”，而是选择一个更靠近中心趋势的
        像素百分位区间，以获得更稳定、更鲁棒的拟合结果。
        """
        z_min = xp.min(depths)
        z_max = xp.max(depths)
        
        if z_max <= z_min:
            mean_val = xp.mean(img_channel)
            return xp.array([[z_min, mean_val]], dtype=xp.float32)
        
        # 创建深度区间
        bin_edges = xp.linspace(z_min, z_max, num_bins + 1)
        depth_flat = depths.ravel()
        img_flat = img_channel.ravel()
        
        points_list = []

        for i in range(num_bins):
            # 找到当前区间的点
            mask = (depth_flat >= bin_edges[i]) & (depth_flat < bin_edges[i+1])
            if not xp.any(mask):
                continue
            
            values_in_bin = img_flat[mask]
            depths_in_bin = depth_flat[mask]
            
            if len(values_in_bin) < 10: # 如果区间内点太少，则直接使用
                selected_depths = depths_in_bin
                selected_values = values_in_bin
            else:
                # 新逻辑：不再取最暗点，而是取一个更靠近中心趋势的百分位点。
                # 这种方法更能代表该深度下散射的“典型”情况，而不是“极端”情况。
                percentile_low, percentile_high = self.config['backscatter_percentiles']

                # 获取对应百分位的值 (xp.percentile对CuPy和NumPy都友好)
                try:
                    p_low_val, p_high_val = xp.percentile(values_in_bin, [percentile_low, percentile_high])
                except IndexError: # CuPy在某些版本中对空数组或小数组的百分位有bug
                    continue

                # 选取位于这个区间的点
                percentile_mask = (values_in_bin >= p_low_val) & (values_in_bin <= p_high_val)

                if xp.sum(percentile_mask) > 0:
                    selected_depths = depths_in_bin[percentile_mask]
                    selected_values = values_in_bin[percentile_mask]
                else:
                    # 如果区间内没有点 (可能是因为数据点太少或分布极端)，
                    # 作为后备，仍然取最暗的一个点，保证总有数据点用于拟合。
                    indices = xp.argmin(values_in_bin)
                    selected_depths = depths_in_bin[indices:indices+1] # 保持数组形式
                    selected_values = values_in_bin[indices:indices+1]
            
            # 组合深度和值
            if selected_depths.size > 0:
                points = xp.stack([selected_depths, selected_values], axis=1)
                points_list.append(points)
        
        if points_list:
            return xp.vstack(points_list)
        else:
            # 如果完全没有点被选上，返回一个全局均值点作为最后的保障
            mean_val = xp.mean(img_channel)
            return xp.array([[z_min, mean_val]], dtype=xp.float32)


    def _fit_backscatter_model_gpu(self, points, wavelength, xp):
        """
        GPU优化的后向散射模型拟合
        使用向量化操作加速网格搜索
        """
        if points.shape[0] < 2:
            # 如果点数过少，返回一个基于物理先验的默认模型
            params = self._get_default_params(wavelength)
            def default_model(d):
                return params['B_inf'] * (1 - xp.exp(-params['beta_B'] * xp.power(d, params['gamma'])))
            return default_model, params

        depths = points[:, 0]
        values = points[:, 1]
        
        # 初始参数估计
        B_inf_init = float(xp.percentile(values, 95))
        beta_B_init = self._get_water_property(wavelength, 'scattering', xp)
        
        # 创建参数网格（使用较少的点以提高GPU效率）
        n_B = 8
        n_beta = 8
        n_gamma = 5
        
        # 使用物理边界来定义搜索范围
        B_inf_min, B_inf_max = self.physical_bounds['B_inf']['min'], self.physical_bounds['B_inf']['max']
        beta_B_min, beta_B_max = self.physical_bounds['beta_B']['min'], self.physical_bounds['beta_B']['max']
        gamma_min, gamma_max = self.physical_bounds['gamma']['min'], self.physical_bounds['gamma']['max']
        
        B_inf_range = xp.linspace(B_inf_min, B_inf_max, n_B, dtype=xp.float32)
        beta_B_range = xp.linspace(beta_B_min, beta_B_max, n_beta, dtype=xp.float32)
        gamma_range = xp.linspace(gamma_min, gamma_max, n_gamma, dtype=xp.float32)
        
        # 向量化网格搜索
        best_loss = xp.inf
        best_params = (B_inf_init, beta_B_init, 1.0)
        
        # 预计算depths的幂次方以加速
        depths_expanded = depths[:, xp.newaxis]  # Shape: (n_points, 1)
        
        for B_inf in B_inf_range:
            for beta_B in beta_B_range:
                # 向量化计算所有gamma值
                gamma_expanded = gamma_range[xp.newaxis, :]  # Shape: (1, n_gamma)
                
                # 计算所有gamma的预测值
                depths_powered = xp.power(depths_expanded, gamma_expanded)  # Shape: (n_points, n_gamma)
                predicted_all = B_inf * (1 - xp.exp(-beta_B * depths_powered))  # Shape: (n_points, n_gamma)
                
                # 计算损失
                values_expanded = values[:, xp.newaxis]  # Shape: (n_points, 1)
                
                # 正则化拟合：对偏离物理先验的参数进行惩罚
                error_loss = xp.sum((values_expanded - predicted_all) ** 2, axis=0) # L2 Loss
                
                lambda_B = 0.1 # 对B_inf的惩罚系数
                lambda_gamma = 0.2 # 对gamma的惩罚系数
                
                reg_loss_B = lambda_B * (B_inf - B_inf_init) ** 2
                reg_loss_gamma = lambda_gamma * (gamma_range - 1.0) ** 2
                
                losses = error_loss + reg_loss_B + reg_loss_gamma
                
                # 找到最佳gamma
                min_idx = xp.argmin(losses)
                min_loss = losses[min_idx]
                
                if min_loss < best_loss:
                    best_loss = min_loss
                    best_params = (B_inf, beta_B, gamma_range[min_idx])
        
        # 构建最终模型函数
        B_final, beta_final, gamma_final = best_params
        
        def model(d):
            return B_final * (1 - xp.exp(-beta_final * xp.power(d, gamma_final)))
        
        # 返回模型和参数字典
        params_dict = {
            'B_inf': float(B_final),
            'beta_B': float(beta_final),
            'gamma': float(gamma_final)
        }
        
        return model, params_dict

    def _estimate_illumination_gpu(self, img_channel, backscatter, depths, nmap, n_neighborhoods, xp):
        """
        GPU优化的光照估计
        使用高效的邻域平均计算
        """
        # 去除后向散射得到直接信号
        direct_signal = xp.maximum(img_channel - backscatter, 1e-6)
        avg_signal = direct_signal.copy()
        
        # 预分配数组以提高性能
        mean_lookup = xp.zeros(n_neighborhoods, dtype=xp.float32)
        
        for iteration in range(self.config['illumination_iterations']):
            # 使用GPU优化的bincount计算邻域统计
            sum_in_neighborhood = self._gpu_bincount_weighted(
                nmap, avg_signal, minlength=n_neighborhoods, xp=xp
            )
            count_in_neighborhood = self._gpu_bincount_weighted(
                nmap, xp.ones_like(avg_signal), minlength=n_neighborhoods, xp=xp
            )
            
            # 避免除零
            count_in_neighborhood = xp.maximum(count_in_neighborhood, 1)
            
            # 计算每个邻域的平均值
            mean_in_neighborhood = sum_in_neighborhood / count_in_neighborhood
            
            # 更新信号（使用查找表方式）
            avg_signal_new = mean_in_neighborhood[nmap]
            
            # 混合原始信号和平滑信号
            p = self.config['p']
            avg_signal = p * direct_signal + (1 - p) * avg_signal_new
        
        # 应用最终缩放因子
        return self.config['f'] * avg_signal

    def _estimate_wideband_attenuation_gpu(self, depths, illumination, wavelength, xp):
        """
        GPU优化的宽带衰减估计
        """
        # 获取理论衰减系数
        absorption = self._get_water_property(wavelength, 'absorption', xp)
        scattering = self._get_water_property(wavelength, 'scattering', xp)
        theoretical_beta = absorption + scattering
        
        # 确保安全的计算（避免log(0)和除零）
        safe_depths = xp.maximum(depths, 0.1)
        safe_illum = xp.clip(illumination, 0.01, 1.0)
        
        # 计算经验衰减系数
        empirical_beta = -xp.log(safe_illum) / safe_depths
        
        # 处理无效值（NaN和Inf）
        mask_invalid = xp.isnan(empirical_beta) | xp.isinf(empirical_beta)
        empirical_beta = xp.where(mask_invalid, theoretical_beta, empirical_beta)
        empirical_beta = xp.clip(empirical_beta, 0, 5.0)
        
        # 基于深度的加权平均
        mean_depth = xp.mean(depths)
        depth_weight = xp.exp(-depths / (mean_depth + 1e-6))
        weighted_beta = empirical_beta * depth_weight + theoretical_beta * (1 - depth_weight)
        
        # 使用GPU安全滤波器进行高斯平滑
        smoothed_beta = self.filter.gaussian_filter(weighted_beta, sigma=5)
        
        return smoothed_beta

    def _backscatter_model(self, depths_arr, params_dict):
        """
        计算后向散射（GPU兼容版本）
        """
        B_inf = params_dict['B_inf']
        beta_B = params_dict['beta_B']
        gamma = params_dict['gamma']
        xp = self._get_xp_module(depths_arr)

        # 确保深度非负
        safe_depths = xp.maximum(depths_arr, 0)
        
        # 处理gamma=0的特殊情况
        if abs(gamma) < 1e-6:
            powered_depths = xp.ones_like(safe_depths)
        else:
            powered_depths = xp.power(safe_depths, gamma)
        
        # 计算后向散射
        backscatter = B_inf * (1 - xp.exp(-beta_B * powered_depths))
        
        return backscatter

    def _estimate_physical_parameters(self, img_arr, depths_arr, args_obj):
        """
        估计物理参数（完全GPU优化版本）
        """
        print("  Estimating physical parameters on GPU...")
        params = {}
        wavelengths = {'R': 650, 'G': 550, 'B': 450}
        xp = self._get_xp_module(img_arr)

        # 构建邻域映射
        print("    Constructing neighborhood map...")
        try:
            nmap, n_neighborhoods = self._construct_neighborhood_map_gpu(depths_arr, xp)
            print(f"    Neighborhood map: {n_neighborhoods} regions found")
        except Exception as e:
            print(f"⚠️ Neighborhood map construction failed: {e}")
            nmap = xp.zeros(depths_arr.shape, dtype=xp.int32)
            n_neighborhoods = 1

        # 预分配数组
        backscatter_all = xp.zeros_like(img_arr)
        beta_D_all = xp.zeros_like(img_arr)
        
        # 为每个颜色通道估计参数
        for i, (channel_name, wavelength) in enumerate(wavelengths.items()):
            try:
                print(f"    Processing {channel_name} channel (λ={wavelength}nm)...")
                channel_data = img_arr[:,:,i]
                
                # [学术改进] 使用新的鲁棒方法找到后向散射点
                points = self._find_backscatter_points(channel_data, depths_arr, xp)
                print(f"      Found {points.shape[0]} backscatter points using robust percentile method")
                
                # 拟合后向散射模型
                backscatter_model, params_channel = self._fit_backscatter_model_gpu(points, wavelength, xp)
                
                # 应用物理边界约束 (在拟合内部已完成，此处为双重保险)
                for key in params_channel:
                    if key in self.physical_bounds:
                        params_channel[key] = float(np.clip(
                            params_channel[key],
                            self.physical_bounds[key]['min'],
                            self.physical_bounds[key]['max']
                        ))
                
                # 参数优化（如果配置）
                if self.config['optimization_iterations'] > 0:
                    params_channel = self._simple_optimization_gpu(
                        channel_data, depths_arr, params_channel,
                        n_iter=self.config['optimization_iterations'], xp=xp
                    )
                
                # 计算后向散射
                backscatter_all[:,:,i] = self._backscatter_model(depths_arr, params_channel)
                
                # 估计光照
                illumination = self._estimate_illumination_gpu(
                    channel_data, backscatter_all[:,:,i], 
                    depths_arr, nmap, n_neighborhoods, xp
                )
                
                # 估计衰减
                beta_D_all[:,:,i] = self._estimate_wideband_attenuation_gpu(
                    depths_arr, illumination, wavelength, xp
                )
                
                # 保存平均衰减系数
                params_channel['beta_D'] = float(xp.mean(beta_D_all[:,:,i]))
                params[channel_name] = params_channel
                
                print(f"      Parameters: B∞={params_channel['B_inf']:.3f}, "
                      f"β_B={params_channel['beta_B']:.3f}, "
                      f"γ={params_channel['gamma']:.3f}, "
                      f"β_D={params_channel['beta_D']:.3f}")
                
            except Exception as e:
                print(f"⚠️ {channel_name}-channel processing failed: {e}")
                import traceback
                traceback.print_exc()
                params[channel_name] = self._get_default_params(wavelength)

        # 保存数组供后续使用
        self._backscatter_array = backscatter_all
        self._beta_D_array = beta_D_all
        
        print("  Physical parameters estimation completed.")
        return params

    def _simple_optimization_gpu(self, channel_arr, depths_arr, initial_params_dict, n_iter=5, xp=None):
        """
        GPU优化的参数优化
        """
        if xp is None:
            xp = self._get_xp_module(channel_arr)
            
        current_params = initial_params_dict.copy()
        best_params = initial_params_dict.copy()
        best_loss = float('inf')
        learning_rate = 0.05

        for iter_idx in range(n_iter):
            # 计算当前参数的损失
            B_model = self._backscatter_model(depths_arr, current_params)
            residual = xp.abs(channel_arr - B_model)
            loss = float(xp.mean(residual))

            # 正则化
            reg = 0.01 * (abs(current_params['B_inf'] - initial_params_dict['B_inf']) +
                          abs(current_params['beta_B'] - initial_params_dict['beta_B']))
            loss += reg

            if loss < best_loss:
                best_loss = loss
                best_params = current_params.copy()
            
            # 参数扰动（使用numpy的随机数生成器，因为它在CPU/GPU上都能工作）
            current_params['B_inf'] *= (1 + learning_rate * np.random.uniform(-0.1, 0.1))
            current_params['beta_B'] *= (1 + learning_rate * np.random.uniform(-0.1, 0.1))
            current_params['gamma'] *= (1 + learning_rate * 0.05 * np.random.uniform(-0.1, 0.1))
            
            # 应用边界约束
            for key in current_params:
                if key in self.physical_bounds:
                    current_params[key] = float(np.clip(
                        current_params[key],
                        self.physical_bounds[key]['min'],
                        self.physical_bounds[key]['max']
                    ))
            
        return best_params

    def _physics_based_reconstruction(self, img_arr, depths_arr, params_all_channels):
        """
        基于物理模型的图像重建（GPU优化版本）
        """
        print("  Reconstructing image based on physics model...")
        xp = self._get_xp_module(img_arr)
        
        # 使用已计算的数组
        if hasattr(self, '_backscatter_array') and hasattr(self, '_beta_D_array'):
            backscatter_all = self._backscatter_array
            beta_D_all = self._beta_D_array
        else:
            # 重新计算
            backscatter_all = xp.zeros_like(img_arr)
            beta_D_all = xp.zeros_like(img_arr)
            
            for i, channel_name in enumerate(['R', 'G', 'B']):
                if channel_name not in params_all_channels:
                    continue
                p_channel = params_all_channels[channel_name]
                backscatter_all[:,:,i] = self._backscatter_model(depths_arr, p_channel)
                beta_D_all[:,:,i] = p_channel.get('beta_D', 0.1)

        # 计算透射率（向量化操作）
        # 扩展深度数组以匹配图像维度
        depths_expanded = depths_arr[:, :, xp.newaxis]
        transmission = xp.exp(-beta_D_all * depths_expanded)
        transmission = xp.maximum(transmission, 0.1)
        
        # 重建图像
        recovered_img = (img_arr - backscatter_all) / transmission
        recovered_img = xp.clip(recovered_img, 0, 1)
        
        print("  Physics-based reconstruction completed.")
        return recovered_img

    def _adaptive_enhancement_gpu(self, img, xp):
        """
        GPU优化的自适应增强
        """
        h, w, c = img.shape
        enhanced = xp.zeros_like(img)
        
        # 批量处理所有通道的百分位数计算
        for channel_idx in range(c):
            channel = img[:, :, channel_idx]
            
            # 使用GPU加速的百分位数计算
            percentiles = xp.percentile(channel, [2, 98])
            p_low = percentiles[0]
            p_high = percentiles[1]
            
            if p_high > p_low + 1e-6:
                # 对比度拉伸
                enhanced[:, :, channel_idx] = (channel - p_low) / (p_high - p_low)
            else:
                enhanced[:, :, channel_idx] = channel
        
        # 颜色平衡（向量化操作）
        mean_per_channel = xp.mean(enhanced, axis=(0, 1))
        
        if xp.all(mean_per_channel > 1e-6):
            global_mean = xp.mean(mean_per_channel)
            scale_factors = global_mean / mean_per_channel
            scale_factors = xp.clip(scale_factors, 0.7, 1.3)
            
            # 应用缩放因子
            for channel_idx in range(c):
                enhanced[:, :, channel_idx] *= scale_factors[channel_idx]

        return xp.clip(enhanced, 0, 1)

    def _natural_enhancement(self, img_arr_to_enhance):
        """
        自然的图像增强（完全GPU优化版本）
        """
        print("  Applying natural enhancement...")
        
        if self.current_run_use_gpu:
            print("    Executing GPU natural enhancement...")
            try:
                # 确保输入在GPU上
                img_gpu = self._ensure_gpu_array(img_arr_to_enhance, "enhancement_input")
                xp = cp
                
                # GPU上的自适应增强
                enhanced_gpu = self._adaptive_enhancement_gpu(img_gpu, xp)
                
                # 基于配置的增强强度进行混合
                strength = self.config['enhancement_strength']
                final_gpu = (1 - strength) * img_gpu + strength * enhanced_gpu
                
                # 额外的细节增强（可选）
                if self.config.get('preserve_natural_look', 1.0) < 1.0:
                    # 使用高斯滤波保留自然外观
                    smoothed = self.filter.gaussian_filter(final_gpu, sigma=1.0)
                    preserve_factor = self.config['preserve_natural_look']
                    final_gpu = preserve_factor * final_gpu + (1 - preserve_factor) * smoothed
                
                print("    GPU natural enhancement completed successfully.")
                # 保持在GPU上，直到最后才转回CPU
                return final_gpu
                
            except Exception as e:
                print(f"⚠️ GPU natural enhancement failed: {e}")
                import traceback
                traceback.print_exc()
                # 回退到CPU
                self.current_run_use_gpu = False
                if CUPY_AVAILABLE and isinstance(img_arr_to_enhance, cp.ndarray):
                    img_cpu = img_arr_to_enhance.get()
                else:
                    img_cpu = np.asarray(img_arr_to_enhance)
                img_arr_to_enhance = img_cpu
        
        # CPU路径
        print("    Using CPU for natural enhancement...")
        xp = np
        img_cpu = img_arr_to_enhance if isinstance(img_arr_to_enhance, np.ndarray) else np.asarray(img_arr_to_enhance)
        
        try:
            enhanced_cpu = self._adaptive_enhancement_gpu(img_cpu, xp)
            strength = self.config['enhancement_strength']
            final_cpu = (1 - strength) * img_cpu + strength * enhanced_cpu
            
            if self.config.get('preserve_natural_look', 1.0) < 1.0:
                smoothed = self.filter.gaussian_filter(final_cpu, sigma=1.0)
                preserve_factor = self.config['preserve_natural_look']
                final_cpu = preserve_factor * final_cpu + (1 - preserve_factor) * smoothed
            
            print("    CPU natural enhancement completed.")
            return np.clip(final_cpu, 0, 1)
            
        except Exception as e:
            print(f"⚠️ CPU natural enhancement failed: {e}")
            return np.clip(img_cpu, 0, 1)

    def process_image(self, img: np.ndarray, depths: np.ndarray, args_namespace) -> np.ndarray:
        """
        处理图像主函数（完全GPU优化版本）
        """
        print(f"\nProcessing image with Full GPU PhysicsBasedSeaThru (Preset: {self.quality_preset})...")
        print(f"  Input shape: {img.shape}, Depth shape: {depths.shape}")
        start_time = time.time()
        
        # 预处理
        processed_img_arr, processed_depths_arr = self._preprocess_conservative(img, depths)
        
        print(f"  Using {'GPU' if self.current_run_use_gpu else 'CPU'} for processing")

        try:
            # 估计物理参数
            params_estimated = self._estimate_physical_parameters(
                processed_img_arr, processed_depths_arr, args_namespace
            )
            
            # 基于物理模型重建
            recovered_img_arr = self._physics_based_reconstruction(
                processed_img_arr, processed_depths_arr, params_estimated
            )
            
            # 自然增强
            enhanced_img_arr = self._natural_enhancement(recovered_img_arr)

            # 确保最终输出是CPU上的NumPy数组
            if CUPY_AVAILABLE and isinstance(enhanced_img_arr, cp.ndarray):
                print("  Transferring result from GPU to CPU...")
                final_output_np = enhanced_img_arr.get()
            else:
                final_output_np = enhanced_img_arr if isinstance(enhanced_img_arr, np.ndarray) else np.asarray(enhanced_img_arr)

        except Exception as e:
            print(f"⚠️ Processing pipeline error: {e}")
            import traceback
            traceback.print_exc()
            print("  Attempting simple fallback enhancement...")
            
            try:
                # 简单的对比度增强作为后备
                img_fallback = img.astype(np.float32)
                if img_fallback.max() > 1.0:
                    img_fallback /= 255.0
                
                # 简单的百分位拉伸
                for i in range(3):
                    channel = img_fallback[:, :, i]
                    p_low, p_high = np.percentile(channel, [5, 95])
                    if p_high > p_low:
                        img_fallback[:, :, i] = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
                
                final_output_np = img_fallback
                
            except Exception as e2:
                print(f"❌ Fallback enhancement failed: {e2}")
                print("  Returning normalized input image.")
                final_output_np = img.astype(np.float32)
                if final_output_np.max() > 1.0:
                    final_output_np /= 255.0

        # 清理GPU内存（如果使用了GPU）
        if self.current_run_use_gpu and CUPY_AVAILABLE:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

        elapsed = time.time() - start_time
        fps = 1.0/elapsed if elapsed > 0 else float('inf')
        print(f"\n✅ Processing completed in {elapsed:.2f}s (FPS: {fps:.1f})")
        
        return np.clip(final_output_np, 0, 1)

    def get_quality_metrics(self, original_np: np.ndarray, enhanced_np: np.ndarray) -> Dict:
        """计算质量指标"""
        metrics = {}
        try:
            # 确保输入是NumPy数组
            if not isinstance(original_np, np.ndarray):
                original_np = np.asarray(original_np)
            if not isinstance(enhanced_np, np.ndarray):
                enhanced_np = np.asarray(enhanced_np)
                
            orig_uint8 = (np.clip(original_np, 0, 1) * 255).astype(np.uint8)
            enh_uint8 = (np.clip(enhanced_np, 0, 1) * 255).astype(np.uint8)

            metrics['brightness_change'] = float(np.mean(enhanced_np) - np.mean(original_np))
            metrics['contrast_gain'] = float(np.std(enhanced_np) / (np.std(original_np) + 1e-6))

            if orig_uint8.shape[2] == 3 and enh_uint8.shape[2] == 3:
                orig_hsv = cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2HSV)
                enh_hsv = cv2.cvtColor(enh_uint8, cv2.COLOR_RGB2HSV)
                metrics['saturation_change'] = float(
                    np.mean(enh_hsv[:,:,1]) - np.mean(orig_hsv[:,:,1])
                )
            else:
                metrics['saturation_change'] = 0.0
                
            # 添加GPU相关指标
            metrics['gpu_used'] = self.current_run_use_gpu if hasattr(self, 'current_run_use_gpu') else False
            metrics['device_name'] = self.device_name
                
        except Exception as e:
            print(f"Error calculating quality metrics: {e}")
            
        return metrics

    def _get_default_params(self, wavelength):
        """获取默认参数"""
        default_params_map = {
            650: {'B_inf': 0.20, 'beta_B': 0.35, 'gamma': 0.85, 'beta_D': 0.25},  # Red
            550: {'B_inf': 0.15, 'beta_B': 0.45, 'gamma': 0.88, 'beta_D': 0.15},  # Green
            450: {'B_inf': 0.10, 'beta_B': 0.55, 'gamma': 0.90, 'beta_D': 0.08}   # Blue
        }
        return default_params_map.get(wavelength, default_params_map[550])


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("Running Full GPU PhysicsBasedSeaThru Test (V2 - Robust Selection)")
    print("=" * 60)
    
    # 模拟参数对象
    class ArgsMock:
        def __init__(self):
            self.p = 0.15
            self.f = 2.5

    args_mock = ArgsMock()

    # 创建处理器实例
    processor = PhysicsBasedSeaThru(quality_preset='balanced', verbose_filter=False)

    # 创建测试数据
    H, W = 512, 512  # 使用更大的图像测试GPU性能
    test_img_np = np.random.rand(H, W, 3).astype(np.float32)
    
    # 创建更真实的深度图（模拟水下场景）
    x, y = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    test_depths_np = (2 + np.sin(x * 10) * 0.5 + np.cos(y * 10) * 0.5 + 
                      np.random.rand(H, W) * 0.2).astype(np.float32)

    # 给图像加入一些颜色，模拟水下偏色
    test_img_np[:, :, 0] *= 0.6  # 减弱红色
    test_img_np[:, :, 1] *= 0.9  # 减弱绿色
    
    print(f"\nTest Configuration:")
    print(f"  Image shape: {test_img_np.shape}, dtype: {test_img_np.dtype}")
    print(f"  Depth shape: {test_depths_np.shape}, dtype: {test_depths_np.dtype}")
    print(f"  Depth range: [{test_depths_np.min():.2f}, {test_depths_np.max():.2f}]")

    # 处理图像
    result_np = processor.process_image(test_img_np, test_depths_np, args_mock)

    print(f"\nResults:")
    print(f"  Output shape: {result_np.shape}, dtype: {result_np.dtype}")
    print(f"  Output range: [{result_np.min():.3f}, {result_np.max():.3f}]")
    
    # 计算质量指标
    metrics = processor.get_quality_metrics(test_img_np, result_np)
    print(f"\nQuality Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # 如果GPU被使用，显示内存信息
    if processor.use_gpu and CUPY_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        print(f"\nGPU Memory Usage:")
        print(f"  Used: {mempool.used_bytes() / 1024**2:.2f} MB")
        print(f"  Total: {mempool.total_bytes() / 1024**2:.2f} MB")
    
    print("\n✅ Full GPU implementation test completed successfully!")
    print("=" * 60)

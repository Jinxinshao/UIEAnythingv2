from __future__ import absolute_import, division, print_function

# 🚨 关键：必须在所有CUDA相关导入之前执行
import cuda_env_setup  # 这会自动配置环境

import datetime
import cv2
import math
import natsort
import os
import sys
import glob
import argparse
import time
import io
import multiprocessing
import warnings
import logging # 添加 logging 导入
from tqdm import tqdm # 添加 tqdm 导入

warnings.filterwarnings('ignore')

import numpy as np
import PIL.Image as pil # Correct import for Pillow's Image module
import matplotlib as mpl
import matplotlib.cm as cm
mpl.use('Agg')  # 使用非交互式后端

# PyTorch导入
import torch
from torchvision import transforms # 从 torchvision 导入 transforms

# 现在可以安全地导入CuPy相关模块
try:
    import cupy as cp
    print("✅ CuPy成功导入")
    # 验证CuPy版本
    print(f"   CuPy版本: {cp.__version__}")
    # 获取CUDA运行时版本
    if hasattr(cp, 'cuda') and hasattr(cp.cuda, 'runtime') and hasattr(cp.cuda.runtime, 'runtimeGetVersion'):
        cuda_runtime_version = cp.cuda.runtime.runtimeGetVersion()
        print(f"   CUDA运行时版本 (来自CuPy): {cuda_runtime_version//1000}.{(cuda_runtime_version%1000)//10}")
    else:
        print("   无法获取CuPy的CUDA运行时版本。")
except ImportError as e:
    print(f"❌ CuPy导入失败: {e}")
    print("   将使用CPU模式运行")
    cp = None # 确保 cp 在导入失败时为 None
except Exception as e:
    print(f"⚠️ CuPy导入警告: {e}")
    cp = None

# 其他项目特定导入
from improved_seathru import run_pipeline # 假设这是主要的CPU处理流程
from balanced_gpu_seathru0619 import PhysicsBasedSeaThru
# from fixed_integrated_gpu_seathru import UltraFastSeaThru # 如果需要，取消注释
from matplotlib import pyplot as plt
# from UCIQE import getUCIQE # UCIQE is used in calculate_quality_score, ensure it's available if that func is used
# from UIQM import calculate_uiqm_from_pil # 如果使用，取消注释

# 深度估计模型
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("⚠️ Depth Anything V2 未找到，深度估计可能受影响。")
    DepthAnythingV2 = None

# 图像处理
from skimage import exposure, color, filters
from skimage.restoration import denoise_tv_chambolle, estimate_sigma
from skimage.filters import gaussian
from scipy import ndimage as scipy_ndimage # 明确使用 scipy.ndimage

# 环境变量设置 (通常在 cuda_env_setup.py 中处理)

# 针对RTX 4090的内存优化 (通常在 cuda_env_setup.py 中处理)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 4090" in gpu_name:
        print("🎯 RTX 4090检测到，启用专用优化 (在main.py中检查)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ: # 避免覆盖 cuda_env_setup.py 的设置
             os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


def scale(img_np):
    img_float_np = img_np.astype(np.float32)
    min_val = np.min(img_float_np)
    max_val = np.max(img_float_np)
    if max_val == min_val:
        if 0.0 <= min_val <= 1.0 and min_val == max_val : 
            return img_float_np 
        else: 
            return np.full_like(img_float_np, 0.5, dtype=np.float32)
    return (img_float_np - min_val) / (max_val - min_val)


def post_process_enhanced(img_np_0_1):
    """
    Enhanced post-processing function.
    Applies a series of filters and adjustments if the initial quality is low,
    or a milder version if the quality is already good.
    """
    if img_np_0_1 is None:
        logging.warning("post_process_enhanced received None image. Returning None.")
        return None
        
    original_dtype = img_np_0_1.dtype
    img_float32 = img_np_0_1.astype(np.float32) 

    def adaptive_tv_denoise(image, weight_factor=0.05):
        sigma_est = estimate_sigma(image, channel_axis=-1 if image.ndim == 3 else None, average_sigmas=True)
        if sigma_est is None or sigma_est < 1e-3: return image
        weight = sigma_est * weight_factor
        image_contiguous = np.ascontiguousarray(image)
        return denoise_tv_chambolle(image_contiguous, weight=weight, channel_axis=-1 if image.ndim == 3 else None, eps=2e-4, max_num_iter=100)

    def enhance_local_contrast(img_lc, clip_limit=0.005, kernel_size=12): 
        img_uint8 = (np.clip(img_lc, 0, 1) * 255).astype(np.uint8)
        if len(img_uint8.shape) == 2: 
            lab_like = cv2.cvtColor(cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB), cv2.COLOR_RGB2LAB)
        elif img_uint8.ndim == 3 and img_uint8.shape[2] == 1: 
            img_rgb_temp = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
            lab_like = cv2.cvtColor(img_rgb_temp, cv2.COLOR_RGB2LAB)
        else: 
            lab_like = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)

        l_channel = lab_like[:,:,0]
        clahe = cv2.createCLAHE(clipLimit=clip_limit*100, tileGridSize=(kernel_size,kernel_size))
        l_channel_clahe = clahe.apply(l_channel)
        lab_like[:,:,0] = l_channel_clahe

        if len(img_uint8.shape) == 2 or (img_uint8.ndim == 3 and img_uint8.shape[2] == 1) :
            result_rgb = cv2.cvtColor(lab_like, cv2.COLOR_LAB2RGB)
            result_gray = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY)
            return result_gray.astype(np.float32) / 255.0
        else:
            result_rgb = cv2.cvtColor(lab_like, cv2.COLOR_LAB2RGB)
            return result_rgb.astype(np.float32) / 255.0

    def color_balance_hsv(img_cb, saturation_factor=1.05): 
        img_uint8_cb = (np.clip(img_cb, 0, 1) * 255).astype(np.uint8)
        if len(img_uint8_cb.shape) == 2 or (img_uint8_cb.ndim == 3 and img_uint8_cb.shape[2] == 1): return img_cb 
        hsv = cv2.cvtColor(img_uint8_cb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation_factor, 0, 255)
        result_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return result_rgb.astype(np.float32) / 255.0

    def sharpen_details_unsharp_mask(img_sd, sigma=0.7, strength=0.3): 
        channel_axis_val = -1 if img_sd.ndim == 3 and img_sd.shape[2] > 1 else None
        blurred = gaussian(img_sd, sigma=sigma, channel_axis=channel_axis_val, preserve_range=True)
        detail = img_sd - blurred
        sharpened = np.clip(img_sd + strength * detail, 0, 1)
        return sharpened

    try:
        current_quality = calculate_quality_score(img_float32)
        
        processed_image = img_float32
        if current_quality < 0.7: 
            processed_image = adaptive_tv_denoise(processed_image)
            processed_image = enhance_local_contrast(processed_image)
            processed_image = color_balance_hsv(processed_image)
            processed_image = sharpen_details_unsharp_mask(processed_image)
        else: 
            processed_image = adaptive_tv_denoise(processed_image, weight_factor=0.02)
            processed_image = color_balance_hsv(processed_image, saturation_factor=1.02)

        final_result = np.clip(processed_image, 0, 1)
        processed_quality = calculate_quality_score(final_result)

        if processed_quality > current_quality or current_quality < 0.5 : 
            logging.info(f"Post-processing quality: {current_quality:.4f} -> {processed_quality:.4f}")
            return final_result.astype(original_dtype)
        else:
            logging.info(f"Post-processing did not improve quality significantly ({current_quality:.4f} -> {processed_quality:.4f}). Returning original (before this enhancement stage).")
            return img_float32.astype(original_dtype) # Return the image as it was before this specific post_process_enhanced

    except Exception as e:
        logging.error(f"Post-processing enhanced failed: {str(e)}. Returning original image.", exc_info=True)
        return img_np_0_1 


def calculate_quality_score(img_np_0_1):
    """
    Calculates a composite quality score for an image.
    Handles grayscale and color images.
    """
    try:
        if img_np_0_1 is None or img_np_0_1.size == 0:
            logging.warning("Cannot calculate quality score for None or empty image.")
            return 0.0
            
        img_uint8 = (np.clip(img_np_0_1, 0, 1) * 255).astype(np.uint8)
        
        if len(img_uint8.shape) == 2 or img_uint8.shape[0] == 0 or img_uint8.shape[1] == 0 : 
             l_channel = img_uint8 if len(img_uint8.shape) == 2 and img_uint8.size > 0 else np.array([[128]], dtype=np.uint8) 
             if l_channel.size == 0: l_channel = np.array([[128]], dtype=np.uint8) 
             a_channel = np.full_like(l_channel, 128, dtype=np.uint8) 
             b_channel = np.full_like(l_channel, 128, dtype=np.uint8)
        elif img_uint8.ndim == 3 and img_uint8.shape[2] == 1: 
            l_channel = img_uint8[:,:,0]
            a_channel = np.full_like(l_channel, 128, dtype=np.uint8)
            b_channel = np.full_like(l_channel, 128, dtype=np.uint8)
        else: 
            try:
                lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
            except cv2.error as e: 
                logging.warning(f"cv2.cvtColor to LAB failed: {e}. Using grayscale for L and neutral A,B.")
                if img_uint8.ndim == 3:
                    l_channel = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
                else: 
                    l_channel = img_uint8
                a_channel = np.full_like(l_channel, 128, dtype=np.uint8) 
                b_channel = np.full_like(l_channel, 128, dtype=np.uint8)

        if l_channel.size == 0: return 0.0 

        contrast = np.std(l_channel.astype(np.float32))
        saturation = np.mean(np.sqrt( (a_channel.astype(np.float32)-128)**2 + (b_channel.astype(np.float32)-128)**2 ))
        sharpness = np.mean(np.abs(scipy_ndimage.laplace(l_channel.astype(np.float32))))
        
        hist, _ = np.histogram(l_channel.flatten(), 256, [0,256])
        if np.sum(hist) < 1e-6: naturalness = 0.0 
        else:
            hist_smooth = scipy_ndimage.gaussian_filter1d(hist.astype(np.float32), sigma=1)
            naturalness = np.sum(np.minimum(hist, hist_smooth)) / np.sum(hist) 

        weights = {'contrast': 0.3, 'saturation': 0.25, 'sharpness': 0.25, 'naturalness': 0.2}
        
        contrast_norm = np.clip(contrast / 70.0, 0, 1) if contrast is not None else 0        
        saturation_norm = np.clip(saturation / 50.0, 0, 1) if saturation is not None else 0   
        sharpness_norm = np.clip(sharpness / 10.0, 0, 1) if sharpness is not None else 0      
        naturalness_norm = np.clip(naturalness, 0, 1) if naturalness is not None else 0         

        score = (weights['contrast'] * contrast_norm +
                 weights['saturation'] * saturation_norm +
                 weights['sharpness'] * sharpness_norm +
                 weights['naturalness'] * naturalness_norm)
        return score
    except Exception as e:
        logging.error(f"Quality score calculation failed: {str(e)}", exc_info=True)
        return 0.0 


def run(image_path, out_path, depth_out_path, before_post_process_save_path, args_obj, log_file_path):
    """
    Main processing function for a single image (NO AWB but WITH Post-Processing).
    Args:
        image_path (str): Path to the input image.
        out_path (str): Path to save the final enhanced image.
        depth_out_path (str): Path to save the depth map.
        before_post_process_save_path (str): Path to save the image after SeaThru but before final post-processing.
        args_obj (argparse.Namespace): Command-line arguments.
        log_file_path (str): Path to the individual log file for this image.
    """
    DEVICE_CHOICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logging.info(f"Using device: {DEVICE_CHOICE} for PyTorch operations.") 
    
    grayscale_depth_np = None 
    initial_img_np_0_1 = None 

    try:
        raw_img_pil = pil.open(image_path).convert('RGB') 
        raw_img_np = np.array(raw_img_pil)
    except FileNotFoundError:
        logging.error(f"Input image not found: {image_path}")
        return
    except Exception as e:
        logging.error(f"Error opening image {image_path}: {e}", exc_info=True)
        return

    # Skip white balance step
    logging.info("White balance is SKIPPED in this version.")
    initial_img_np_0_1 = scale(raw_img_np)

    # Depth Estimation Step
    img_for_depth_estimation = initial_img_np_0_1

    if DepthAnythingV2 is None:
        logging.error("DepthAnythingV2 model class not loaded. Using dummy depth map.")
        h, w = img_for_depth_estimation.shape[:2]
        dummy_depth = np.zeros((h, w), dtype=np.float32)
        grayscale_depth_np = (dummy_depth * 255).astype(np.uint8)
    else:
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        depth_model_config = model_configs.get(args_obj.encoder, model_configs['vits']) 
        depth_anything_model = DepthAnythingV2(**depth_model_config)
        checkpoint_path = f'checkpoints/depth_anything_v2_{args_obj.encoder}.pth'
        try:
            depth_anything_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            depth_anything_model = depth_anything_model.to(DEVICE_CHOICE).eval()
            
            img_for_depth_uint8 = (np.clip(img_for_depth_estimation, 0, 1) * 255).astype(np.uint8)

            with torch.no_grad():
                depth_map_direct_output = depth_anything_model.infer_image(img_for_depth_uint8, args_obj.size) 

            if isinstance(depth_map_direct_output, torch.Tensor):
                depth_map_np = depth_map_direct_output.cpu().numpy()
            elif isinstance(depth_map_direct_output, np.ndarray):
                depth_map_np = depth_map_direct_output
            else:
                logging.error(f"Depth model infer_image returned unexpected type: {type(depth_map_direct_output)}. Attempting to convert.")
                try: depth_map_np = np.array(depth_map_direct_output)
                except Exception as e_conv:
                    logging.error(f"Could not convert depth output to NumPy array: {e_conv}", exc_info=True)
                    raise TypeError(f"Depth model infer_image returned unconvertible type: {type(depth_map_direct_output)}")

            if depth_map_np.max() - depth_map_np.min() < 1e-6: 
                depth_map_np_scaled = np.full_like(depth_map_np, 0.5, dtype=np.float32)
            else:
                depth_map_np_scaled = (depth_map_np - depth_map_np.min()) / (depth_map_np.max() - depth_map_np.min())
            
            depth_map_np_adjusted = 0.10 + 0.90 * depth_map_np_scaled 
            grayscale_depth_np = (np.clip(depth_map_np_adjusted,0,1) * 255).astype(np.uint8)

        except FileNotFoundError:
            logging.error(f"Depth model checkpoint '{checkpoint_path}' not found! Using dummy depth.")
            h, w = img_for_depth_estimation.shape[:2]
            dummy_depth = np.zeros((h, w), dtype=np.float32)
            grayscale_depth_np = (dummy_depth * 255).astype(np.uint8)
        except Exception as e_depth: 
            logging.error(f"Error during depth estimation: {e_depth}", exc_info=True)
            h, w = img_for_depth_estimation.shape[:2]
            dummy_depth = np.zeros((h, w), dtype=np.float32)
            grayscale_depth_np = (dummy_depth * 255).astype(np.uint8)

    if grayscale_depth_np is None: 
        logging.error("Grayscale depth map is None. Using a black image placeholder.")
        h, w = img_for_depth_estimation.shape[:2]
        grayscale_depth_np = np.zeros((h, w), dtype=np.uint8)

    logging.info("Depth estimation processed.")
    img_for_seathru = initial_img_np_0_1

    # SeaThru Processing
    if not hasattr(run, 'seathru_processor_instance'):
        preset = args_obj.quality_preset if hasattr(args_obj, 'quality_preset') else 'balanced'
        run.seathru_processor_instance = PhysicsBasedSeaThru(quality_preset=preset)

    seathru_args = argparse.Namespace(p=args_obj.p, f=args_obj.f) 
    depths_for_seathru_0_1 = grayscale_depth_np.astype(np.float32) / 255.0

    recovered_np_0_1 = None
    try:
        if args_obj.use_gpu and torch.cuda.is_available() and cp is not None:
            logging.info("Using GPU for SeaThru.")
            recovered_np_0_1 = run.seathru_processor_instance.process_image(img_for_seathru, depths_for_seathru_0_1, seathru_args)
        else:
            logging.info("Using CPU for SeaThru (improved_seathru.run_pipeline).")
            recovered_np_0_1 = run_pipeline(img_for_seathru, depths_for_seathru_0_1, seathru_args) 
    except Exception as e_seathru:
        logging.error(f"Error during SeaThru processing: {e_seathru}", exc_info=True)
        recovered_np_0_1 = img_for_seathru # Fallback to image before SeaThru

    if recovered_np_0_1 is None: 
        logging.warning("SeaThru processing returned None. Using scaled raw image.")
        recovered_np_0_1 = initial_img_np_0_1

    # ===== Save image after SeaThru but BEFORE final post-processing =====
    if recovered_np_0_1 is not None:
        try:
            # Ensure the image is in [0, 1] range and float type for consistent processing
            img_before_pp_0_1 = np.clip(recovered_np_0_1.astype(np.float32), 0, 1)
            
            # Convert to uint8 for saving as PNG
            img_before_pp_uint8 = (img_before_pp_0_1 * 255).astype(np.uint8)
            
            # Create PIL image
            img_pil_before_pp = pil.fromarray(img_before_pp_uint8)
            
            # Save the image
            img_pil_before_pp.save(before_post_process_save_path)
            logging.info(f"Saved image after SeaThru (before final post-processing) to {before_post_process_save_path}")
        except Exception as e:
            logging.error(f"Failed to save image before final post-processing: {e}", exc_info=True)
    else:
        logging.warning("recovered_np_0_1 is None, cannot save image before final post-processing.")

    # Apply final post-processing
    logging.info("Applying final post-processing step.")
    input_to_final_pp = recovered_np_0_1 if recovered_np_0_1 is not None else initial_img_np_0_1
    recovered_final_np_0_1 = post_process_enhanced(input_to_final_pp)
    
    if recovered_final_np_0_1 is None: # If post_process_enhanced also returns None
        logging.error("Final post-processing returned None. Saving the image that went into final post-processing instead.")
        recovered_final_np_0_1 = input_to_final_pp # Fallback to its input
        if recovered_final_np_0_1 is None: # If that was also None, extreme fallback
             recovered_final_np_0_1 = scale(raw_img_np)

    # Save final enhanced image
    output_img_pil = pil.fromarray((np.clip(recovered_final_np_0_1, 0, 1) * 255.0).astype(np.uint8))
    try:
        output_img_pil.save(out_path)
        logging.info(f"Saved final enhanced image to {out_path}")
    except Exception as e:
        logging.error(f"Failed to save final enhanced image to {out_path}: {e}", exc_info=True)

    # Save depth map
    depth_img_pil = pil.fromarray(grayscale_depth_np)
    try:
        depth_img_pil.save(depth_out_path)
        logging.info(f"Saved depth map to {depth_out_path}")
    except Exception as e:
        logging.error(f"Failed to save depth map to {depth_out_path}: {e}", exc_info=True)


def process_file(file_name, base_input_path, base_output_path, base_depth_path, base_before_post_process_path, base_log_path, cmd_args):
    """
    Wrapper function to process a single file, managing paths and logging.
    """
    input_file_path = os.path.join(base_input_path, file_name)
    base_name = os.path.splitext(file_name)[0]
    output_file_name = base_name + '.png' 
    output_file_path = os.path.join(base_output_path, output_file_name)
    depth_mipr_file_path = os.path.join(base_depth_path, base_name + '_depth.png')
    
    # Path for image before final post-processing (saved in BeforePostProcessImages)
    before_pp_file_name = base_name + '_before_pp.png' # Suffix for image before final post-process
    before_post_process_file_path = os.path.join(base_before_post_process_path, before_pp_file_name)

    log_file_path_ind = os.path.join(base_log_path, base_name + '_processing_log.txt')

    if os.path.isfile(input_file_path):
        file_logger = logging.getLogger(file_name) 
        if not file_logger.handlers or not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path_ind for h in file_logger.handlers):
            for handler in file_logger.handlers[:]:
                handler.close()
                file_logger.removeHandler(handler)
            fh = logging.FileHandler(log_file_path_ind, mode='w') 
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            file_logger.addHandler(fh)
            file_logger.setLevel(logging.INFO)
            file_logger.propagate = False

        file_logger.info(f'Starting processing for file: {file_name}')
        img_start_time = time.time()
        try:
            run(input_file_path, output_file_path, depth_mipr_file_path, before_post_process_file_path, cmd_args, log_file_path_ind)
        except Exception as e:
            file_logger.error(f"CRITICAL ERROR processing {file_name}: {e}", exc_info=True) 
        img_end_time = time.time()
        img_elapsed_time = img_end_time - img_start_time
        file_logger.info(f"Finished processing {file_name} in {img_elapsed_time:.2f}s")
        
        for handler in file_logger.handlers[:]: 
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file_path_ind:
                handler.close()
                file_logger.removeHandler(handler)
    else:
        logging.warning(f"File not found, skipping: {input_file_path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)]) 

    parser = argparse.ArgumentParser(description="Underwater Image Enhancement Main Script (NO AWB but WITH Post-Processing)")
    parser.add_argument('--input', type=str, default=r'E:\Desktop\UIEAnythingv2\UIEB890\raw-890', help='Input image or folder path')
    parser.add_argument('--output_folder', type=str, default='./output_results_UIEB890_noawb_withpp_fast_UIEB8900619_improved_seathru', help='Folder to save results')
    parser.add_argument('--f', type=float, default=2.5, help='SeaThru brightness factor f') 
    parser.add_argument('--l', type=float, default=0.5, help='SeaThru attenuation balance l (not directly used in simplified SeaThru args)')
    parser.add_argument('--p', type=float, default=0.15, help='SeaThru illumination locality p') 
    parser.add_argument('--min-depth', type=float, default=0.0, help='Min depth for estimation (0-1) (SeaThru internal)')
    parser.add_argument('--max-depth', type=float, default=1.0, help='Max depth for estimation (0-1) (SeaThru internal)')
    parser.add_argument('--spread-data-fraction', type=float, default=0.05, help='(SeaThru internal)')
    parser.add_argument('--size', type=int, default=518, help='Target size for depth estimation (e.g., 518 for DepthAnythingV2)') 
    parser.add_argument('--raw', action='store_true', help='Input is RAW (not currently implemented in this flow)')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'], help='Depth Anything V2 encoder type')
    parser.add_argument('--use-gpu', action='store_true', default=False, help='Use GPU for SeaThru if available (PyTorch device selection is separate)')
    parser.add_argument('--quality_preset', type=str, default='fast', choices=['fast', 'balanced', 'quality'], help='SeaThru quality preset for GPU')

    args = parser.parse_args()

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_folder = os.path.join(args.output_folder, timestamp_str)

    output_images_path = os.path.join(main_output_folder, "EnhancedImages")
    depth_maps_path = os.path.join(main_output_folder, "DepthMaps")
    before_post_process_images_path = os.path.join(main_output_folder, "BeforePostProcessImages") # New folder
    logs_path = os.path.join(main_output_folder, "Logs")

    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(depth_maps_path, exist_ok=True)
    os.makedirs(before_post_process_images_path, exist_ok=True) # Create the new directory
    os.makedirs(logs_path, exist_ok=True)

    main_log_file_path = os.path.join(logs_path, f"processing_summary_{timestamp_str}.txt")
    main_file_handler = logging.FileHandler(main_log_file_path, mode='w')
    main_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(main_file_handler) 

    if os.path.isfile(args.input):
        files_to_process = [os.path.basename(args.input)]
        input_base_path = os.path.dirname(args.input)
    elif os.path.isdir(args.input):
        files_to_process = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))
                            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))] 
        files_to_process = natsort.natsorted(files_to_process)
        input_base_path = args.input
    else:
        logging.error(f"Input path {args.input} is not a valid file or directory.")
        sys.exit(1)

    if not files_to_process:
        logging.info(f"No image files found in {args.input}.")
        sys.exit(0)

    logging.info(f"Found {len(files_to_process)} images to process.")
    total_start_time = time.time()
    
    cpu_cores = multiprocessing.cpu_count()
    # Adjust num_processes: min 1, max 4, and generally cpu_cores - 1 for safety.
    num_processes = min(max(1, cpu_cores - 1 if cpu_cores > 1 else 1), 4) 

    # Condition for parallelism: more than 1 process, not on Windows (for stability with spawn/CUDA), and enough files.
    can_parallelize = num_processes > 1 and sys.platform != "win32" and len(files_to_process) >= num_processes * 1.5

    if can_parallelize:
        logging.info(f"Using {num_processes} processes for parallel execution.")
        try:
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                 multiprocessing.set_start_method('spawn', force=True) 
                 logging.info("Successfully set multiprocessing start method to 'spawn'.")
            else:
                 logging.info("Multiprocessing start method already 'spawn' or cannot be changed now.")
        except RuntimeError as e_sm: 
            logging.warning(f"Could not set multiprocessing start_method to 'spawn' ({e_sm}), using default or previously set method.")
        
        pool_args = [(file_item, input_base_path, output_images_path, depth_maps_path, 
                      before_post_process_images_path, logs_path, args) 
                     for file_item in files_to_process]
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            list(tqdm(pool.starmap(process_file, pool_args), total=len(files_to_process), desc="Processing images (parallel)"))
            
    else:
        if sys.platform == "win32" and num_processes > 1 and len(files_to_process) >= num_processes * 1.5 : 
            logging.info("On Windows, using sequential execution for stability with CUDA, or due to other parallelism constraints.")
        else:
            logging.info("Using sequential execution (not enough files or cores for effective parallelism, or single core detected).")

        for file_item in tqdm(files_to_process, desc="Processing images (sequential)"):
             process_file(file_item, input_base_path, output_images_path, depth_maps_path, 
                          before_post_process_images_path, logs_path, args)

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    logging.info(f'Total processing time for {len(files_to_process)} images: {total_elapsed_time:.2f} seconds.')
    if files_to_process: 
        avg_time_per_image = total_elapsed_time/len(files_to_process) if len(files_to_process) > 0 else 0
        logging.info(f'Average time per image: {avg_time_per_image:.2f} seconds.')

    logging.info(f"All results saved in: {main_output_folder}")
    logging.info(f"Summary log saved to: {main_log_file_path}")
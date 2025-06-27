"""
CUDA环境配置模块
================
解决RTX 4090 + CUDA 12.x + CuPy兼容性问题
必须在所有CUDA相关导入之前执行
"""

import os
import sys
import platform
import subprocess
import warnings

def setup_openmp_environment():
    """
    配置OpenMP环境，解决多个OpenMP运行时冲突问题
    """
    # 解决OpenMP冲突的关键设置
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 额外的OpenMP优化设置
    if 'OMP_NUM_THREADS' not in os.environ:
        # 设置OpenMP线程数，通常设为CPU核心数
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(max(1, num_cores - 2))  # 留2个核心给系统
    
    # Intel MKL特定设置（如果使用）
    os.environ['MKL_NUM_THREADS'] = os.environ['OMP_NUM_THREADS']
    os.environ['NUMEXPR_NUM_THREADS'] = os.environ['OMP_NUM_THREADS']
    
    # 禁用OpenMP嵌套并行（可能导致性能问题）
    os.environ['OMP_NESTED'] = 'FALSE'
    
    print(f"✅ OpenMP环境已配置: {os.environ['OMP_NUM_THREADS']} 线程")

def setup_library_import_order():
    """
    设置库导入顺序以最小化OpenMP冲突
    """
    # 预加载关键库的顺序很重要
    try:
        # 首先加载NumPy（通常包含MKL）
        import numpy as np
        print(f"✅ NumPy {np.__version__} 已加载")
        
        # 然后加载PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__} 已加载")
        
        # 最后加载其他科学计算库
        import scipy
        print(f"✅ SciPy {scipy.__version__} 已加载")
    except ImportError as e:
        print(f"⚠️ 库导入警告: {e}")

def setup_cuda_environment():
    """
    配置CUDA编译环境，确保兼容性
    """
    # 首先解决OpenMP冲突
    setup_openmp_environment()
    
    # 1. 基础C++标准配置
    os.environ['CCCL_IGNORE_DEPRECATED_CPP_DIALECT'] = '1'
    os.environ['CUDACXX_FLAGS'] = '-std=c++17'
    
    # 2. NVCC编译器标志
    nvcc_flags = [
        '-std=c++17',
        '-DCCCL_IGNORE_DEPRECATED_CPP_DIALECT',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
        '--expt-relaxed-constexpr',
        '-gencode=arch=compute_89,code=sm_89',  # RTX 4090专用
    ]
    
    os.environ['NVCC_PREPEND_FLAGS'] = ' '.join(nvcc_flags)
    
    # 3. CuPy特定配置
    os.environ['CUPY_CACHE_IN_MEMORY'] = '1'
    os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = '1'  # 保存源码便于调试
    os.environ['CUPY_CUDA_COMPILE_WITH_DEBUG'] = '0'  # 发布模式
    os.environ['CUPY_NVCC_GENERATE_CODE'] = 'arch=compute_89,code=sm_89'
    
    # 4. CUDA运行时配置
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步执行
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # 5. Windows特定配置
    if platform.system() == 'Windows':
        # 确保使用正确的编译器
        vs_path = find_visual_studio_compiler()
        if vs_path:
            os.environ['CUDACXX'] = vs_path
            os.environ['NVCC_CCBIN'] = vs_path
        
        # Windows特定的DLL搜索路径优化
        setup_windows_dll_search_path()
    
    # 6. 验证CUDA安装
    verify_cuda_installation()
    
    # 7. 设置库导入顺序
    setup_library_import_order()

def setup_windows_dll_search_path():
    """
    Windows特定：优化DLL搜索路径以避免冲突
    """
    if platform.system() == 'Windows':
        # 添加CUDA和cuDNN路径
        cuda_path = os.environ.get('CUDA_PATH', 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9')
        if os.path.exists(cuda_path):
            cuda_bin = os.path.join(cuda_path, 'bin')
            if cuda_bin not in os.environ['PATH']:
                os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']
                print(f"✅ 添加CUDA路径到PATH: {cuda_bin}")
        
        # 尝试使用延迟DLL加载
        try:
            # Python 3.8+
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(cuda_bin)
                print("✅ 使用add_dll_directory优化DLL加载")
        except Exception as e:
            print(f"⚠️ DLL目录设置警告: {e}")

def find_visual_studio_compiler():
    """
    查找Visual Studio C++编译器
    """
    possible_paths = [
        r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64\cl.exe',
        r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64\cl.exe',
        r'C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64\cl.exe',
        r'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到VS编译器: {path}")
            return path
    
    # 尝试通过vswhere查找
    try:
        vswhere_path = r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
        if os.path.exists(vswhere_path):
            result = subprocess.run([vswhere_path, '-latest', '-property', 'installationPath'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                install_path = result.stdout.strip()
                # 构建cl.exe路径
                import glob
                pattern = os.path.join(install_path, 'VC', 'Tools', 'MSVC', '*', 'bin', 'Hostx64', 'x64', 'cl.exe')
                cl_paths = glob.glob(pattern)
                if cl_paths:
                    print(f"✅ 通过vswhere找到编译器: {cl_paths[0]}")
                    return cl_paths[0]
    except Exception as e:
        print(f"⚠️ vswhere查找失败: {e}")
    
    print("⚠️ 未找到Visual Studio编译器，使用默认设置")
    return None

def verify_cuda_installation():
    """
    验证CUDA安装和版本
    """
    try:
        # 检查nvcc
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVCC编译器可用")
            print(result.stdout.split('\n')[3])  # 打印版本信息
        else:
            print("⚠️ NVCC编译器不可用")
    except FileNotFoundError:
        print("❌ 未找到nvcc，请确保CUDA已正确安装并添加到PATH")

def get_gpu_info():
    """
    获取GPU信息
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"\n🎯 检测到 {gpu_count} 个GPU设备:")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                capability = torch.cuda.get_device_capability(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {name}")
                print(f"    - 计算能力: {capability}")
                print(f"    - 显存: {memory:.1f}GB")
                
                # RTX 4090特定优化
                if "RTX 4090" in name:
                    print(f"    - 🚀 启用RTX 4090专用优化")
                    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    # 针对RTX 4090的内存池配置
                    torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%的GPU内存
    except ImportError:
        print("⚠️ PyTorch未安装，跳过GPU检测")

def diagnose_openmp_conflict():
    """
    诊断OpenMP冲突的来源
    """
    print("\n🔍 诊断OpenMP配置:")
    
    # 检查已加载的DLL（Windows）
    if platform.system() == 'Windows':
        try:
            import psutil
            import os
            current_process = psutil.Process(os.getpid())
            dlls = [dll.path for dll in current_process.memory_maps() if 'iomp' in dll.path.lower() or 'omp' in dll.path.lower()]
            if dlls:
                print("  检测到的OpenMP DLL:")
                for dll in dlls:
                    print(f"    - {dll}")
        except Exception as e:
            print(f"  无法检查DLL: {e}")
    
    # 检查环境变量
    omp_vars = [var for var in os.environ if 'OMP' in var or 'MKL' in var]
    if omp_vars:
        print("  OpenMP相关环境变量:")
        for var in sorted(omp_vars):
            print(f"    - {var}={os.environ[var]}")

# 立即执行环境设置
print("=" * 60)
print("CUDA和OpenMP环境初始化")
print("=" * 60)
setup_cuda_environment()
get_gpu_info()
# 诊断OpenMP配置（可选，用于调试）
# diagnose_openmp_conflict()
print("=" * 60)

# 导出给其他模块使用
__all__ = ['setup_cuda_environment', 'get_gpu_info', 'setup_openmp_environment']
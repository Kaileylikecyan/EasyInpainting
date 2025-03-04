#参考官方文档：https://scikit-image.org/docs/stable/api/skimage.metrics.html

import os
from skimage import io, metrics
import numpy as np
from PIL import Image

def calculate_psnr_and_ssim(original_folder, repaired_folder):
    psnr_scores = []
    ssim_scores = []
    
    # 获取两个文件夹中的文件列表
    original_files = sorted(os.listdir(original_folder))
    repaired_files = sorted(os.listdir(repaired_folder))
    
    # 确保文件列表长度相同
    assert len(original_files) == len(repaired_files), "The number of files in both folders must be the same."
    
    # 遍历文件列表
    for orig_file, rep_file in zip(original_files, repaired_files):
        # 读取原始图像和修复后的图像
        original_img = io.imread(os.path.join(original_folder, orig_file))
        repaired_img = io.imread(os.path.join(repaired_folder, rep_file))
        
        # 确保图像尺寸和颜色通道相同
        assert original_img.shape == repaired_img.shape, "Image dimensions must match."
        
        # 计算PSNR
        psnr_value = metrics.peak_signal_noise_ratio(original_img, repaired_img)
        psnr_scores.append(psnr_value)
        
        # 计算SSIM
        ssim_value = metrics.structural_similarity(original_img, repaired_img, multichannel=True)
        ssim_scores.append(ssim_value)
    
    # 计算平均PSNR和SSIM
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    
    return avg_psnr, avg_ssim

# 示例用法
original_folder = 'path/to/original/images'
repaired_folder = 'path/to/repaired/images'
avg_psnr, avg_ssim = calculate_psnr_and_ssim(original_folder, repaired_folder)

print(f"Average PSNR: {avg_psnr}")
print(f"Average SSIM: {avg_ssim}")

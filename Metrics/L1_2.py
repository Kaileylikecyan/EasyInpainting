import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor

def calculate_l1_l2_loss(original_folder, repaired_folder):
    l1_losses = []
    l2_losses = []

    # 获取两个文件夹中的文件列表
    original_files = sorted(os.listdir(original_folder))
    repaired_files = sorted(os.listdir(repaired_folder))

    # 确保文件列表长度相同
    assert len(original_files) == len(repaired_files), "The number of files in both folders must be the same."

    # 遍历文件列表
    for original_file, repaired_file in zip(original_files, repaired_files):
        # 构建完整的文件路径
        original_path = os.path.join(original_folder, original_file)
        repaired_path = os.path.join(repaired_folder, repaired_file)

        # 加载图像
        original_img = Image.open(original_path).convert('RGB')
        repaired_img = Image.open(repaired_path).convert('RGB')

        # 将图像转换为张量
        original_tensor = ToTensor()(original_img).unsqueeze(0)  # 添加批次维度
        repaired_tensor = ToTensor()(repaired_img).unsqueeze(0)  # 添加批次维度

        # 计算L1和L2损失
        l1_loss = torch.nn.functional.l1_loss(original_tensor, repaired_tensor)
        l2_loss = torch.nn.functional.mse_loss(original_tensor, repaired_tensor)

        # 将损失添加到列表中
        l1_losses.append(l1_loss.item())
        l2_losses.append(l2_loss.item())

    # 计算平均损失
    avg_l1_loss = sum(l1_losses) / len(l1_losses)
    avg_l2_loss = sum(l2_losses) / len(l2_losses)

    return avg_l1_loss, avg_l2_loss

# 示例用法
original_folder = 'path/to/original/images'
repaired_folder = 'path/to/repaired/images'
avg_l1_loss, avg_l2_loss = calculate_l1_l2_loss(original_folder, repaired_folder)

print(f"Average L1 Loss: {avg_l1_loss}")
print(f"Average L2 Loss: {avg_l2_loss}")

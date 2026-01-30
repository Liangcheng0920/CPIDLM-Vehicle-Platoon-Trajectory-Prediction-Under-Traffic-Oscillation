import os
import torch
import torch.nn as nn

# 设置环境变量解决某些库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 自动选择计算设备
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# 初始化全局设备变量，方便其他模块调用
device = get_device()

def check_data(data, name="data"):
    """
    检查输入数据中是否包含NaN或Inf值。
    """
    # print(f"正在检查 {name} 是否包含 NaN 或 Inf 值...") # 可选：减少日志输出
    has_nan = torch.isnan(data).any().item()
    has_inf = torch.isinf(data).any().item()
    if has_nan or has_inf:
        print(f"警告: {name} 包含 NaN: {has_nan}, Inf: {has_inf}!")

def initialize_weights(model):
    """
    对神经网络模型的权重进行Xavier均匀初始化，偏置项初始化为0。
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            if param.data.dim() > 1:
                nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)
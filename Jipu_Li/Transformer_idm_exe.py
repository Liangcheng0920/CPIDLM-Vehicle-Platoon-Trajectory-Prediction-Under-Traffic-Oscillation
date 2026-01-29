import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --- 数据检查函数 ---
def check_data(data, name="data"):
    """
    检查PyTorch张量中是否存在NaN或Inf值。

    Args:
        data (torch.Tensor): 要检查的张量。
        name (str): 数据的名称，用于打印输出。
    """
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# --- 定义混合模型 (将 LSTM 替换为 Transformer) ---
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        """
        融合了Transformer编码器和IDM（Intelligent Driver Model）的混合模型。

        Args:
            input_dim (int): 输入特征的维度。
            model_dim (int): Transformer模型的隐藏维度。
            num_heads (int): Transformer多头注意力机制的头数。
            num_layers (int): Transformer编码器层的数量。
            dropout (float): Dropout比率。
        """
        super(HybridIDMModel, self).__init__()
        self.model_dim = model_dim
        # 输入线性层，将输入维度转换为Transformer的model_dim
        self.input_linear = nn.Linear(input_dim, model_dim)
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全连接层，用于输出IDM参数
        self.fc = nn.Linear(model_dim, 6)  # IDM参数：[v_desired, T, a_max, b_safe, delta, s0]
        self.softplus = nn.Softplus()  # 确保参数为正
        self.delta_t = 0.1  # 时间步长

    def forward(self, x):
        """
        模型的前向传播。

        Args:
            x (torch.Tensor): 输入数据，形状为 (batch_size, sequence_length, input_dim)。

        Returns:
            torch.Tensor: 预测的IDM参数。
        """
        # 将输入维度转换为model_dim
        x = self.input_linear(x)
        # 通过Transformer编码器
        out = self.transformer_encoder(x)
        # 取最后一个时间步的输出，通过全连接层预测IDM参数
        params = self.fc(out[:, -1, :])
        params = self.softplus(params)  # 确保参数为正
        return params

    def predict_speed(self, x, s_safe):
        """
        根据输入和IDM参数预测下一时刻的速度。

        Args:
            x (torch.Tensor): 输入数据，形状为 (batch_size, sequence_length, input_dim)。
                              x[:, -1, 0] 为当前速度 v_n
                              x[:, -1, 2] 为速度差 delta_v
            s_safe (torch.Tensor): 安全距离 s_safe，形状为 (batch_size,)。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 预测速度和IDM参数。
        """
        params = self.forward(x)
        v_n = x[:, -1, 0]  # 当前速度
        delta_v = x[:, -1, 2]  # 速度差

        # 提取IDM参数
        v_desired, T, a_max, b_safe, delta, s0 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:,
                                                                                                         4], params[:,
                                                                                                             5]

        # IDM模型核心计算
        s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe))
        s_star = torch.clamp(s_star, min=0)  # 确保s_star非负

        # 计算加速度并更新速度
        v_follow = v_n + self.delta_t * a_max * (1 - (v_n / v_desired) ** delta - (s_star / s_safe) ** 2)
        predicted_speed = torch.clamp(v_follow, min=0)  # 确保预测速度非负
        return predicted_speed, params


# --- 初始化权重和偏置 ---
def initialize_weights(model):
    """
    使用Xavier均匀分布初始化模型的权重，偏置初始化为0。
    修正：对偏置不使用Xavier初始化，直接置为0。

    Args:
        model (nn.Module): 要初始化权重的模型。
    """
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:  # 确保是权重且维度至少为2
            nn.init.xavier_uniform_(param)
        elif "bias" in name:  # 偏置直接置为0
            nn.init.constant_(param, 0)
        # 对于其他情况（如1D权重或不符合条件的权重），可以跳过或采用其他初始化方式
        # 例如，对于Transformer的层归一化参数，通常默认初始化即可，或者使用nn.init.ones_ for weight, zeros_ for bias


# --- 模型训练 ---
def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
    """
    训练模型。

    Args:
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        optimizer (Optimizer): 优化器。
        criterion (LossFunction): 损失函数。
        num_epochs (int): 训练轮数。

    Returns:
        nn.Module: 训练好的模型。
    """
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed, batch_s_safe in train_loader:
            optimizer.zero_grad()  # 梯度清零
            predicted_speed, _ = model.predict_speed(batch_data, batch_s_safe)
            loss = criterion(predicted_speed, batch_speed)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()  # 累加损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    return model


# --- 模型评估 ---
def evaluate_model(model, test_loader):
    """
    评估模型在测试集上的性能，并绘制真实速度与预测速度的对比图。

    Args:
        model (nn.Module): 要评估的模型。
        test_loader (DataLoader): 测试数据加载器。
    """
    model.eval()  # 设置模型为评估模式
    mse_loss = nn.MSELoss()
    total_mse = 0
    all_predicted = []
    all_true = []
    with torch.no_grad():  # 在评估阶段禁用梯度计算
        for batch_idx, (batch_data, batch_speed, batch_s_safe) in enumerate(test_loader):
            predicted_speed, params = model.predict_speed(batch_data, batch_s_safe)
            loss = mse_loss(predicted_speed, batch_speed)
            total_mse += loss.item()
            all_predicted.append(predicted_speed)
            all_true.append(batch_speed)

    mse = total_mse / len(test_loader)
    rmse = torch.sqrt(torch.tensor(mse))
    mae = torch.mean(torch.abs(torch.cat(all_predicted) - torch.cat(all_true))).item()
    print(f"Evaluation Metrics:\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # 绘制对比图
    plt.figure(figsize=(10, 6))
    # 仅绘制前100个样本
    plt.plot(torch.cat(all_true)[:100].cpu().numpy(), label='True Speed', linestyle='--', marker='o')
    plt.plot(torch.cat(all_predicted)[:100].cpu().numpy(), label='Predicted Speed', linestyle='-', marker='x')
    plt.title('True vs Predicted Speed (Hybrid IDM-Transformer)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()


def save_predictions_to_csv(model, test_loader, output_file="predictions.csv"):
    """
    保存测试数据的预测结果到CSV文件。

    Args:
        model (nn.Module): 训练好的模型。
        test_loader (DataLoader): 测试数据加载器。
        output_file (str): 输出CSV文件的路径。
    """
    model.eval()  # 设置模型为评估模式
    all_true_speeds = []
    all_predicted_speeds = []
    all_params = []

    with torch.no_grad():  # 在评估阶段禁用梯度计算
        for batch_data, batch_speed, batch_s_safe in test_loader:
            # 获取模型预测输出
            predicted_speed, params = model.predict_speed(batch_data, batch_s_safe)

            # 保存真实速度和预测参数
            all_true_speeds.append(batch_speed.cpu())
            all_predicted_speeds.append(predicted_speed.cpu())
            all_params.append(params.cpu())

    # 转换为Tensor格式并拼接
    all_true_speeds = torch.cat(all_true_speeds).numpy()
    all_predicted_speeds = torch.cat(all_predicted_speeds).numpy()
    all_params = torch.cat(all_params).numpy()

    # 合并为一个完整的二维数组
    result_data = pd.DataFrame({
        "True Speed": all_true_speeds.flatten(),  # 确保展平
        "Predicted Speed": all_predicted_speeds.flatten(),  # 确保展平
        "v_desired": all_params[:, 0],
        "T": all_params[:, 1],
        "a_max": all_params[:, 2],
        "b_safe": all_params[:, 3],
        "delta": all_params[:, 4],
        "s0": all_params[:, 5]
    })

    # 保存为CSV文件
    result_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def compute_position_and_spacing_and_save(model,
                                          test_data,
                                          test_real_speed,
                                          raw_data,
                                          label_data,
                                          train_size,
                                          test_s_safe,
                                          dt=0.1,
                                          output_file="predictions_extended.xlsx"
                                          ):
    """
    基于预测车速，推算下一时刻Y坐标与车头间距，并保存结果到Excel文件。

    Args:
        model (nn.Module): 训练好的模型。
        test_data (torch.Tensor): 测试集的输入数据。
        test_real_speed (torch.Tensor): 测试集的真实速度。
        raw_data (torch.Tensor): 原始全部输入数据（用于获取当前Y坐标）。
        label_data (torch.Tensor): 原始全部标签数据（用于获取真实Y坐标和真实间距）。
        train_size (int): 训练集的大小，用于确定测试集在原始数据中的起始索引。
        test_s_safe (torch.Tensor): 测试集对应的安全距离。
        dt (float): 时间步长。
        output_file (str): 输出Excel文件的路径。
    """
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在评估阶段禁用梯度计算
        pred_speed_tensor, params_tensor = model.predict_speed(test_data, test_s_safe)
        pred_speed = pred_speed_tensor.squeeze().cpu().numpy()
    true_speed = test_real_speed.cpu().numpy()

    N_test = test_data.shape[0]
    idx = np.arange(train_size, train_size + N_test)

    # 获取当前自车Y坐标（单位：ft，并转换为m）
    current_Y_ft = raw_data[idx, -1, 4].cpu().numpy()  # raw_data[:, -1, 4] 是最后一帧的第五个特征（自车Y坐标）
    current_Y_m = current_Y_ft * 0.3048

    # 获取当前自车速度（单位：m/s）
    current_speed_m = test_data[:, -1, 0].cpu().numpy()

    # 获取真实下一时刻的Y坐标和间距（单位：ft，并转换为m）
    true_Y_ft = label_data[idx, -1, 3].cpu().numpy()  # label_data[:, -1, 3] 是下一时刻的自车Y坐标
    true_Y_m = true_Y_ft * 0.3048
    true_spacing_ft = label_data[idx, -1, 1].cpu().numpy()  # label_data[:, -1, 1] 是下一时刻的车头间距
    true_spacing_m = true_spacing_ft * 0.3048

    # 根据预测车速计算位移 (使用等加速度运动公式近似：s = v0*t + 0.5*a*t^2)
    # 假设预测速度是下一时刻的速度，则加速度 a = (pred_speed - current_speed_m) / dt
    acceleration_m = (pred_speed - current_speed_m) / dt
    disp_m = current_speed_m * dt + 0.5 * acceleration_m * dt ** 2

    # 预测下一时刻的Y坐标
    pred_Y_m = current_Y_m + disp_m

    # 为了与原代码逻辑保持一致，并对其进行合理化修改：
    # 假设 label_data[idx, -1, 3] 是真实下一时刻的自车Y坐标
    # 假设 label_data[idx, -1, 1] 是真实下一时刻的车头间距
    # 那么真实的前车Y坐标 = 真实自车Y + 真实间距
    true_leader_Y_m = true_Y_m + true_spacing_m
    # 预测的车头间距 = 真实前车Y - 预测自车Y
    pred_spacing_m = true_leader_Y_m - pred_Y_m

    rmse_Y = np.sqrt(np.mean((pred_Y_m - true_Y_m) ** 2))
    # 避免除以零或非常小的数
    mape_Y = np.mean(np.abs((pred_Y_m - true_Y_m) / (true_Y_m + 1e-6))) * 100
    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m) ** 2))
    mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m) / (true_spacing_m + 1e-6))) * 100

    print(f"Position Error    -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
    print(f"Spacing  Error    -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")

    df = pd.DataFrame({
        "Pred Speed (m/s)": pred_speed,
        "True Speed (m/s)": true_speed,
        "Predicted Y (m)": pred_Y_m,
        "True Y (m)": true_Y_m,
        "Predicted Spacing (m)": pred_spacing_m,
        "True Spacing (m)": true_spacing_m,
    })
    sheet_name = "Transformer-IDM_1"  # 指定新工作表名称

    # 如果文件存在则在其基础上添加新 sheet，否则新建文件
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a" if os.path.exists(output_file) else "w") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"New model results saved to '{output_file}' in sheet '{sheet_name}'.")


# --- 主函数 ---
if __name__ == "__main__":
    torch.manual_seed(42)  # 设置随机种子以保证结果可复现性

    # 加载数据
    # 请根据你的实际数据路径修改此行
    data = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')
    raw_data = torch.tensor(data['train_data'], dtype=torch.float32)  # 原始输入数据
    label_data = torch.tensor(data['lable_data'], dtype=torch.float32)  # 原始标签数据

    print("原始raw_data形状:", raw_data.shape)
    print("原始label_data形状:", label_data.shape)

    # 从原始数据中提取用于训练的特征和标签
    # 这里选择最后50个时间步的特征，并选取特定的列作为输入特征
    # [0] 自车车速, [1] 前车距离, [2] 速度差, [3] 自车加速度, [-1] 前车速度 (假设-1是前车速度，需要根据实际数据确认)
    # 请根据你数据的实际含义调整列索引
    train_data = raw_data[:, -50:, [0, 1, 2, 3, -1]]
    # train_real_speed 是标签数据中最后一帧的第一个特征，即下一时刻的真实速度
    train_real_speed = label_data[:, -1, 0]

    # train_s_safe 是从 train_data 的最后一帧的第2列（索引1）提取的，表示当前时刻的车头间距
    # 这里的 `clone()` 是为了确保 `train_s_safe` 不会共享 `train_data` 的内存，防止意外修改。
    train_s_safe = torch.tensor(train_data[:, -1, 1].clone(), dtype=torch.float32)

    # 打印输出，检查处理后的数据形状
    print(f"train_data shape: {train_data.shape}")
    print(f"train_real_speed shape: {train_real_speed.shape}")
    print(f"train_s_safe shape: {train_s_safe.shape}")

    # 数据预处理：单位转换（例如ft/s转为m/s，ft转为m）
    # 假设原始数据是以英尺(ft)和英尺/秒(ft/s)为单位，转换为米(m)和米/秒(m/s)
    # 对所有相关特征进行转换
    train_data[:, :, 0] *= 0.3048  # 自车车速 (ft/s -> m/s)
    train_data[:, :, 1] *= 0.3048  # 前车距离 (ft -> m)
    train_data[:, :, 2] *= 0.3048  # 速度差 (ft/s -> m/s)
    train_data[:, :, 3] *= 0.3048  # 自车加速度 (ft/s^2 -> m/s^2)
    train_data[:, :, 4] *= 0.3048  # 前车速度 (ft/s -> m/s)
    train_real_speed *= 0.3048  # 真实速度 (ft/s -> m/s)
    train_s_safe *= 0.3048  # 安全距离 (ft -> m)

    # 选取前 10% 的数据进行训练和测试，以加快运行速度 (可根据需要调整或移除)
    total_samples = train_data.shape[0]
    sample_size = int(total_samples * 0.1)
    train_data = train_data[:sample_size]
    train_real_speed = train_real_speed[:sample_size]
    train_s_safe = train_s_safe[:sample_size]

    # 检查数据中是否存在 NaN 或 Inf 值
    check_data(train_data, "train_data")
    check_data(train_real_speed, "train_real_speed")
    check_data(train_s_safe, "train_s_safe")

    # 划分训练集和测试集 (80% 训练, 20% 测试)
    dataset_size = train_data.shape[0]
    train_size = int(dataset_size * 0.8)

    # 分割数据集
    train_x, test_x = train_data[:train_size], train_data[train_size:]
    train_y, test_y = train_real_speed[:train_size], train_real_speed[train_size:]
    train_s_safe_split, test_s_safe_split = train_s_safe[:train_size], train_s_safe[train_size:]

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y, train_s_safe_split)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y, test_s_safe_split)

    batch_size = 32  # 批处理大小
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、权重、损失函数、优化器
    input_dim = train_data.shape[2]  # 输入特征数 (例如 5: 自车车速、前车距离、速度差、自车加速度、前车速度)
    model_dim = 128  # Transformer模型的隐藏维度
    num_heads = 4  # Transformer多头注意力机制的头数
    num_layers = 2  # Transformer编码器层的数量

    # 实例化HybridIDMModel，现在使用Transformer
    model = HybridIDMModel(input_dim, model_dim, num_heads, num_layers)
    initialize_weights(model)  # 初始化模型权重

    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam优化器，学习率为0.0005

    # 训练模型
    num_epochs = 100  # 训练轮数
    print("\n--- 开始模型训练 ---")
    model = train_model(model, train_loader, optimizer, criterion, num_epochs)

    # 评估模型
    print("\n--- 开始模型评估 ---")
    evaluate_model(model, test_loader)

    # 保存预测结果到CSV
    # output_csv_file = "transformer_idm_predictions.csv"
    # save_predictions_to_csv(model, test_loader, output_file=output_csv_file)

    # 计算并保存位置和间距推算结果到Excel
    output_excel_file = "predictions_extended.xlsx"
    compute_position_and_spacing_and_save(
        model,
        test_x,  # 测试集的输入特征
        test_y,  # 测试集的真实速度
        raw_data,  # 原始全部输入数据
        label_data,  # 原始全部标签数据
        train_size,  # 训练集大小
        test_s_safe_split,  # 测试集对应的安全距离
        dt=0.1,  # 时间步长
        output_file=output_excel_file
    )

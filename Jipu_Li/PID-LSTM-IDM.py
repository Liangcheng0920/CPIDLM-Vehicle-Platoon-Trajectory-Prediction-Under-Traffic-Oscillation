# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import pandas as pd
# import numpy as np
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
#
# # --- 数据检查函数 ---
# def check_data(data, name="data"):
#     print(f"Checking {name} for NaN or Inf values...")
#     print(f"Has NaN: {torch.isnan(data).any().item()}")
#     print(f"Has Inf: {torch.isinf(data).any().item()}")
#
#
# # --- 固定 IDM 参数预测函数 ---
# def idm_fixed(v_n, s_safe, delta_v,
#               v_desired=10.13701546, T=0.50290469, a_max= 0.10995557,
#               b_safe=4.98369406, delta=5.35419582, s0=0.10337701,
#               delta_t=0.1):
#     device = v_n.device
#     # 把常数转成 0-d tensor
#     a_max  = torch.tensor(a_max, device=device).clamp(min=1e-6)
#     b_safe = torch.tensor(b_safe, device=device).clamp(min=1e-6)
#     s_safe = s_safe.clamp(min=1e-6)
#
#     s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe))
#     s_star = s_star.clamp(min=0.0)
#
#     v_follow = v_n + delta_t * a_max * (
#         1 - (v_n / v_desired) ** delta - (s_star / s_safe) ** 2
#     )
#     return v_follow.clamp(min=0.0)
#
#
#
# # --- 定义新融合模型 ---
# class HybridIDMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers=2):
#         super(HybridIDMModel, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         # 只输出一个预测速度
#         self.fc = nn.Linear(hidden_dim, 1)
#         # 可学习的 alpha_raw，经 sigmoid 映射到 [0,1]
#         self.alpha_raw = nn.Parameter(torch.tensor(0.0))
#
#     def forward(self, x, s_safe):
#         """
#         :param x:  输入序列，shape=(batch, seq_len, input_dim)
#         :param s_safe: 当前安全距离，shape=(batch,)
#         :return:
#           y_lstm: LSTM 直接输出的预测速度，shape=(batch,)
#           y_idm: 用固定 IDM 计算的预测速度，shape=(batch,)
#           alpha: 当前学习到的权重标量
#         """
#         batch_size = x.size(0)
#         out, _ = self.lstm(x)  # (batch, seq, hidden_dim)
#         y_lstm = self.fc(out[:, -1, :]).squeeze(1)  # (batch,)
#
#         # 计算固定 IDM 预测
#         v_n = x[:, -1, 0]
#         delta_v = x[:, -1, 2]
#         y_idm = idm_fixed(v_n, s_safe, delta_v)
#
#         alpha = torch.sigmoid(self.alpha_raw)  # 标量 [0,1]
#         return y_lstm, y_idm, alpha
#
#
# def initialize_weights(model):
#     for name, param in model.named_parameters():
#         if "weight" in name:
#             nn.init.xavier_uniform_(param)
#         elif "bias" in name:
#             nn.init.constant_(param, 0)
#
#
# # --- 训练函数 ---
# def train_model(model, train_loader, optimizer, num_epochs=30):
#     model.train()
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         for batch_x, batch_y, batch_s_safe in train_loader:
#             optimizer.zero_grad()
#             y_lstm, y_idm, alpha = model(batch_x, batch_s_safe)
#             # 加权 MSE
#             loss = alpha * (y_lstm - batch_y).pow(2).mean() \
#                    + (1 - alpha) * (y_lstm - y_idm).pow(2).mean()
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         print(f"Epoch {epoch + 1}/{num_epochs}  Loss: {epoch_loss / len(train_loader):.6f}  α={alpha.item():.4f}")
#     return model
#
#
# # --- 测试/评估函数 ---
# def evaluate_model(model, test_loader):
#     model.eval()
#     all_pred, all_true = [], []
#     with torch.no_grad():
#         for batch_x, batch_y, batch_s_safe in test_loader:
#             y_lstm, y_idm, alpha = model(batch_x, batch_s_safe)
#             all_pred.append(y_lstm.cpu())
#             all_true.append(batch_y.cpu())
#     y_pred = torch.cat(all_pred)
#     y_true = torch.cat(all_true)
#     mse = nn.MSELoss()(y_pred, y_true).item()
#     rmse = torch.sqrt(torch.tensor(mse)).item()
#     mae = torch.mean(torch.abs(y_pred - y_true)).item()
#     print(f"\nTest Results -- MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, α={alpha.item():.4f}")
#     # 绘图
#     plt.figure(figsize=(10, 6))
#     plt.plot(y_true[:100].numpy(), '--o', label='True')
#     plt.plot(y_pred[:100].numpy(), '-x', label='Pred')
#     plt.legend();
#     plt.grid();
#     plt.show()
#
#
# # === 修改后的 compute_position_and_spacing_and_save ===
# def compute_position_and_spacing_and_save(model,
#                                           test_data,
#                                           test_real_speed,
#                                           raw_data,
#                                           label_data,
#                                           train_size,
#                                           test_s_safe,
#                                           dt=0.1,
#                                           output_file="predictions_extended.xlsx"):
#     """
#     :param model: 已训练好的 HybridIDMModel
#     :param test_data:   torch.Tensor, shape=(N_test, seq_len, feat)
#     :param test_real_speed: torch.Tensor, shape=(N_test,)
#     :param raw_data:    torch.Tensor 原始 ft/ft·s 序列, shape=(N_all, seq_len, feat_raw)
#     :param label_data:  torch.Tensor 标签 ft, shape=(N_all, seq_len, feat_lab)
#     :param train_size:  int, 训练集样本数，用于定位 raw_data/label_data 中测试段的起点
#     :param test_s_safe: torch.Tensor, shape=(N_test,), 安全距离 (m)
#     :param dt:          float, 时间步长 (s)
#     :param output_file: str, 输出 Excel 文件路径
#     """
#     model.eval()
#     with torch.no_grad():
#         # 1) 调用 model，拿到 lstm 预测&idm 预测 & alpha
#         y_lstm, y_idm, alpha = model(test_data, test_s_safe)
#         # 2) 融合输出
#         pred_speed = (alpha * y_lstm + (1 - alpha) * y_idm).cpu().numpy()
#
#     true_speed = test_real_speed.cpu().numpy()
#
#     N_test = test_data.size(0)
#     idx = np.arange(train_size, train_size + N_test)
#
#     # 3) 原始坐标 & 速度
#     current_Y_ft     = raw_data[idx, -1, 4].numpy()      # ft
#     current_speed_m  = test_data[:, -1, 0].numpy()       # m/s
#
#     true_Y_ft        = label_data[idx, -1, 3].numpy()    # ft
#     true_spacing_m   = label_data[idx, -1, 1].numpy() * 0.3048  # 转成 m
#
#     # 4) 计算位移 disp = v_prev*dt + 0.5*a*dt^2
#     v_prev_ft = current_speed_m / 0.3048
#     pred_v_ft = pred_speed      / 0.3048
#     disp_ft   = v_prev_ft * dt + 0.5 * ((pred_v_ft - v_prev_ft) / dt) * (dt ** 2)
#
#     pred_Y_ft = current_Y_ft + disp_ft
#
#     # 5) 转回米，计算间距
#     pred_Y_m       = pred_Y_ft * 0.3048
#     true_Y_m       = true_Y_ft * 0.3048
#     pred_spacing_m = (true_Y_ft - pred_Y_ft) * 0.3048 + true_spacing_m
#
#     # 6) 打印误差
#     rmse_Y  = np.sqrt(np.mean((pred_Y_m - true_Y_m)**2))
#     mape_Y  = np.mean(np.abs((pred_Y_m - true_Y_m)/true_Y_m)) * 100
#     rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m)**2))
#     mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m)/true_spacing_m)) * 100
#
#     print(f"Position Error -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
#     print(f" Spacing  Error -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")
#
#     # 7) 保存到 Excel
#     df = pd.DataFrame({
#         "Pred Speed (m/s)"  : pred_speed,
#         "True Speed (m/s)"  : true_speed,
#         "Predicted Y (m)"   : pred_Y_m,
#         "True Y (m)"        : true_Y_m,
#         "Pred Spacing (m)"  : pred_spacing_m,
#         "True Spacing (m)"  : true_spacing_m,
#     })
#     sheet_name = "PID-LSTM-IDM"
#     mode = "a" if os.path.exists(output_file) else "w"
#     with pd.ExcelWriter(output_file, engine="openpyxl", mode=mode) as writer:
#         df.to_excel(writer, sheet_name=sheet_name, index=False)
#
#     print(f"Results saved to '{output_file}' sheet '{sheet_name}'.")
#
#
# # --- 主流程 ---
# if __name__ == "__main__":
#     torch.manual_seed(42)
#     # --- 加载 MAT 数据 ---
#     data = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')
#     raw = torch.tensor(data['train_data'], dtype=torch.float32)
#     lab = torch.tensor(data['lable_data'], dtype=torch.float32)
#
#     # 取最后 50 步及需要的列
#     # 假设列 [0]=v_n, [1]=s_safe, [2]=Δv, [3]=加速度, [4]=前车速度
#     seq = raw[:, -50:, [0, 1, 2, 3, -1]].clone()
#     y = lab[:, -1, 0].clone().squeeze()
#     s_safe = seq[:, -1, 1]
#
#     # 单位转换 ft→m, ft/s→m/s
#     seq *= 0.3048
#     y *= 0.3048
#     s_safe *= 0.3048
#
#     # 只用前 10% 加速示例
#     N = int(seq.size(0) * 0.1)
#     seq, y, s_safe = seq[:N], y[:N], s_safe[:N]
#
#     # 划分 80/20
#     split = int(N * 0.8)
#     train_size = split
#     train_seq = seq[:split];
#     test_seq = seq[split:]
#     train_y = y[:split];
#     test_y = y[split:]
#     train_s_safe = s_safe[:split];
#     test_s_safe = s_safe[split:]
#     train_ds = torch.utils.data.TensorDataset(seq[:split], y[:split], s_safe[:split])
#     test_ds = torch.utils.data.TensorDataset(seq[split:], y[split:], s_safe[split:])
#     train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
#
#     # 模型/优化器设定
#     input_dim = seq.size(2)
#     hidden_dim = 128
#     model = HybridIDMModel(input_dim, hidden_dim, num_layers=1)
#     initialize_weights(model)
#     optimizer = optim.Adam(model.parameters(), lr=5e-4)
#
#     # 训练 & 评估
#     model = train_model(model, train_loader, optimizer, num_epochs=100)
#     evaluate_model(model, test_loader)
#     compute_position_and_spacing_and_save(
#         model,
#         test_seq,  # test_data
#         test_y,  # test_real_speed
#         raw,
#         lab,
#         train_size,
#         test_s_safe,  # 新增这一行
#         dt=0.1,
#         output_file="predictions_extended.xlsx"
#     )



import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np

# 设置环境变量，允许OpenMP库的副本，避免某些情况下的冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --- 数据检查函数 ---
def check_data(data, name="data"):
    """
    检查PyTorch张量中是否存在NaN（非数字）或Inf（无穷大）值。
    :param data: 待检查的PyTorch张量。
    :param name: 数据名称，用于打印输出。
    """
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# --- 固定 IDM 参数预测函数 ---
def idm_fixed(v_n, s_safe, delta_v,
              v_desired=10.13701546, T=0.50290469, a_max=0.10995557,
              b_safe=4.98369406, delta=5.35419582, s0=0.10337701,
              delta_t=0.1):
    """
    使用固定的参数集执行一步IDM（智能驾驶员模型）预测。
    根据当前车辆状态（速度、与前车间距、速度差）和预设参数，计算下一时刻的速度。

    :param v_n: 当前自车速度 (m/s) - 张量。
    :param s_safe: 当前实际车头间距 (m) - 张量。
    :param delta_v: 当前速度差 (前车速度 - 自车速度, m/s) - 张量。
    :param v_desired: 期望速度 (m/s) - IDM参数。
    :param T: 安全时间 headway (s) - IDM参数。
    :param a_max: 最大加速度 (m/s^2) - IDM参数。
    :param b_safe: 舒适减速度 (m/s^2) - IDM参数。
    :param delta: 加速度指数 (无量纲) - IDM参数。
    :param s0: 最小静止间距 (m) - IDM参数。
    :param delta_t: 时间步长 (s) - 预测的时间间隔。
    :return: 下一时间步的预测自车速度 (m/s) - 张量。
    """
    device = v_n.device  # 获取输入张量所在的设备 (CPU 或 CUDA)
    # 将IDM的常数参数转换为与输入数据相同设备和类型的0维张量
    # clamp(min=1e-6) 防止除以零或无效值
    a_max_t = torch.tensor(a_max, device=device, dtype=v_n.dtype).clamp(min=1e-6)
    b_safe_t = torch.tensor(b_safe, device=device, dtype=v_n.dtype).clamp(min=1e-6)
    s0_t = torch.tensor(s0, device=device, dtype=v_n.dtype)
    v_desired_t = torch.tensor(v_desired, device=device, dtype=v_n.dtype)
    T_t = torch.tensor(T, device=device, dtype=v_n.dtype)
    delta_param_t = torch.tensor(delta, device=device, dtype=v_n.dtype)
    delta_t_tensor = torch.tensor(delta_t, device=device, dtype=v_n.dtype)

    s_safe = s_safe.clamp(min=1e-6)  # 确保间距为正，防止计算错误

    # 计算期望间距 s* (s_star)
    s_star = s0_t + v_n * T_t + (v_n * delta_v) / (2 * torch.sqrt(a_max_t * b_safe_t) + 1e-6)
    s_star = s_star.clamp(min=0.0)  # 期望间距不能为负

    # 处理 v_desired 可能为零的情况
    v_n_ratio = torch.zeros_like(v_n)
    mask_v_desired_nonzero = v_desired_t.abs() > 1e-6
    if mask_v_desired_nonzero.any():
        v_n_ratio[mask_v_desired_nonzero] = (v_n[mask_v_desired_nonzero] / v_desired_t[mask_v_desired_nonzero])

    # 计算加速度项
    acceleration_term = a_max_t * (
            1 - v_n_ratio ** delta_param_t - (s_star / s_safe) ** 2
    )
    # 根据加速度更新速度
    v_follow = v_n + delta_t_tensor * acceleration_term
    return v_follow.clamp(min=0.0)  # 速度不能为负


# --- 定义新融合模型 ---
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        """
        混合模型，结合LSTM和IDM。
        :param input_dim: LSTM输入特征的维度。
        :param hidden_dim: LSTM隐藏层的维度。
        :param num_layers: LSTM的层数。
        """
        super(HybridIDMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层将LSTM的输出映射到预测速度
        self.fc = nn.Linear(hidden_dim, 1)
        # 固定 alpha 值，不再作为可学习参数
        self.alpha = torch.tensor(0.7, dtype=torch.float32)  # 固定 alpha 为 0.7

    def forward(self, x, s_safe):
        """
        模型的前向传播。

        :param x: 输入序列，shape=(batch_size, seq_len, input_dim)。
                  x[:, -1, 0] 是当前自车速度。
                  x[:, -1, 2] 是当前速度差 (前车 - 自车)。
        :param s_safe: 当前安全距离 (m)，shape=(batch_size,)。
        :return: LSTM的预测输出，shape=(batch_size,)。
                 IDM的预测输出 (仅用于内部计算，不直接作为最终输出)。
                 固定 alpha 值。
        """
        batch_size = x.size(0)
        # LSTM前向传播
        out, _ = self.lstm(x)  # out shape: (batch_size, seq_len, hidden_dim)
        # 取LSTM最后一个时间步的输出，并通过全连接层得到预测速度
        y_lstm = self.fc(out[:, -1, :]).squeeze(1)  # y_lstm shape: (batch_size,)

        # 计算固定 IDM 预测所需参数
        v_n = x[:, -1, 0]  # 当前自车速度
        delta_v = x[:, -1, 2]  # 当前速度差
        # 使用固定参数的IDM函数计算预测速度
        y_idm = idm_fixed(v_n, s_safe, delta_v)

        # 最终输出直接使用LSTM的预测
        return y_lstm, y_idm, self.alpha.to(x.device)  # 确保alpha在正确的设备上


def initialize_weights(model):
    """
    初始化模型权重，提高训练稳定性。
    :param model: 待初始化的PyTorch模型。
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            # 使用Xavier均匀初始化权重
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            # 偏置初始化为0
            nn.init.constant_(param, 0)


# --- 训练函数 ---
def train_model(model, train_loader, optimizer, num_epochs=30, device='cpu'):
    """
    训练模型。

    :param model: 待训练的模型。
    :param train_loader: 训练数据加载器。
    :param optimizer: 优化器。
    :param num_epochs: 训练轮数。
    :param device: 指定运行设备 (例如 'cpu' 或 'cuda:0')
    :return: 训练后的模型。
    """
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y, batch_s_safe in train_loader:
            # 将数据移动到指定设备
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_s_safe = batch_s_safe.to(device)

            optimizer.zero_grad()  # 梯度清零

            # 模型前向传播，y_lstm是最终输出
            y_lstm, y_idm, alpha_fixed = model(batch_x, batch_s_safe)

            # 由于最终输出是y_lstm，损失函数直接计算y_lstm与真实值batch_y之间的MSE
            loss = 0.7 * (y_lstm - batch_y).pow(2).mean() + (1 - 0.7) * (y_lstm - y_idm).pow(2).mean()

            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            epoch_loss += loss.item()  # 累加批次损失

        # 打印当前轮次的平均损失
        print(f"Epoch {epoch + 1}/{num_epochs}  Loss: {epoch_loss / len(train_loader):.6f}")
    return model


# --- 测试/评估函数 ---
def evaluate_model(model, test_loader, device='cpu'):
    """
    评估模型在测试集上的性能。

    :param model: 待评估的模型。
    :param test_loader: 测试数据加载器。
    :param device: 指定运行设备 (例如 'cpu' 或 'cuda:0')
    """
    model.eval()  # 设置模型为评估模式
    all_pred, all_true = [], []
    with torch.no_grad():  # 在评估阶段，不计算梯度
        for batch_x, batch_y, batch_s_safe in test_loader:
            # 将数据移动到指定设备
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_s_safe = batch_s_safe.to(device)

            # 模型前向传播，y_lstm是最终输出
            y_lstm, y_idm, alpha_fixed = model(batch_x, batch_s_safe)
            # 收集LSTM的预测结果和真实值
            all_pred.append(y_lstm.cpu())  # 将结果移回CPU进行拼接
            all_true.append(batch_y.cpu())  # 将结果移回CPU进行拼接

    y_pred = torch.cat(all_pred)  # 将所有预测结果拼接
    y_true = torch.cat(all_true)  # 将所有真实值拼接

    # 计算评估指标
    mse = nn.MSELoss()(y_pred, y_true).item()  # 均方误差 (MSE)
    rmse = torch.sqrt(torch.tensor(mse)).item()  # 均方根误差 (RMSE)
    mae = torch.mean(torch.abs(y_pred - y_true)).item()  # 平均绝对误差 (MAE)

    print(f"\nTest Results -- MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # 绘制预测速度与真实速度的对比图
    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:100].numpy(), '--o', label='True')  # 绘制前100个样本的真实值
    plt.plot(y_pred[:100].numpy(), '-x', label='Pred')  # 绘制前100个样本的预测值
    plt.title('Predicted vs. True Speed (First 100 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()


# === 修改后的 compute_position_and_spacing_and_save ===
def compute_position_and_spacing_and_save(model,
                                          test_data,
                                          test_real_speed,
                                          raw_data,
                                          label_data,
                                          train_size,
                                          test_s_safe,
                                          dt=0.1,
                                          output_file="predictions_extended.xlsx",
                                          device='cpu'):
    """
    基于模型预测的速度，计算未来一步的自车位置和间距，并与真实值进行比较。
    然后将详细的预测结果保存到Excel文件。

    :param model: 已训练好的 HybridIDMModel。
    :param test_data: PyTorch张量，测试集输入特征，shape=(N_test, seq_len, feat)。
    :param test_real_speed: PyTorch张量，测试集真实速度标签，shape=(N_test,)。
    :param raw_data: PyTorch张量，原始英尺/英尺·秒单位的完整数据集特征，shape=(N_all, seq_len, feat_raw)。
    :param label_data: PyTorch张量，原始英尺单位的完整数据集标签，shape=(N_all, seq_len, feat_lab)。
    :param train_size: 整数，训练集样本数，用于定位 raw_data/label_data 中测试段的起点。
    :param test_s_safe: PyTorch张量，测试集当前安全距离 (m)，shape=(N_test,)。
    :param dt: 浮点数，时间步长 (s)。
    :param output_file: 字符串，输出 Excel 文件路径。
    :param device: 指定运行设备 (例如 'cpu' 或 'cuda:0')
    """
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 在评估阶段，不计算梯度
        # 将输入数据移动到指定设备
        test_data = test_data.to(device)
        test_s_safe = test_s_safe.to(device)

        # 调用 model，由于模型最终输出直接是LSTM的预测，所以只取y_lstm作为pred_speed
        # y_idm 和 alpha_fixed 仍然从模型返回，但在这里不再用于融合
        pred_speed_lstm, _, _ = model(test_data, test_s_safe)
        pred_speed = pred_speed_lstm.cpu().numpy()  # 将LSTM预测结果转为numpy数组

    true_speed = test_real_speed.cpu().numpy()  # 真实速度

    N_test = test_data.size(0)  # 测试集样本数
    # 计算测试集样本在原始完整数据中的索引
    idx = np.arange(train_size, train_size + N_test)

    # 从原始数据中提取当前自车和前车的位置及速度 (英尺单位)
    current_Y_ft = raw_data[idx, -1, 4].numpy()  # 当前自车位置 (ft)
    # test_data[:, -1, 0] 是已经转换为米/秒的自车速度，需要转换回ft/s用于位移计算
    # 注意：这里test_data已经通过.to(device)移到GPU，但在索引切片后要记得.cpu()再.numpy()
    current_speed_ftps = test_data[:, -1, 0].cpu().numpy() / 0.3048  # 当前自车速度 (ft/s)

    # 原始标签数据中的真实未来自车位置和间距
    true_Y_ft = label_data[idx, -1, 3].numpy()  # 真实自车位置 (ft)
    true_spacing_ft = label_data[idx, -1, 1].numpy()  # 真实间距 (ft)
    true_spacing_m = true_spacing_ft * 0.3048  # 真实间距 (m)

    # 将预测速度从米/秒转换为英尺/秒
    pred_speed_ftps = pred_speed / 0.3048

    # 计算自车位移 (使用预测速度和初始速度的平均作为近似)
    disp_ft = ((current_speed_ftps + pred_speed_ftps) / 2) * dt

    # 预测下一时刻的自车位置
    pred_Y_ft = current_Y_ft + disp_ft

    # 将预测的位置和间距从英尺转换为米
    pred_Y_m = pred_Y_ft * 0.3048
    true_Y_m = true_Y_ft * 0.3048  # 真实自车位置 (m)

    # 预测间距 (保持原有逻辑)
    pred_spacing_m = (true_Y_ft - pred_Y_ft) * 0.3048 + true_spacing_m

    # 计算位置和间距的误差指标 (RMSE, MAPE)
    rmse_Y = np.sqrt(np.mean((pred_Y_m - true_Y_m) ** 2))
    # 避免除以零，对true_Y_m取绝对值并设置一个小的阈值
    valid_mask_Y = np.abs(true_Y_m) > 1e-6
    if np.sum(valid_mask_Y) > 0:
        mape_Y = np.mean(np.abs((pred_Y_m[valid_mask_Y] - true_Y_m[valid_mask_Y]) / true_Y_m[valid_mask_Y])) * 100
    else:
        mape_Y = float('nan')  # 如果所有真实位置都接近零，MAPE无意义

    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m) ** 2))
    # 避免除以零，对true_spacing_m取绝对值并设置一个小的阈值
    valid_mask_sp = np.abs(true_spacing_m) > 1e-6
    if np.sum(valid_mask_sp) > 0:
        mape_sp = np.mean(np.abs(
            (pred_spacing_m[valid_mask_sp] - true_spacing_m[valid_mask_sp]) / true_spacing_m[valid_mask_sp])) * 100
    else:
        mape_sp = float('nan')  # 如果所有真实间距都接近零，MAPE无意义

    print(f"Position Error -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y if not np.isnan(mape_Y) else 'N/A'}%")
    print(f" Spacing  Error -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp if not np.isnan(mape_sp) else 'N/A'}%")

    # 保存结果到 Excel 文件
    df = pd.DataFrame({
        "Pred Speed (m/s)": pred_speed,
        "True Speed (m/s)": true_speed,
        "Predicted Y (m)": pred_Y_m,
        "True Y (m)": true_Y_m,
        "Pred Spacing (m)": pred_spacing_m,
        "True Spacing (m)": true_spacing_m,
    })
    sheet_name = "PID-LSTM-IDM"  # Excel工作表名称
    # 根据文件是否存在选择写入模式（追加或新建）
    mode = "a" if os.path.exists(output_file) else "w"
    with pd.ExcelWriter(output_file, engine="openpyxl", mode=mode) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Results saved to '{output_file}' sheet '{sheet_name}'.")


# --- 主流程 ---
if __name__ == "__main__":
    torch.manual_seed(42)  # 设置随机种子以保证结果可复现

    # --- 1. 设备配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. 加载 MAT 数据 ---
    data_file_path = 'E:\pythonProject1\data_fine_0.1.mat'
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
        exit()  # 如果文件不存在，程序退出

    data = sio.loadmat(data_file_path)
    raw = torch.tensor(data['train_data'], dtype=torch.float32)
    lab = torch.tensor(data['lable_data'], dtype=torch.float32)

    # --- 3. 数据预处理与特征选择 ---
    seq = raw[:, -50:, [0, 1, 2, 3, -1]].clone()
    y = lab[:, -1, 0].clone().squeeze()
    s_safe = seq[:, -1, 1]

    # --- 4. 单位转换: 英尺/秒 (ft/s) 到 米/秒 (m/s) ---
    seq *= 0.3048
    y *= 0.3048
    s_safe *= 0.3048

    # --- 5. 数据抽样 (只用前 10% 加速示例) ---
    N = int(seq.size(0) * 0.1)
    seq, y, s_safe = seq[:N], y[:N], s_safe[:N]
    print(f"Using {N} data samples for processing (first 10% of total {seq.size(0)}).")

    # --- 6. 数据集划分与加载器准备 ---
    split_ratio = 0.8
    train_size = int(N * split_ratio)

    train_seq = seq[:train_size]
    test_seq = seq[train_size:]
    train_y = y[:train_size]
    test_y = y[train_size:]
    train_s_safe = s_safe[:train_size]
    test_s_safe = s_safe[train_size:]

    train_ds = torch.utils.data.TensorDataset(train_seq, train_y, train_s_safe)
    test_ds = torch.utils.data.TensorDataset(test_seq, test_y, test_s_safe)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    # --- 7. 模型实例化和权重初始化 ---
    input_dim = seq.size(2)
    hidden_dim = 128
    model = HybridIDMModel(input_dim, hidden_dim, num_layers=1).to(device)  # 将模型移到指定设备
    initialize_weights(model)

    # --- 8. 优化器设定 ---
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # --- 9. 训练 & 评估模型 ---
    print("\n--- Starting Model Training ---")
    model = train_model(model, train_loader, optimizer, num_epochs=100, device=device)  # 传递 device
    print("\n--- Starting Model Evaluation ---")
    evaluate_model(model, test_loader, device=device)  # 传递 device

    # --- 10. 计算并保存位置和间距预测结果 ---
    print("\n--- Computing and Saving Position/Spacing Predictions ---")
    compute_position_and_spacing_and_save(
        model,
        test_seq,
        test_y,
        raw,
        lab,
        train_size,
        test_s_safe,
        dt=0.1,
        output_file="predictions_extended.xlsx",
        device=device  # 传递 device
    )

    print("\nAll operations completed.")

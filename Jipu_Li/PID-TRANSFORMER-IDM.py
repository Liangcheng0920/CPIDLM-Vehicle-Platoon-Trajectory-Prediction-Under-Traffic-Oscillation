import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import glob  # 用于查找文件路径
import os  # 操作系统接口，用于路径操作和环境变量
import math  # Transformer Positional Encoding 需要

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许多个OpenMP库存在，避免某些环境下的冲突

# --- 全局路径定义 ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # 数据集存放目录 (请根据您的实际路径修改)
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim_modified_transformer_single_step"  # 实验结果保存目录 (修改了目录名以区分)

# 确保结果目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 全局常量 ---
DT = 0.1  # 时间步长 (s) - 从原代码推断
PRED_HORIZON = 1  # 预测步长 K - 修改为单步预测


# --- 数据检查函数 ---
def check_data(data, name="data"):
    """ 检查数据中是否包含NaN或Inf值 """
    print(f"正在检查 {name} 是否包含 NaN 或 Inf 值...")
    print(f"包含 NaN: {torch.isnan(data).any().item()}")
    print(f"包含 Inf: {torch.isinf(data).any().item()}")


# --- 固定 IDM 参数预测函数 ---
def idm_fixed(v_n, s_safe, delta_v,
              v_desired=10.13701546, T=0.50284384, a_max=0.10995557,
              b_safe=4.98369406, delta=5.35419582, s0=0.10337701,
              delta_t=DT):
    """
    使用固定的参数集执行一步IDM（智能驾驶员模型）预测。
    :param v_n: 当前自车速度 (m/s)
    :param s_safe: 当前实际间距 (m)
    :param delta_v: 当前速度差 (前车速度 - 自车速度, m/s)
    :param v_desired:期望速度 (m/s)
    :param T: 安全时间余量 (s)
    :param a_max: 最大加速度 (m/s^2)
    :param b_safe: 舒适减速度 (m/s^2)
    :param delta: 加速度指数
    :param s0: 最小静止间距 (m)
    :param delta_t: 时间步长 (s)
    :return: 下一时间步的预测自车速度 (m/s)
    """
    current_device = v_n.device  # 从输入张量v_n推断设备
    # 将IDM参数转换为与输入数据相同设备和类型的张量
    v_desired = torch.tensor(v_desired, device=current_device, dtype=v_n.dtype)
    T = torch.tensor(T, device=current_device, dtype=v_n.dtype)
    a_max = torch.tensor(a_max, device=current_device, dtype=v_n.dtype).clamp(min=1e-6)  # 避免除以零
    b_safe = torch.tensor(b_safe, device=current_device, dtype=v_n.dtype).clamp(min=1e-6)  # 避免除以零
    s0 = torch.tensor(s0, device=current_device, dtype=v_n.dtype)
    delta_param = torch.tensor(delta, device=current_device, dtype=v_n.dtype)  # 'delta'是IDM中的指数参数
    delta_t_tensor = torch.tensor(delta_t, device=current_device, dtype=v_n.dtype)

    s_safe = s_safe.clamp(min=1e-6)  # 确保安全间距为正

    s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe) + 1e-6)
    s_star = s_star.clamp(min=0.0)  # 期望间距不能为负

    v_n_ratio = torch.zeros_like(v_n)
    mask_v_desired_nonzero = v_desired.abs() > 1e-6
    if mask_v_desired_nonzero.any():
        v_n_ratio[mask_v_desired_nonzero] = (v_n[mask_v_desired_nonzero] / v_desired[mask_v_desired_nonzero])

    acceleration_term = a_max * (
            1 - v_n_ratio ** delta_param - (s_star / s_safe) ** 2
    )
    v_follow = v_n + delta_t_tensor * acceleration_term
    return v_follow.clamp(min=0.0)  # 速度不能为负


# --- Transformer 位置编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):  # d_model是模型的维度 (embedding_dim), max_len是序列最大长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) -> (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: 输入张量，形状为 (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- 定义新的基于 Transformer 的融合模型 (单步预测) ---
class HybridIDMTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,  # 基本参数
                 nhead=4, transformer_num_layers=2, dim_feedforward=512, dropout_transformer=0.1):  # Transformer特定参数
        super(HybridIDMTransformerModel, self).__init__()
        self.pred_horizon = PRED_HORIZON  # 预测步长 K, 固定为1
        self.model_dim = hidden_dim  # Transformer的内部维度 (d_model)

        self.input_fc = nn.Linear(input_dim, self.model_dim)
        self.pos_encoder = PositionalEncoding(self.model_dim, dropout_transformer)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_transformer,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=transformer_num_layers)
        # 输出线性层：将Transformer最后一个时间步的输出映射到1个预测值 (单步)
        self.fc = nn.Linear(self.model_dim, self.pred_horizon)  # output_dim is 1

        # IDM参数 (固定值)
        self.v_desired_idm = 12.64798288
        self.T_idm = 0.50284384
        self.a_max_idm = 0.10033688
        self.b_safe_idm = 4.98937183
        self.delta_idm = 1.0
        self.s0_idm = 0.13082412

    def forward(self, x, s_safe_initial, v_lead_initial):
        """
        模型的前向传播 (单步预测)。
        :param x: 输入序列, shape=(batch, seq_len, input_dim)
        :param s_safe_initial: 当前安全距离 (真实观测值，用于IDM), shape=(batch,)
        :param v_lead_initial: 当前前车速度 (真实观测值，用于IDM), shape=(batch,)
        :return:
          y_nn_pred: 基于Transformer的单步速度预测, shape=(batch, 1)
          y_idm_pred: 基于IDM的单步速度预测, shape=(batch, 1)
        """
        # Transformer 单步预测
        x_transformed = self.input_fc(x)
        x_transformed = self.pos_encoder(x_transformed)
        transformer_out = self.transformer_encoder(x_transformed)

        # 取Transformer最后一个时间步的输出进行预测
        y_nn_pred = self.fc(transformer_out[:, -1, :])  # -> (batch, 1)

        # IDM 单步预测
        y_idm_pred_list = []
        v_ego_current_idm = x[:, -1, 0].clone()  # 自车当前速度
        s_current_idm = s_safe_initial.clone()  # 当前间距
        v_lead_constant_idm = v_lead_initial.clone()  # 前车速度

        # 由于 PRED_HORIZON = 1, 这个循环只执行一次
        for _ in range(self.pred_horizon):
            delta_v_idm = v_lead_constant_idm - v_ego_current_idm
            v_ego_next_pred_idm = idm_fixed(
                v_ego_current_idm, s_current_idm, delta_v_idm,
                v_desired=self.v_desired_idm, T=self.T_idm, a_max=self.a_max_idm,
                b_safe=self.b_safe_idm, delta=self.delta_idm, s0=self.s0_idm, delta_t=DT
            )
            y_idm_pred_list.append(v_ego_next_pred_idm.unsqueeze(1))
            # 对于单步预测，不需要更新 s_current_idm 和 v_ego_current_idm 进行下一步迭代

        y_idm_pred = torch.cat(y_idm_pred_list, dim=1)  # -> (batch, 1)

        return y_nn_pred, y_idm_pred


def initialize_weights(model):
    """ 初始化模型权重 """
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)


# --- 训练函数 (单步预测版) ---
def train_model(model, train_loader, device, num_epochs=30, alpha_decay_loss=0.1, lr_nn=5e-4):
    model.train()
    # 对于单步预测 PRED_HORIZON=1, loss_weights 实际上是 tensor([1.0])
    loss_weights_device = next(model.parameters()).device
    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(PRED_HORIZON, dtype=torch.float32)).to(
        loss_weights_device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * PRED_HORIZON

    nn_params = [param for name, param in model.named_parameters()]
    optimizer_nn = optim.Adam(nn_params, lr=lr_nn)

    alpha_fixed = 0.7  # Alpha固定为0.7

    print(f"--- 使用固定 Alpha 开始训练 (单步预测, 设备: {device}) ---")
    print(f"神经网络参数将根据 L_nn = alpha * L_true + (1-alpha) * L_idm 进行优化。")
    print(f"Alpha 固定为: {alpha_fixed}")
    print(f"------------------------------------")

    for epoch in range(num_epochs):
        epoch_loss_nn_objective = 0.0

        for batch_x, batch_y_target, batch_s_safe_initial, batch_v_lead_initial in train_loader:
            batch_x = batch_x.to(device)
            batch_y_target = batch_y_target.to(device)  # batch_y_target shape: (batch, 1)
            batch_s_safe_initial = batch_s_safe_initial.to(device)
            batch_v_lead_initial = batch_v_lead_initial.to(device)

            optimizer_nn.zero_grad()

            y_nn_pred, y_idm_pred = model(batch_x, batch_s_safe_initial,
                                          batch_v_lead_initial)  # y_nn_pred, y_idm_pred shape: (batch, 1)

            # 损失1: 神经网络预测与真实值之间的差异
            loss_nn_vs_true = ((y_nn_pred - batch_y_target).pow(2) * loss_weights.unsqueeze(0)).mean()
            # 损失2: 神经网络预测与IDM预测之间的差异
            loss_nn_vs_idm = ((y_nn_pred - y_idm_pred.detach()).pow(2) * loss_weights.unsqueeze(0)).mean()

            loss_for_nn_params = alpha_fixed * loss_nn_vs_true + \
                                 (1 - alpha_fixed) * loss_nn_vs_idm

            loss_for_nn_params.backward()
            optimizer_nn.step()

            epoch_loss_nn_objective += loss_for_nn_params.item()

        avg_nn_loss = epoch_loss_nn_objective / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}  神经网络目标损失: {avg_nn_loss:.6f}  (α={alpha_fixed})")
    return model


# --- 测试/评估函数 (单步预测版) ---
def evaluate_model(model, test_loader, device, alpha_decay_loss=0.1, dataset_name="", results_dir=""):
    model.eval()
    all_pred_nn, all_pred_idm, all_true = [], [], []

    loss_weights_device = next(model.parameters()).device
    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(PRED_HORIZON, dtype=torch.float32)).to(
        loss_weights_device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * PRED_HORIZON  # Will be tensor([1.0])

    total_mse_nn_vs_true_weighted = 0
    total_mse_idm_vs_true_weighted = 0

    with torch.no_grad():
        for batch_x, batch_y_target, batch_s_safe_initial, batch_v_lead_initial in test_loader:
            batch_x = batch_x.to(device)
            batch_y_target = batch_y_target.to(device)  # shape (batch, 1)
            batch_s_safe_initial = batch_s_safe_initial.to(device)
            batch_v_lead_initial = batch_v_lead_initial.to(device)

            y_nn_pred, y_idm_pred = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)  # shape (batch, 1)

            all_pred_nn.append(y_nn_pred.cpu())
            all_pred_idm.append(y_idm_pred.cpu())
            all_true.append(batch_y_target.cpu())

            loss_nn_vs_true_batch_weighted = ((y_nn_pred - batch_y_target).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_nn_vs_true_weighted += loss_nn_vs_true_batch_weighted.item() * batch_x.size(0)

            loss_idm_vs_true_batch_weighted = ((y_idm_pred - batch_y_target).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_idm_vs_true_weighted += loss_idm_vs_true_batch_weighted.item() * batch_x.size(0)

    num_samples = len(test_loader.dataset)
    avg_mse_nn_vs_true_weighted = total_mse_nn_vs_true_weighted / num_samples
    avg_mse_idm_vs_true_weighted = total_mse_idm_vs_true_weighted / num_samples

    fixed_alpha_for_metrics = 0.7

    y_pred_nn_cat = torch.cat(all_pred_nn)  # shape (num_samples, 1)
    y_pred_idm_cat = torch.cat(all_pred_idm)  # shape (num_samples, 1)
    y_true_cat = torch.cat(all_true)  # shape (num_samples, 1)

    y_final_prediction_cat = y_pred_nn_cat  # 最终预测使用神经网络输出

    # 计算总体评估指标 (针对单步预测)
    mse_val_overall = torch.mean((y_final_prediction_cat - y_true_cat).pow(2)).item()
    rmse_val_overall = np.sqrt(mse_val_overall)
    mae_val_overall = torch.mean(torch.abs(y_final_prediction_cat - y_true_cat)).item()

    abs_error_overall = torch.abs(y_final_prediction_cat - y_true_cat)
    abs_true_overall = torch.abs(y_true_cat)
    valid_mape_mask_overall = abs_true_overall > 1e-6
    mape_p_overall = float('nan')
    if torch.sum(valid_mape_mask_overall) > 0:
        mape_p_overall = torch.mean(
            abs_error_overall[valid_mape_mask_overall] / abs_true_overall[valid_mape_mask_overall]
        ).item() * 100

    print(f"\n--- 测试结果总结 (单步预测, 最终使用神经网络预测) ---")
    print(
        f"  神经网络预测 vs 真实 -- MSE: {mse_val_overall:.4f}, RMSE: {rmse_val_overall:.4f}, MAE: {mae_val_overall:.4f}, MAPE: {mape_p_overall if not np.isnan(mape_p_overall) else 'N/A'}%")
    print(
        f"  (参考: IDM预测 vs 真实 MSE: {avg_mse_idm_vs_true_weighted:.4f})")  # MSE is weighted, but for single step it's same as unweighted
    print(f"  (参考: 神经网络预测 vs 真实 MSE (训练指标): {avg_mse_nn_vs_true_weighted:.4f})")
    print(f"  Alpha 使用值 (固定)={fixed_alpha_for_metrics:.4f}")

    # 对于单步预测，详细指标与总体指标相同
    print(f"\n--- 单步预测详细指标 (步骤 1) ---")
    y_pred_nn_step_1 = y_pred_nn_cat[:, 0]
    y_pred_idm_step_1 = y_pred_idm_cat[:, 0]
    y_true_step_1 = y_true_cat[:, 0]

    mse_step_nn = nn.MSELoss()(y_pred_nn_step_1, y_true_step_1).item()
    rmse_step_nn = np.sqrt(mse_step_nn)
    mae_step_nn = torch.mean(torch.abs(y_pred_nn_step_1 - y_true_step_1)).item()

    abs_error_step = torch.abs(y_pred_nn_step_1 - y_true_step_1)
    abs_true_step = torch.abs(y_true_step_1)
    valid_mape_mask_step = abs_true_step > 1e-6
    mape_step_nn = float('nan')
    if torch.sum(valid_mape_mask_step) > 0:
        mape_step_nn = torch.mean(
            abs_error_step[valid_mape_mask_step] / abs_true_step[valid_mape_mask_step]
        ).item() * 100
    mse_step_idm = nn.MSELoss()(y_pred_idm_step_1, y_true_step_1).item()

    print(f"  步骤 1:")
    print(
        f"    神经网络预测 -- MSE: {mse_step_nn:.4f}, RMSE: {rmse_step_nn:.4f}, MAE: {mae_step_nn:.4f}, MAPE: {mape_step_nn if not np.isnan(mape_step_nn) else 'N/A'}%")
    print(f"    IDM (参考) -- MSE: {mse_step_idm:.4f}")

    # 绘制单步预测的对比图
    plt.figure(figsize=(12, 7))
    plt.plot(y_true_cat[:100, 0].numpy(), '--o', label=f'真实值 (步骤 1)')
    plt.plot(y_pred_nn_cat[:100, 0].numpy(), '-x', label=f'神经网络预测 (步骤 1) (最终使用)')
    plt.plot(y_pred_idm_cat[:100, 0].numpy(), '-s', label=f'IDM预测 (步骤 1) (参考)')

    y_pred_combined_for_plot_cat = fixed_alpha_for_metrics * y_pred_nn_cat + \
                                   (1 - fixed_alpha_for_metrics) * y_pred_idm_cat
    plt.plot(y_pred_combined_for_plot_cat[:100, 0].numpy(), '-.',
             label=f'假设融合 (步骤 1, α={fixed_alpha_for_metrics:.2f}) (图形参考)')

    plt.title(f'速度预测对比 (前100样本, 步骤 1) ({dataset_name})')
    plt.xlabel("样本索引")
    plt.ylabel("速度 (m/s)")
    plt.legend()
    plt.grid()
    plot_filename = os.path.join(results_dir, f"{dataset_name}_speed_comparison_PITRANSFORMER_IDM_single_step.png")
    plt.savefig(plot_filename)
    print(f"速度对比图已保存至 {plot_filename}")
    plt.close()

    return avg_mse_nn_vs_true_weighted, np.sqrt(avg_mse_nn_vs_true_weighted), mae_val_overall, mape_p_overall


# === 修改后的 compute_position_and_spacing_and_save (单步预测版) ===
def compute_position_and_spacing_and_save(model,
                                          test_loader,
                                          raw_data_all,  # 原始数据集 (用于获取初始状态)
                                          label_data_all,  # 标签数据集 (用于获取真实未来位置/间距)
                                          train_size,  # 训练集大小
                                          device,  # 设备
                                          dt=0.1,  # 时间步长
                                          output_file="predictions_singlestep_extended.xlsx",
                                          dataset_name=""):
    model.eval()
    test_start_idx_in_all_data = train_size

    y_nn_list_mps, y_true_speeds_list_mps = [], []
    initial_ego_pos_ft_collected = []
    initial_lead_pos_ft_collected = []
    initial_ego_speed_ftps_collected = []
    initial_lead_speed_ftps_collected = []
    true_future_ego_pos_ft_collected = []  # 真实下一时刻自车位置
    true_future_spacing_ft_collected = []  # 真实下一时刻间距

    with torch.no_grad():
        for i, (batch_x_mps, batch_y_target_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps) in enumerate(
                test_loader):
            batch_x_mps = batch_x_mps.to(device)
            batch_s_safe_initial_m = batch_s_safe_initial_m.to(device)
            batch_v_lead_initial_mps = batch_v_lead_initial_mps.to(device)

            y_nn_pred_mps, _ = model(batch_x_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps)  # shape (batch, 1)

            y_nn_list_mps.append(y_nn_pred_mps.cpu())
            y_true_speeds_list_mps.append(batch_y_target_mps.cpu())  # batch_y_target_mps shape (batch, 1)

            current_batch_indices_in_all_data = np.arange(
                test_start_idx_in_all_data + i * test_loader.batch_size,
                test_start_idx_in_all_data + i * test_loader.batch_size + batch_x_mps.size(0)
            )
            # 提取初始状态 (英尺单位)
            initial_ego_pos_ft_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 4].cpu())  # 自车位置Y
            initial_lead_pos_ft_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 4].cpu())  # 前车位置Y
            initial_ego_speed_ftps_collected.append(
                raw_data_all[current_batch_indices_in_all_data, -1, 0].cpu())  # 自车速度
            initial_lead_speed_ftps_collected.append(
                raw_data_all[current_batch_indices_in_all_data, -1, 5].cpu())  # 前车速度

            # 提取真实的下一时刻位置和间距 (英尺单位)
            # label_data_all[:, 0, col_idx] 表示取未来第1个时间步的特定标签
            true_future_ego_pos_ft_collected.append(
                label_data_all[current_batch_indices_in_all_data, 0, 3].cpu().unsqueeze(-1))  # 自车未来位置Y, shape (batch,1)
            true_future_spacing_ft_collected.append(
                label_data_all[current_batch_indices_in_all_data, 0, 1].cpu().unsqueeze(-1))  # 未来间距, shape (batch,1)

    y_nn_all_mps = torch.cat(y_nn_list_mps, dim=0)  # (num_test_samples, 1)
    y_true_speeds_all_mps = torch.cat(y_true_speeds_list_mps, dim=0)  # (num_test_samples, 1)

    final_pred_speeds_mps = y_nn_all_mps  # (num_test_samples, 1)

    initial_ego_pos_ft = torch.cat(initial_ego_pos_ft_collected, dim=0)  # (num_test_samples,)
    initial_lead_pos_ft = torch.cat(initial_lead_pos_ft_collected, dim=0)  # (num_test_samples,)
    initial_ego_speed_ftps = torch.cat(initial_ego_speed_ftps_collected, dim=0)  # (num_test_samples,)
    initial_lead_speed_ftps = torch.cat(initial_lead_speed_ftps_collected, dim=0)  # (num_test_samples,)

    true_future_ego_pos_ft = torch.cat(true_future_ego_pos_ft_collected, dim=0)  # (num_test_samples, 1)
    true_future_spacing_ft = torch.cat(true_future_spacing_ft_collected, dim=0)  # (num_test_samples, 1)

    # 初始化用于存储单步预测位置和间距的张量 (单位: ft)
    # Shape will be (num_test_samples, 1) as PRED_HORIZON is 1
    pred_ego_pos_next_step_ft = torch.zeros_like(final_pred_speeds_mps)
    pred_lead_pos_next_step_ft = torch.zeros_like(final_pred_speeds_mps)
    pred_spacing_next_step_ft = torch.zeros_like(final_pred_speeds_mps)

    final_pred_speeds_ftps = final_pred_speeds_mps / 0.3048  # (num_test_samples, 1)

    # 当前迭代的自车和前车位置 (ft)，从初始观测值开始
    current_ego_pos_ft = initial_ego_pos_ft.clone()  # (num_test_samples,)
    current_lead_pos_ft = initial_lead_pos_ft.clone()  # (num_test_samples,)
    lead_speed_constant_ftps = initial_lead_speed_ftps  # (num_test_samples,)

    # 计算下一时刻的位置和间距 (由于PRED_HORIZON=1, 循环只执行一次 for k=0)
    # k=0 : 预测 t+1 时刻的状态
    # 使用 *初始观测* 的自车速度 (t时刻) 来计算 t 到 t+1 的位移
    speed_ego_this_step_ftps = initial_ego_speed_ftps  # (num_test_samples,)

    disp_ego_ft = speed_ego_this_step_ftps * dt
    disp_lead_ft = lead_speed_constant_ftps * dt

    # 下一时刻的预测位置
    pred_ego_pos_next_step_ft[:, 0] = current_ego_pos_ft + disp_ego_ft
    pred_lead_pos_next_step_ft[:, 0] = current_lead_pos_ft + disp_lead_ft
    pred_spacing_next_step_ft[:, 0] = pred_lead_pos_next_step_ft[:, 0] - pred_ego_pos_next_step_ft[:, 0]

    pred_ego_pos_m = pred_ego_pos_next_step_ft.numpy() * 0.3048  # (num_test_samples, 1)
    true_ego_pos_m = true_future_ego_pos_ft.numpy() * 0.3048  # (num_test_samples, 1)
    pred_spacing_m = pred_spacing_next_step_ft.numpy() * 0.3048  # (num_test_samples, 1)
    true_spacing_m = true_future_spacing_ft.numpy() * 0.3048  # (num_test_samples, 1)

    print(f"\n--- 基于神经网络单步速度预测的位置和间距误差评估 (步骤 1) ---")
    # k_s = 0 for single step
    pos_err_sq_step = (pred_ego_pos_m[:, 0] - true_ego_pos_m[:, 0]) ** 2
    rmse_Y_step = np.sqrt(np.mean(pos_err_sq_step))
    valid_true_Y_step_mask = np.abs(true_ego_pos_m[:, 0]) > 1e-6
    mape_Y_step = float('nan')
    if np.sum(valid_true_Y_step_mask) > 0:
        mape_Y_step = np.mean(np.abs(
            (pred_ego_pos_m[valid_true_Y_step_mask, 0] - true_ego_pos_m[valid_true_Y_step_mask, 0]) /
            true_ego_pos_m[valid_true_Y_step_mask, 0])) * 100

    spacing_err_sq_step = (pred_spacing_m[:, 0] - true_spacing_m[:, 0]) ** 2
    rmse_sp_step = np.sqrt(np.mean(spacing_err_sq_step))
    valid_true_sp_step_mask = np.abs(true_spacing_m[:, 0]) > 1e-6
    mape_sp_step = float('nan')
    if np.sum(valid_true_sp_step_mask) > 0:
        mape_sp_step = np.mean(np.abs(
            (pred_spacing_m[valid_true_sp_step_mask, 0] - true_spacing_m[valid_true_sp_step_mask, 0]) /
            true_spacing_m[valid_true_sp_step_mask, 0])) * 100

    print(f"  步骤 1:")
    print(f"    位置误差 -- RMSE: {rmse_Y_step:.4f} m, MAPE: {mape_Y_step if not np.isnan(mape_Y_step) else 'N/A'}%")
    print(f"    间距误差 -- RMSE: {rmse_sp_step:.4f} m, MAPE: {mape_sp_step if not np.isnan(mape_sp_step) else 'N/A'}%")

    # 对于单步预测, "last_step" is the first and only step.
    rmse_p_last_step = rmse_Y_step
    mape_p_last_step = mape_Y_step

    df_data = {}
    df_data[f"神经网络预测速度 (m/s) 步骤 1"] = final_pred_speeds_mps[:, 0].numpy()
    df_data[f"真实速度 (m/s) 步骤 1"] = y_true_speeds_all_mps[:, 0].numpy()
    df_data[f"预测自车位置 Y (m) 步骤 1"] = pred_ego_pos_m[:, 0]
    df_data[f"真实自车位置 Y (m) 步骤 1"] = true_ego_pos_m[:, 0]
    df_data[f"预测间距 (m) 步骤 1"] = pred_spacing_m[:, 0]
    df_data[f"真实间距 (m) 步骤 1"] = true_spacing_m[:, 0]
    df_pos = pd.DataFrame(df_data)

    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            df_pos.to_excel(writer, sheet_name=dataset_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df_pos.to_excel(writer, sheet_name=dataset_name, index=False)

    print(f"{dataset_name} 的单步位置和间距预测已保存到 '{output_file}' 的 sheet '{dataset_name}'.")
    return rmse_p_last_step, mape_p_last_step


# # --- 存储和保存评估指标的辅助函数 ---
# all_datasets_metrics_summary = []
#
#
# def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse_next_step,
#                           pos_mape_next_step):
#     """存储单个数据集的评估指标 (单步)"""
#     metrics = {
#         "数据集 (Dataset)": dataset_name,
#         "速度MSE_NN (Speed_MSE_NN_summary)": speed_mse,
#         "速度RMSE_NN (Speed_RMSE_NN_summary)": speed_rmse,
#         "速度MAE_NN (Speed_MAE_NN_overall)": speed_mae,
#         "速度MAPE_NN (%) (Speed_MAPE_NN_overall_percent)": speed_mape,
#         "下一步位置RMSE_NN (m) (Position_RMSE_NN_next_step_m)": pos_rmse_next_step,
#         "下一步位置MAPE_NN (%) (Position_MAPE_NN_next_step_percent)": pos_mape_next_step
#     }
#     all_datasets_metrics_summary.append(metrics)
#
#
# def save_all_metrics_to_csv(filepath="evaluation_summary_PItransform_idm_single_step.csv"):
#     """将所有数据集的评估指标汇总保存到CSV文件"""
#     if not all_datasets_metrics_summary:
#         print("没有评估指标可以保存。")
#         return
#     df_metrics = pd.DataFrame(all_datasets_metrics_summary)
#     df_metrics.to_csv(filepath, index=False, encoding='utf-8-sig')
#     print(f"所有数据集的评估指标汇总已保存至 {filepath}")


# --- 主流程 ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    if not data_files:
        print(f"在目录 {DATA_DIR} 中未找到 .mat 文件。程序将退出。")
        exit()
    print(f"找到以下数据集文件: {data_files}")

    position_predictions_excel_path = os.path.join(RESULTS_DIR,
                                                   "pred_positions_all_datasets_pitransformer_idm_single_step1128.xlsx")
    LR_NN_PARAMS = 5e-4
    data = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')
    raw_all_ft = torch.tensor(data['train_data'], dtype=torch.float32)  # (samples, seq_len, features)
    lab_all_ft = torch.tensor(data['lable_data'], dtype=torch.float32)  # (samples, future_steps, label_features)

    # 特征提取
    # raw_all_ft 包含: 0:自车速度, 1:间距, 2:速度差, 3:前车加速度, 4:自车位置Y, 5:前车速度, 6:自车加速度, 7:前车位置Y
    # 我们选择的特征: 自车速度(0), 间距(1), 速度差(2), 自车位置Y(3), 前车速度(5) -> 实际使用时注意对齐
    # 当前代码中 seq_ft 使用的索引是 [0, 1, 2, 3, 5] 对应原始特征，但这些特征的含义需要明确
    # 假设 train_data 的列: [ego_v, spacing, delta_v, lead_acc, ego_pos_Y, lead_v, ego_acc, lead_pos_Y]
    # seq_ft = raw_all_ft[:, :, [0, 1, 2, 3, 5]].clone() # 之前代码的特征选择
    # 假设模型输入特征: 自车速度, 间距, 速度差, 自车加速度, 前车加速度 (与原始LSTM代码保持一致)
    # 原始 LSTM 代码的特征: v_ego, s_safe, delta_v, a_ego, a_lead
    # train_data[:,:,0] -> v_ego (自车速度)
    # train_data[:,:,1] -> s_safe (安全间距)
    # train_data[:,:,2] -> delta_v (速度差)
    # train_data[:,:,6] -> a_ego (自车加速度)
    # train_data[:,:,3] -> a_lead (前车加速度)
    # 因此，输入特征的索引应为 [0, 1, 2, 6, 3]
    seq_ft = raw_all_ft[:, :, [0, 1, 2, 3, 5]].clone()  # (samples, seq_len, 5 input features)

    # 目标变量: 下一时刻的自车速度 (lable_v_NextTime)
    # lab_all_ft[:, 0, 0] -> 取第一个预测步 (index 0), 第一个特征 (index 0, 即速度)
    y_target_ftps = lab_all_ft[:, 0, 0].unsqueeze(-1).clone()  # (samples, 1)


    # 初始安全距离 s_safe_initial_ft 和前车速度 v_lead_initial_ftps
    # 从输入序列的最后一个时间步获取
    # s_safe_initial_ft: seq_ft[:, -1, 1] (间距是第二个特征)
    # v_lead_initial_ftps: raw_all_ft[:, -1, 5] (前车速度是原始数据中的第6个特征)
    s_safe_initial_ft = seq_ft[:, -1, 1].clone()  # (samples,)
    v_lead_initial_ftps = raw_all_ft[:, -1, 5].clone()  # (samples,)

    # 单位转换: 英尺/英尺每秒 -> 米/米每秒
    seq_mps = seq_ft.clone()
    # 速度特征索引: 0 (自车速)
    # 加速度特征索引: 3 (自车加), 4 (前车加)
    # 距离特征索引: 1 (间距)
    # 速度差索引: 2
    # 假设: 速度和加速度特征需要转换单位，间距也需要
    seq_mps[:, :, [0, 2, 3, 4]] *= 0.3048  # v_ego, delta_v, a_ego, a_lead
    seq_mps[:, :, 1] *= 0.3048  # s_safe

    y_target_mps = y_target_ftps * 0.3048  # (samples, 1)
    s_safe_initial_m = s_safe_initial_ft * 0.3048  # (samples,)
    v_lead_initial_mps = v_lead_initial_ftps * 0.3048  # (samples,)

    N_total = seq_mps.size(0)
    N = int(N_total * 0.1)  # 使用全部数据
    print(f"将使用 {N} / {N_total} 条数据进行训练和测试。")

    seq_mps_selected = seq_mps[:N]
    y_target_mps_selected = y_target_mps[:N]
    s_safe_initial_m_selected = s_safe_initial_m[:N]
    v_lead_initial_mps_selected = v_lead_initial_mps[:N]
    raw_all_ft_selected = raw_all_ft[:N]  # 用于后续位置计算
    lab_all_ft_selected = lab_all_ft[:N]  # 用于后续位置计算

    split_ratio = 0.8
    train_size = int(N * split_ratio)

    train_seq = seq_mps_selected[:train_size]
    test_seq = seq_mps_selected[train_size:]
    train_y_target = y_target_mps_selected[:train_size]
    test_y_target = y_target_mps_selected[train_size:]
    train_s_safe_initial = s_safe_initial_m_selected[:train_size]
    test_s_safe_initial = s_safe_initial_m_selected[train_size:]
    train_v_lead_initial = v_lead_initial_mps_selected[:train_size]
    test_v_lead_initial = v_lead_initial_mps_selected[train_size:]

    # 不需要移动到device在这里，DataLoader会处理
    batch_size = 32
    train_ds = torch.utils.data.TensorDataset(train_seq, train_y_target, train_s_safe_initial, train_v_lead_initial)
    test_ds = torch.utils.data.TensorDataset(test_seq, test_y_target, test_s_safe_initial, test_v_lead_initial)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = train_seq.size(2)  # Should be 5
    hidden_dim = 128
    n_head = 4
    transformer_layers = 2
    feedforward_dim = 512
    transformer_dropout = 0.1

    model = HybridIDMTransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        # output_dim is implicitly PRED_HORIZON inside the model class
        nhead=n_head,
        transformer_num_layers=transformer_layers,
        dim_feedforward=feedforward_dim,
        dropout_transformer=transformer_dropout
    ).to(device)

    initialize_weights(model)


    model = train_model(model, train_loader, device=device, num_epochs=50,  # num_epochs=100
                        alpha_decay_loss=0.05, lr_nn=LR_NN_PARAMS)


    speed_mse_summary, speed_rmse_summary, speed_mae_overall, speed_mape_overall = evaluate_model(
        model, test_loader, device=device, alpha_decay_loss=0.05,
        dataset_name='data1', results_dir=RESULTS_DIR
    )

    print(f"开始计算和评估位置/间距预测: {'data1'}...")
    pos_rmse_next_step, pos_mape_next_step = compute_position_and_spacing_and_save(
        model, test_loader, raw_all_ft_selected, lab_all_ft_selected, train_size,
        device=device, dt=DT,
        output_file=position_predictions_excel_path, dataset_name='data1'
    )




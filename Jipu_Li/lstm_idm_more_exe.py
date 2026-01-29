import os  # 导入os模块，用于与操作系统交互，例如文件路径操作
import torch  # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import scipy.io as sio  # 导入scipy.io模块，用于加载.mat格式的数据文件 (MATLAB格式)
import pandas as pd  # 导入pandas库，用于数据处理和分析
import numpy as np  # 导入numpy库，用于数值计算
import glob  # 用于查找文件路径

# 设置环境变量KMP_DUPLICATE_LIB_OK为TRUE，
# 这通常是为了解决某些环境下Intel MKL库（常用于加速数学运算）可能与PyTorch自带的库冲突的问题，
# 允许加载重复的动态链接库。
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --- 全局路径定义 ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # 数据集存放目录
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim_lstm_idm_only"  # 实验结果保存目录 (修改目录名以区分)

# 确保结果目录存在 (Ensure results directory exists)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 自动选择计算设备：如果CUDA GPU可用，则使用cuda；否则使用CPU。
# device对象后续会用于将张量和模型转移到选定的设备上。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备是: {device}")  # 打印当前使用的设备


# =========================
# 数据检查与初始化函数
# =========================
def check_data(data, name="data"):
    """
    检查输入数据中是否包含NaN (Not a Number，非数字) 或 Inf (Infinity，无穷大) 值。
    这些值通常表示计算错误或数据问题。

    Args:
        data (torch.Tensor): 需要检查的PyTorch张量。
        name (str): 数据的名称，用于在打印信息时区分不同的数据来源。
    """
    print(f"正在检查 {name} 是否包含 NaN 或 Inf 值...")
    has_nan = torch.isnan(data).any().item()  # 检查张量中是否有任何NaN值
    has_inf = torch.isinf(data).any().item()  # 检查张量中是否有任何Inf值
    print(f"是否包含 NaN: {has_nan}")
    print(f"是否包含 Inf: {has_inf}")
    if has_nan or has_inf:
        print(f"警告: {name} 包含 NaN 或 Inf 值!")


def initialize_weights(model):
    """
    对神经网络模型的权重进行Xavier均匀初始化，偏置项初始化为0。
    良好的权重初始化有助于模型的训练和收敛。Xavier初始化常用于激活函数为tanh或sigmoid的网络。

    Args:
        model (nn.Module): 需要初始化权重的PyTorch模型。
    """
    for name, param in model.named_parameters():  # 遍历模型的所有命名参数 (包括权重和偏置)
        if "weight" in name:  # 如果参数名称中包含"weight"
            if param.data.dim() > 1:  # 检查参数的维度。通常权重是多维的
                nn.init.xavier_uniform_(param)  # 使用Xavier均匀分布初始化权重
        elif "bias" in name:  # 如果参数名称中包含"bias"
            nn.init.constant_(param, 0)  # 将偏置项初始化为0


# =========================
# 1. Hybrid IDM 模型 (混合智能驾驶员模型)
# =========================
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dt=0.1):
        """
        混合IDM模型的初始化函数。
        Args:
            input_dim (int): LSTM的输入特征维度。
            hidden_dim (int): LSTM的隐藏层维度。
            num_layers (int): LSTM的层数。
            dt (float): 模拟的时间步长 (单位：秒)。
        """
        super(HybridIDMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 6)  # 输出IDM的6个参数
        self.softplus = nn.Softplus()  # Softplus确保参数为正
        self.delta_t = dt

    def forward(self, x):
        """
        模型的前向传播过程。LSTM处理时序信息，FC层输出IDM参数。
        Args:
            x (torch.Tensor): 输入的历史序列数据，维度为 (batch_size, seq_len, input_dim)。
        Returns:
            torch.Tensor: 经过Softplus激活后的IDM参数，维度为 (batch_size, 6)。
        """
        out, (hn, cn) = self.lstm(x)
        # 取LSTM最后一个时间步的输出作为IDM参数的依据
        params_raw = self.fc(out[:, -1, :])
        # 应用Softplus确保参数物理意义上的合理性（例如正值）
        params_activated = self.softplus(params_raw)
        return params_activated

    def predict_speed(self, x, s_actual):
        """
        使用IDM (Intelligent Driver Model) 跟驰公式预测跟驰车辆在下一时刻的速度。
        Args:
            x (torch.Tensor): 跟驰车辆的历史输入序列数据 (batch, seq_len, input_dim)。
                              特征: [自车速度, 实际间距(历史), 与前车速度差, 自车加速度, 前车速度]
                              注意: IDM公式中的间距是当前实际间距s_actual。
            s_actual (torch.Tensor): 当前的实际观测间距 (batch,)。
        Returns:
            torch.Tensor: 预测的下一时刻跟驰车速度 (batch, 1)。
            torch.Tensor: IDM模型参数 (batch, 6)，经过网络输出和钳位。
        """
        params = self.forward(x)  # 获取神经网络预测的IDM参数
        v_n = x[:, -1, 0]  # 当前自车速度 (序列最后一个时间点)
        delta_v_hist = x[:, -1, 2]  # 当前速度差 (v_leader - v_follower)

        # 解析并钳位IDM参数到合理范围，增强模型稳定性
        v_des_raw, T_raw, a_max_raw, b_safe_raw, delta_idm_raw, s0_raw = \
            params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]

        v_des = torch.clamp(v_des_raw, min=0.1, max=50.0)  # 期望速度 (m/s)
        T = torch.clamp(T_raw, min=0.1, max=5.0)  # 安全时距 (s)
        a_max = torch.clamp(a_max_raw, min=0.1, max=5.0)  # 最大加速度 (m/s^2)
        b_safe = torch.clamp(b_safe_raw, min=0.1, max=9.0)  # 舒适减速度 (m/s^2)
        delta_idm = torch.clamp(delta_idm_raw, min=1.0, max=10.0)  # 加速度指数
        s0 = torch.clamp(s0_raw, min=0.0, max=10.0)  # 最小静止间距 (m)
        s_actual_clamped = torch.clamp(s_actual, min=0.5)  # 钳位实际间距，防止过小

        # IDM公式计算
        # 期望动态间距 s_star
        sqrt_ab_clamped = torch.clamp(torch.sqrt(a_max * b_safe), min=1e-6)  # 防止开方结果为0
        # IDM交互项中的速度差 (v_n - v_l)，对应 -delta_v_hist
        interaction_term = (v_n * (-delta_v_hist)) / (2 * sqrt_ab_clamped + 1e-9)  # 加微小量防止除零
        s_star = s0 + torch.clamp(v_n * T, min=0.0) + interaction_term  # v_n * T 应非负
        s_star = torch.clamp(s_star, min=s0)  # s_star至少为s0

        # 加速度计算
        v_n_clamped = torch.clamp(v_n, min=0.0)  # 当前速度非负
        speed_ratio = (v_n_clamped + 1e-6) / (v_des + 1e-6)  # 加微小量防止除零
        term_speed_ratio = speed_ratio.pow(delta_idm)
        spacing_ratio = s_star / (s_actual_clamped + 1e-6)  # 加微小量防止除零
        term_spacing_ratio = spacing_ratio.pow(2)

        accel_component = 1.0 - term_speed_ratio - term_spacing_ratio
        a_idm_val = a_max * accel_component  # IDM计算出的加速度

        # 速度更新: v(t+dt) = v(t) + a(t)*dt
        v_follow = v_n + a_idm_val * self.delta_t
        v_follow = torch.clamp(v_follow, min=0.0, max=60.0)  # 预测速度钳位到合理范围 (0 ~ 216km/h)

        # NaN/Inf检查，用于调试
        if torch.isnan(v_follow).any() or torch.isinf(v_follow).any():
            print("警告: HybridIDMModel.predict_speed 中检测到 NaN/Inf 输出。")
            # 此处可以添加更详细的参数打印，帮助定位问题源头
        return v_follow.unsqueeze(1), params


# =========================
# 2. LNN 模型 (Liquid Neural Network) - 基类及用于前车预测的实现
# =========================
class LiquidCellMulti(nn.Module):  # 单个LNN神经元细胞
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        super(LiquidCellMulti, self).__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt  # ODE求解的时间步长
        # 线性变换层，不含偏置 (bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)  # 隐藏状态到隐藏状态
        self.W_u = nn.Linear(input_dim, hidden_dim, bias=False)  # 输入到隐藏状态
        self.bias = nn.Parameter(torch.zeros(hidden_dim))  # 可学习的偏置
        self.act = nn.Tanh()  # 激活函数

    def forward(self, u, h):
        # h_dot = -h + act(W_h*h + W_u*u + bias)
        # h_new = h_old + dt * h_dot (欧拉法)
        if h.shape[-1] != self.hidden_dim:  # 初始化隐藏状态 (通常在序列开始时)
            h = torch.zeros(u.shape[0], self.hidden_dim, device=u.device)
        dh = -h + self.act(self.W_h(h) + self.W_u(u) + self.bias)
        return h + self.dt * dh


class LiquidNeuralNetworkMultiStep(nn.Module):  # 用于前车(Leader)预测
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1, num_steps=50, dt=0.1):
        super(LiquidNeuralNetworkMultiStep, self).__init__()
        self.input_dim = input_dim
        self.cells = nn.ModuleList()  # 存储LNN层
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(LiquidCellMulti(current_input_dim, hidden_dim, dt=dt))
        self.fc = nn.Linear(hidden_dim, prediction_steps)  # 输出层
        self.num_steps = num_steps  # LNN内部模拟步数，通常等于输入序列长度
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch, seq, features = x.shape
        h_states = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        effective_seq_len = min(seq, self.num_steps)  # LNN实际处理的序列长度

        for t in range(effective_seq_len):  # 沿时间序列迭代
            u_t_layer = x[:, t, :]  # 当前时间步的输入
            for i in range(self.num_layers):  # 遍历LNN的每一层
                input_signal_for_cell = h_states[i - 1] if i > 0 else u_t_layer  # 第一层输入原始信号，后续层输入前一层隐态
                h_states[i] = self.cells[i](input_signal_for_cell, h_states[i])
        return self.fc(h_states[-1])  # 用最后一层LNN的最终隐藏状态进行预测

    def predict_speed(self, x):  # 便捷方法
        return self.forward(x)


# =========================
# LNN-Ego 模型 (LiquidNeuralNetworkMultiStepEgo) 已被移除
# =========================

# =========================
# 融合LSTM模型 (FusionLSTMModel) 已被移除
# =========================


# =========================
# 训练函数定义
# =========================
def train_generic_model(model, loader, optimizer, criterion, epochs=30, model_name="Generic Model", clip_value=1.0):
    """
    通用训练函数，用于训练前车LNN模型。
    Args:
        model (nn.Module): 需要训练的模型。
        loader (torch.utils.data.DataLoader): 训练数据加载器，产生 (x_batch, y_batch)。
        optimizer (torch.optim.Optimizer): 优化器。
        criterion (nn.Module): 损失函数。
        epochs (int): 训练的总轮数。
        model_name (str): 模型名称，用于打印日志。
        clip_value (float): 梯度裁剪的阈值。
    Returns:
        nn.Module: 训练完成的模型。
    """
    model.train()  # 设置模型为训练模式
    for ep in range(epochs):  # 遍历训练轮数
        tot_loss = 0  # 当前轮总损失
        num_batches_processed = 0  # 当前轮处理的批次数

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # 数据送往设备
            optimizer.zero_grad()
            # 对于LNN模型, predict_speed 等同于 forward
            pred = model.predict_speed(x_batch) if hasattr(model, 'predict_speed') else model(x_batch)
            loss = criterion(pred, y_batch)  # 计算损失

            # NaN/Inf 损失检查
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: {model_name} 在 epoch {ep + 1} 出现 NaN/Inf 损失。跳过此批次。")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            tot_loss += loss.item()
            num_batches_processed += 1

        # 计算并打印平均损失
        avg_loss = tot_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
        print(f"[{model_name}] Epoch {ep + 1}/{epochs}, 平均损失: {avg_loss:.4f}")
        # 如果非第一轮训练出现NaN平均损失，则可能存在严重问题，提前停止
        if np.isnan(avg_loss) and ep > 0:
            print(f"警告: {model_name} 平均损失为 NaN，训练提前停止。")
            break
    return model


def precompute_leader_trajectories_for_idm_training(
        leader_model, raw_data_slice, pred_steps_K, dt, device, hist_len=50
):
    """
    为IDM模型训练预计算前车轨迹 (速度和位置)。
    这可以避免在IDM训练的每个迭代中重复进行前车轨迹预测，提高效率。
    Args:
        leader_model (nn.Module): 预训练的前车LNN模型。
        raw_data_slice (torch.Tensor): 原始数据切片 (num_samples, seq_len, features)。
        pred_steps_K (int): 预测的未来步数。
        dt (float): 时间步长。
        device (torch.device): 计算设备。
        hist_len (int): 前车LNN模型输入历史长度。
    Returns:
        tuple: 包含IDM训练所需的各种张量。
    """
    leader_model.eval()  # 设置前车模型为评估模式
    num_samples = raw_data_slice.shape[0]

    # 处理空输入的情况，返回形状正确但为空的张量
    if num_samples == 0:
        _IDM_INPUT_DIM_placeholder = 5  # 应该与全局定义一致
        empty_tensor_k_steps = torch.empty(0, pred_steps_K, dtype=torch.float32, device=device)
        empty_tensor_idm_input = torch.empty(0, hist_len, _IDM_INPUT_DIM_placeholder, dtype=torch.float32,
                                             device=device)
        empty_tensor_scalar_batch = torch.empty(0, dtype=torch.float32, device=device)
        return empty_tensor_idm_input, empty_tensor_scalar_batch, empty_tensor_scalar_batch, \
            empty_tensor_k_steps, empty_tensor_k_steps, empty_tensor_scalar_batch

    # 准备IDM模型的初始输入序列 (特征: v_f, s, dv, a_f, v_l)
    initial_idm_input_seqs = raw_data_slice[:, -hist_len:, [0, 1, 2, 3, 5]].clone() * 0.3048  # 单位转换
    initial_follower_poses = raw_data_slice[:, -1, 4].clone() * 0.3048  # 自车初始位置
    initial_leader_poses_val = raw_data_slice[:, -1, -1].clone() * 0.3048  # 前车初始位置
    initial_s_safes = initial_idm_input_seqs[:, -1, 1].clone()  # 初始实际间距 (已经转换单位)
    # d1是初始 leader_pos - follower_pos - s_safe，用于后续间距校正
    batch_d1 = initial_leader_poses_val - initial_follower_poses - initial_s_safes

    # 准备前车LNN模型的输入 (特征: v_l, a_l)
    leader_hist_for_lnn = raw_data_slice[:, -hist_len:, [5, 6]].clone() * 0.3048  # 单位转换

    pred_leader_speeds_K_list = []  # 存储每个样本预测的前车速度序列
    pred_leader_pos_K_list = []  # 存储每个样本预测的前车位置序列
    current_dt = dt if dt > 1e-6 else 1e-6  # 确保dt有效

    with torch.no_grad():  # 不计算梯度
        # 批量预测前车未来K步速度
        all_pred_l_speeds_k_steps_tensor = leader_model.predict_speed(leader_hist_for_lnn.to(device)).cpu()  # 转到CPU处理

        for i in range(num_samples):  # 遍历每个样本
            pred_l_speeds_k_steps_tensor_i = all_pred_l_speeds_k_steps_tensor[i]  # 当前样本的预测速度序列
            pred_leader_speeds_K_list.append(pred_l_speeds_k_steps_tensor_i)

            # 基于预测速度，迭代计算前车未来K步位置
            current_l_pos = initial_leader_poses_val[i].item()  # 当前前车位置 (t=0)
            prev_l_v = leader_hist_for_lnn[i, -1, 0].item()  # 前车历史末端速度 (v at t=0)
            l_pos_k_steps = []  # 存储当前样本的未来K步位置

            for k_idx in range(pred_steps_K):  # 遍历未来K步
                vp = pred_l_speeds_k_steps_tensor_i[k_idx].item()  # 预测的 v(t+k+1)
                a_leader = (vp - prev_l_v) / current_dt  # 平均加速度
                displacement_leader = prev_l_v * current_dt + 0.5 * a_leader * current_dt * current_dt  # 位移
                next_l_pos = current_l_pos + displacement_leader  # 新位置
                l_pos_k_steps.append(next_l_pos)

                prev_l_v = vp  # 更新速度以供下一步计算
                current_l_pos = next_l_pos  # 更新位置
            pred_leader_pos_K_list.append(torch.tensor(l_pos_k_steps, dtype=torch.float32))

    # 将列表中的张量堆叠
    pred_leader_speeds_K = torch.stack(pred_leader_speeds_K_list) if num_samples > 0 else torch.empty(0, pred_steps_K,
                                                                                                      dtype=torch.float32)
    pred_leader_pos_K = torch.stack(pred_leader_pos_K_list) if num_samples > 0 else torch.empty(0, pred_steps_K,
                                                                                                dtype=torch.float32)

    # 返回所有计算好的张量，并确保它们在正确的设备上 (尤其是要输入到模型的)
    return initial_idm_input_seqs.to(device), initial_follower_poses.to(device), initial_s_safes.to(device), \
        pred_leader_speeds_K.to(device), pred_leader_pos_K.to(device), batch_d1.to(device)


def train_idm_model_multistep(
        model, train_loader, optimizer,
        num_epochs=30, pred_steps_K=5, dt=0.1, alpha_decay=0.0,  # alpha_decay 控制损失权重
        teacher_forcing_initial_ratio=1.0,  # 初始Teacher Forcing比例
        min_teacher_forcing_ratio=0.0,  # 最小Teacher Forcing比例
        teacher_forcing_decay_epochs_ratio=0.75,  # TF比例衰减所占轮数比例
        clip_value=1.0  # 梯度裁剪值
):
    """ 训练LSTM-IDM混合模型，采用多步预测和计划采样 (Scheduled Sampling / Teacher Forcing) """
    model.train()
    criterion_mse_elementwise = nn.MSELoss(reduction='none')  # 逐元素MSE，用于加权
    # 损失权重: w_t = exp(-alpha_decay * t)，越远预测步权重越小 (如果alpha_decay > 0)
    loss_weights = torch.exp(-alpha_decay * torch.arange(pred_steps_K, device=device).float())
    decay_epochs = int(num_epochs * teacher_forcing_decay_epochs_ratio)  # TF衰减的总轮数
    current_dt = dt if dt > 1e-6 else 1e-6  # 确保dt有效

    for epoch in range(num_epochs):
        total_loss_epoch = 0
        num_valid_batches = 0  # 记录有效（未产生NaN）的批次数

        # 计算当前轮次的Teacher Forcing比率 (线性衰减)
        current_teacher_forcing_ratio = teacher_forcing_initial_ratio - \
                                        (teacher_forcing_initial_ratio - min_teacher_forcing_ratio) * \
                                        (float(epoch) / decay_epochs if decay_epochs > 0 else 0)
        current_teacher_forcing_ratio = max(min_teacher_forcing_ratio, current_teacher_forcing_ratio)
        print(
            f"[LSTM-IDM 多步训练] Epoch [{epoch + 1}/{num_epochs}], Teacher Forcing Ratio: {current_teacher_forcing_ratio:.4f}")

        # 训练数据加载器提供预计算好的批次数据
        for batch_idx, (batch_initial_idm_input_seq,  # IDM初始输入历史
                        batch_true_follower_speeds_K_steps_for_loss,  # 真实自车未来K步速度 (用于损失)
                        batch_initial_follower_pos,  # 自车初始位置
                        batch_initial_s_safe,  # 初始实际间距
                        batch_pred_leader_speeds_K_steps,  # 预测的前车未来K步速度
                        batch_pred_leader_pos_K_steps,  # 预测的前车未来K步位置
                        batch_d1_offset,  # 间距校正量d1
                        batch_true_follower_all_features_K_steps,  # 真实自车未来K步所有相关特征 (用于TF)
                        batch_true_follower_pos_K_steps  # 真实自车未来K步位置 (用于TF)
                        ) in enumerate(train_loader):

            # 数据已经由precompute函数或DataLoader转移到device，但标签等可能需再次确认
            batch_true_follower_speeds_K_steps_for_loss = batch_true_follower_speeds_K_steps_for_loss.to(device)
            batch_true_follower_all_features_K_steps = batch_true_follower_all_features_K_steps.to(device)
            batch_true_follower_pos_K_steps = batch_true_follower_pos_K_steps.to(device)

            optimizer.zero_grad()
            # 初始化循环预测所需的状态变量 (复制以避免原地修改)
            batch_current_idm_input_torch = batch_initial_idm_input_seq.clone()
            batch_current_follower_speed_pred = batch_current_idm_input_torch[:, -1, 0].clone()  # 当前自车速度 (预测起点)
            batch_current_follower_pos = batch_initial_follower_pos.clone()  # 当前自车位置
            batch_current_s_actual_for_idm = batch_initial_s_safe.clone()  # 当前IDM的实际间距输入
            all_predicted_follower_speeds_batch_list = []  # 存储K步预测
            skip_batch_update = False  # 标记是否因NaN/Inf跳过当前批次

            # 在pred_steps_K个未来时间步上循环预测
            for k_step in range(pred_steps_K):
                # 输入检查
                if torch.isnan(batch_current_idm_input_torch).any() or torch.isinf(
                        batch_current_idm_input_torch).any() or \
                        torch.isnan(batch_current_s_actual_for_idm).any() or torch.isinf(
                    batch_current_s_actual_for_idm).any():
                    print(f"警告: IDM输入在 E:{epoch + 1}, B:{batch_idx}, K:{k_step} 包含NaN/Inf。跳过此批次。")
                    skip_batch_update = True;
                    break

                # IDM模型预测一步 (v_follower_t+k+1)
                v_follower_t_plus_k_plus_1_pred_batch_unsqueeze, _ = model.predict_speed(
                    batch_current_idm_input_torch, batch_current_s_actual_for_idm)
                v_follower_t_plus_k_plus_1_pred_batch = v_follower_t_plus_k_plus_1_pred_batch_unsqueeze.squeeze(1)

                # 预测输出检查
                if torch.isnan(v_follower_t_plus_k_plus_1_pred_batch).any() or torch.isinf(
                        v_follower_t_plus_k_plus_1_pred_batch).any():
                    print(f"警告: IDM预测速度在 E:{epoch + 1}, B:{batch_idx}, K:{k_step} 为NaN/Inf。跳过此批次。")
                    skip_batch_update = True;
                    break
                all_predicted_follower_speeds_batch_list.append(v_follower_t_plus_k_plus_1_pred_batch.unsqueeze(1))

                # 准备下一时间步 (k_step+1) 的IDM输入 (如果不是最后一步预测)
                if k_step < pred_steps_K - 1:
                    use_ground_truth = torch.rand(1).item() < current_teacher_forcing_ratio  # 决定是否用真实值

                    # 获取 t+k+1 时刻的前车预测速度和位置
                    v_leader_t_plus_k_plus_1_batch = batch_pred_leader_speeds_K_steps[:, k_step]
                    pos_leader_t_plus_k_plus_1_batch = batch_pred_leader_pos_K_steps[:, k_step]

                    if use_ground_truth:  # Teacher Forcing: 使用真实值构造下一步输入
                        v_f_next_true = batch_true_follower_all_features_K_steps[:, k_step,
                                        0]  # 对应k_step的真实值，作为 k_step+1 的输入状态
                        s_actual_next_true = batch_true_follower_all_features_K_steps[:, k_step, 1]
                        delta_v_next_true = batch_true_follower_all_features_K_steps[:, k_step, 2]
                        a_f_next_true = batch_true_follower_all_features_K_steps[:, k_step, 3]
                        pos_f_next_true = batch_true_follower_pos_K_steps[:, k_step]

                        # 新特征切片: [v_f_true, s_true, dv_true, a_f_true, v_l_pred]
                        new_feature_slice_batch = torch.stack([
                            v_f_next_true, s_actual_next_true, delta_v_next_true,
                            a_f_next_true, v_leader_t_plus_k_plus_1_batch  # 前车信息来自LNN预测
                        ], dim=1)
                        # 更新下一轮IDM预测所需的状态 (基于真实值)
                        batch_current_follower_speed_pred = v_f_next_true.clone()
                        batch_current_follower_pos = pos_f_next_true.clone()
                        batch_current_s_actual_for_idm = s_actual_next_true.clone()
                    else:  # Student Forcing: 使用模型自身预测构造下一步输入
                        # 计算自车加速度: a = (v_pred(t+1) - v_current_pred(t)) / dt
                        a_follower_t_plus_k_plus_1_batch = (
                                                                   v_follower_t_plus_k_plus_1_pred_batch - batch_current_follower_speed_pred) / current_dt
                        a_follower_t_plus_k_plus_1_batch = torch.clamp(a_follower_t_plus_k_plus_1_batch, -10.0,
                                                                       10.0)  # 钳位

                        # 计算自车位移和新位置
                        disp_follower_batch = batch_current_follower_speed_pred * current_dt + 0.5 * a_follower_t_plus_k_plus_1_batch * current_dt ** 2
                        pos_follower_t_plus_k_plus_1_batch = batch_current_follower_pos + disp_follower_batch

                        # 计算新间距
                        spacing_raw_t_plus_k_plus_1 = pos_leader_t_plus_k_plus_1_batch - pos_follower_t_plus_k_plus_1_batch
                        spacing_adjusted_t_plus_k_plus_1 = spacing_raw_t_plus_k_plus_1 - batch_d1_offset  # 校正
                        spacing_adjusted_t_plus_k_plus_1 = torch.clamp(spacing_adjusted_t_plus_k_plus_1, min=0.1)  # 钳位

                        # 计算新速度差
                        delta_v_t_plus_k_plus_1_batch = v_leader_t_plus_k_plus_1_batch - v_follower_t_plus_k_plus_1_pred_batch

                        # 新特征切片: [v_f_pred, s_pred, dv_pred, a_f_pred, v_l_pred]
                        new_feature_slice_batch = torch.stack([
                            v_follower_t_plus_k_plus_1_pred_batch, spacing_adjusted_t_plus_k_plus_1,
                            delta_v_t_plus_k_plus_1_batch, a_follower_t_plus_k_plus_1_batch,
                            v_leader_t_plus_k_plus_1_batch
                        ], dim=1)
                        # 更新下一轮IDM预测所需的状态 (基于模型自身预测)
                        batch_current_follower_speed_pred = v_follower_t_plus_k_plus_1_pred_batch.clone()
                        batch_current_follower_pos = pos_follower_t_plus_k_plus_1_batch.clone()
                        batch_current_s_actual_for_idm = spacing_adjusted_t_plus_k_plus_1.clone()

                    # 检查新生成的特征切片
                    if torch.isnan(new_feature_slice_batch).any() or torch.isinf(new_feature_slice_batch).any():
                        print(
                            f"警告: new_feature_slice 在 E:{epoch + 1}, B:{batch_idx}, K:{k_step} 包含NaN/Inf。跳过此批次。")
                        skip_batch_update = True;
                        break

                    # 更新IDM的输入序列: 移除最旧的，加入最新的
                    batch_current_idm_input_torch = torch.cat(
                        [batch_current_idm_input_torch[:, 1:, :], new_feature_slice_batch.unsqueeze(1)], dim=1)

            if skip_batch_update: optimizer.zero_grad(); continue  # 如果因NaN跳出，清零梯度并跳过此批次

            # 计算当前批次的损失
            batch_predicted_multi_step_speeds = torch.cat(all_predicted_follower_speeds_batch_list, dim=1)
            if torch.isnan(batch_predicted_multi_step_speeds).any() or torch.isinf(
                    batch_predicted_multi_step_speeds).any():
                print(f"警告: 最终预测速度序列在 E:{epoch + 1}, B:{batch_idx} 包含NaN/Inf。跳过此批次。")
                optimizer.zero_grad();
                continue

            squared_errors = criterion_mse_elementwise(batch_predicted_multi_step_speeds,
                                                       batch_true_follower_speeds_K_steps_for_loss)
            loss = (squared_errors * loss_weights.unsqueeze(0)).sum(dim=1).mean()  # 加权并求平均损失

            # 反向传播和参数更新
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 最终损失在 E:{epoch + 1}, B:{batch_idx} 为NaN/Inf。跳过参数更新。")
                optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                total_loss_epoch += loss.item()
                num_valid_batches += 1

        avg_loss_epoch = total_loss_epoch / num_valid_batches if num_valid_batches > 0 else float('nan')
        print(
            f"[LSTM-IDM 多步训练] Epoch [{epoch + 1}/{num_epochs}], 平均损失: {avg_loss_epoch:.4f} (基于 {num_valid_batches}/{len(train_loader)} 有效批次)")
        if np.isnan(avg_loss_epoch) and epoch > 0: print("警告: 平均损失为 NaN，训练提前停止。"); break
    return model


def evaluate_generic_model(model, test_loader, pred_steps=5, model_name="Generic Model", device_eval=None):
    """
    通用评估函数，用于评估前车LNN模型。
    """
    model.eval()  # 设置模型为评估模式
    all_predicted, all_true = [], []

    # 检查测试加载器是否有效
    if not test_loader or not hasattr(test_loader, 'dataset') or len(test_loader.dataset) == 0:
        print(f"{model_name} 评估: 测试数据集为空。跳过评估。")
        return

    with torch.no_grad():  # 不计算梯度
        for batch_data, batch_target_speed in test_loader:
            batch_data, batch_target_speed = batch_data.to(device_eval), batch_target_speed.to(device_eval)
            predicted_speed = model.predict_speed(batch_data) if hasattr(model, 'predict_speed') else model(batch_data)
            all_predicted.append(predicted_speed.cpu())  # 转到CPU存储
            all_true.append(batch_target_speed.cpu())  # 转到CPU存储

    if not all_predicted: print(f"{model_name} 评估: 没有做出预测。"); return
    all_predicted_cat = torch.cat(all_predicted, dim=0).numpy()
    all_true_cat = torch.cat(all_true, dim=0).numpy()
    if all_true_cat.shape[0] == 0: print(f"{model_name} 评估: 真实数据为空。"); return

    # 计算评估指标
    mse_val = np.mean((all_predicted_cat - all_true_cat) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted_cat - all_true_cat))
    # 分步指标
    mse_per_step = np.mean((all_predicted_cat - all_true_cat) ** 2, axis=0)
    rmse_per_step = np.sqrt(mse_per_step)
    mae_per_step = np.mean(np.abs(all_predicted_cat - all_true_cat), axis=0)

    print(f"\n{model_name} 评估 (总体 {pred_steps} 步):")
    print(f"  均方误差 (MSE): {mse_val:.4f}, 均方根误差 (RMSE): {rmse_val:.4f}, 平均绝对误差 (MAE): {mae_val:.4f}")
    # 确保打印分步指标时索引不越界
    for i in range(min(pred_steps, rmse_per_step.shape[0])):
        print(f"  第 {i + 1} 步预测: RMSE: {rmse_per_step[i]:.4f}, MAE: {mae_per_step[i]:.4f}")

    # 绘图：拼接部分样本的预测与真实轨迹
    num_plot_samples = min(30, all_true_cat.shape[0])  # 最多绘制30个完整序列
    # 计算采样间隔，确保均匀选取样本进行绘图
    plot_step_interval = max(1, all_true_cat.shape[0] // num_plot_samples if num_plot_samples > 0 else 1)
    true_concat_plot, pred_concat_plot = [], []

    # 从所有测试样本中，按间隔采样，并将每个采样样本的K步预测/真实值展平后拼接到绘图列表
    for i in range(0, all_true_cat.shape[0], plot_step_interval):
        if len(true_concat_plot) / pred_steps >= num_plot_samples: break  # 如果已达到绘图样本数上限
        true_concat_plot.extend(all_true_cat[i])  # 扩展真实值列表
        pred_concat_plot.extend(all_predicted_cat[i])  # 扩展预测值列表

    if true_concat_plot:  # 如果有数据可供绘图
        plt.figure(figsize=(12, 6))
        plt.plot(np.array(true_concat_plot), linestyle='--', marker='o', markersize=3, label=f'真实 {model_name} 速度')
        plt.plot(np.array(pred_concat_plot), linestyle='-', marker='x', markersize=3, label=f'预测 {model_name} 速度')
        plt.title(f'{model_name} 速度多步预测 (样本拼接)')
        plt.xlabel('预测范围中的时间步 (拼接后)')
        plt.ylabel('速度 (m/s)')
        plt.legend()
        plt.grid(True)
    else:
        print(f"{model_name} 测试集中没有足够的样本用于绘图。")


# 辅助函数，用于在LSTM-IDM模型评估时获取其多步预测
def get_idm_multistep_predictions(
        idm_model, leader_model, initial_idm_input_seq_batch, raw_data_slice_batch,
        pred_steps_K, dt, hist_len, device_compute
):
    """
    为LSTM-IDM模型评估，使用LSTM-IDM模型进行多步预测。
    Args:
        idm_model (HybridIDMModel): 预训练的LSTM-IDM模型。
        leader_model (LiquidNeuralNetworkMultiStep): 预训练的前车LNN模型。
        initial_idm_input_seq_batch (torch.Tensor): 当前批次的IDM初始输入历史 (batch, hist_len, idm_input_dim)。
        raw_data_slice_batch (torch.Tensor): 当前批次的原始数据切片，用于获取初始位置和前车历史。
        pred_steps_K (int): 预测步数。
        dt (float): 时间步长。
        hist_len (int): 历史序列长度。
        device_compute (torch.device): 计算设备。
    Returns:
        torch.Tensor: LSTM-IDM预测的未来K步自车速度 (batch, pred_steps_K)。
    """
    idm_model.eval()  # 设为评估模式
    leader_model.eval()  # 设为评估模式

    # 为当前批次预计算前车轨迹
    (_, initial_f_pos_batch, initial_s_safe_batch,
     pred_l_speeds_K_batch, pred_l_pos_K_batch, d1_offset_batch) = \
        precompute_leader_trajectories_for_idm_training(
            leader_model, raw_data_slice_batch.to(device_compute),
            pred_steps_K, dt, device_compute, hist_len
        )
    initial_idm_input_seq_batch = initial_idm_input_seq_batch.to(device_compute)
    initial_f_pos_batch = initial_f_pos_batch.to(device_compute)
    initial_s_safe_batch = initial_s_safe_batch.to(device_compute)
    pred_l_speeds_K_batch = pred_l_speeds_K_batch.to(device_compute)
    pred_l_pos_K_batch = pred_l_pos_K_batch.to(device_compute)
    d1_offset_batch = d1_offset_batch.to(device_compute)

    # 初始化循环预测的状态变量
    batch_current_idm_input_torch = initial_idm_input_seq_batch.clone()
    batch_current_follower_speed_pred = batch_current_idm_input_torch[:, -1, 0].clone()
    batch_current_follower_pos = initial_f_pos_batch.clone()
    batch_current_s_actual_for_idm = initial_s_safe_batch.clone()

    all_predicted_follower_speeds_batch_list = []
    current_dt_val = dt if dt > 1e-6 else 1e-6

    with torch.no_grad():
        for k_step in range(pred_steps_K):
            if torch.isnan(batch_current_idm_input_torch).any() or torch.isnan(batch_current_s_actual_for_idm).any():
                v_follower_pred = batch_current_follower_speed_pred.clone()
                if k_step == 0 and torch.isnan(v_follower_pred).any():
                    v_follower_pred = torch.zeros_like(v_follower_pred)
                print(f"警告: get_idm_multistep_predictions 中IDM输入含NaN (step {k_step})。使用回退速度。")
            else:
                v_follower_pred_unsqueeze, _ = idm_model.predict_speed(
                    batch_current_idm_input_torch, batch_current_s_actual_for_idm
                )
                v_follower_pred = v_follower_pred_unsqueeze.squeeze(1)

            if torch.isnan(v_follower_pred).any() or torch.isinf(v_follower_pred).any():
                nan_inf_mask = torch.isnan(v_follower_pred) | torch.isinf(v_follower_pred)
                v_follower_pred[nan_inf_mask] = batch_current_follower_speed_pred[nan_inf_mask]
                if torch.isnan(v_follower_pred).any():
                    v_follower_pred[torch.isnan(v_follower_pred)] = 0.0
                print(f"警告: get_idm_multistep_predictions 中IDM预测含NaN/Inf (step {k_step})。使用回退速度。")

            all_predicted_follower_speeds_batch_list.append(v_follower_pred.unsqueeze(1))

            if k_step < pred_steps_K - 1:
                v_leader_next = pred_l_speeds_K_batch[:, k_step]
                pos_leader_next = pred_l_pos_K_batch[:, k_step]
                a_follower_next = (v_follower_pred - batch_current_follower_speed_pred) / current_dt_val
                a_follower_next = torch.clamp(a_follower_next, -10.0, 10.0)
                disp_follower_batch = batch_current_follower_speed_pred * current_dt_val + \
                                      0.5 * a_follower_next * current_dt_val ** 2
                pos_follower_next = batch_current_follower_pos + disp_follower_batch
                spacing_raw_next = pos_leader_next - pos_follower_next
                spacing_adjusted_next = spacing_raw_next - d1_offset_batch
                spacing_adjusted_next = torch.clamp(spacing_adjusted_next, min=0.1)
                delta_v_next = v_leader_next - v_follower_pred
                new_feature_slice = torch.stack([
                    v_follower_pred, spacing_adjusted_next, delta_v_next,
                    a_follower_next, v_leader_next
                ], dim=1)

                if torch.isnan(new_feature_slice).any():
                    print(f"警告: get_idm_multistep_predictions 中new_feature_slice含NaN (step {k_step})。")
                else:
                    batch_current_idm_input_torch = torch.cat(
                        [batch_current_idm_input_torch[:, 1:, :], new_feature_slice.unsqueeze(1)], dim=1
                    )

                batch_current_follower_speed_pred = v_follower_pred.clone()
                batch_current_follower_pos = pos_follower_next.clone()
                batch_current_s_actual_for_idm = spacing_adjusted_next.clone()

    return torch.cat(all_predicted_follower_speeds_batch_list, dim=1)


def evaluate_final_lstm_idm_model(
        idm_model, leader_model_for_idm,
        raw_data_test_slice, label_data_test_slice,
        dt, pred_steps, hist_len_idm, device_comp,
        output_excel_filepath="lstm_idm_model_test_predictions.xlsx",
        excel_sheet_name="all_data"
):
    """
    执行最终的LSTM-IDM多步预测，评估其性能，并将详细预测结果保存到Excel。
    """
    idm_model.eval()
    leader_model_for_idm.eval()

    N_test = raw_data_test_slice.shape[0]
    if N_test == 0: print("LSTM-IDM评估: 测试数据为空。跳过评估和保存。"); return None, None, None, None, None, None

    # 获取 LSTM-IDM模型的预测输出
    idm_input_hist_test = raw_data_test_slice[:, -hist_len_idm:, [0, 1, 2, 3, 5]].clone() * 0.3048
    with torch.no_grad():
        y_lstm_idm_pred_speeds = get_idm_multistep_predictions(
            idm_model, leader_model_for_idm, idm_input_hist_test.to(device_comp), raw_data_test_slice,
            # raw_data_test_slice 会在函数内to(device)
            pred_steps, dt, hist_len_idm, device_comp
        )
    y_lstm_idm_pred_speeds_np = y_lstm_idm_pred_speeds.cpu().numpy()  # (N_test, pred_steps)

    # 加载真实自车未来K步速度用于评估
    true_f_speeds_np = label_data_test_slice[:, :pred_steps, 0].clone().cpu().numpy() * 0.3048

    # 基于预测的自车速度推算自车未来多步位置
    initial_ego_speeds_m_s = raw_data_test_slice[:, -1, 0].clone().cpu().numpy() * 0.3048
    initial_ego_positions_m = raw_data_test_slice[:, -1, 4].clone().cpu().numpy() * 0.3048
    pred_ego_positions_m_np = np.zeros_like(y_lstm_idm_pred_speeds_np)

    for i in range(N_test):
        current_speed_for_pos_calc = initial_ego_speeds_m_s[i]
        current_pos_for_pos_calc = initial_ego_positions_m[i]
        for k in range(pred_steps):
            predicted_speed_step_k = y_lstm_idm_pred_speeds_np[i, k]
            acceleration = (predicted_speed_step_k - current_speed_for_pos_calc) / dt
            displacement = current_speed_for_pos_calc * dt + 0.5 * acceleration * dt * dt
            new_position = current_pos_for_pos_calc + displacement
            pred_ego_positions_m_np[i, k] = new_position
            current_speed_for_pos_calc = predicted_speed_step_k
            current_pos_for_pos_calc = new_position

    # 加载真实自车未来K步位置用于对比
    true_ego_positions_m_np = label_data_test_slice[:, :pred_steps, 3].clone().cpu().numpy() * 0.3048

    # 计算速度评估指标
    true_f_speeds_mape_denom = np.where(np.abs(true_f_speeds_np) < 1e-5, 1e-5, true_f_speeds_np)  # 避免除零
    mse_speed = np.mean((y_lstm_idm_pred_speeds_np - true_f_speeds_np) ** 2)
    rmse_speed = np.sqrt(mse_speed)
    mae_speed = np.mean(np.abs(y_lstm_idm_pred_speeds_np - true_f_speeds_np))
    mape_speed = np.mean(np.abs((y_lstm_idm_pred_speeds_np - true_f_speeds_np) / true_f_speeds_mape_denom)) * 100
    rmse_per_step_speed = np.sqrt(np.mean((y_lstm_idm_pred_speeds_np - true_f_speeds_np) ** 2, axis=0))
    mae_per_step_speed = np.mean(np.abs(y_lstm_idm_pred_speeds_np - true_f_speeds_np), axis=0)

    print(f"\n最终 LSTM-IDM 模型预测结果 ({pred_steps} 步) - 自车速度:")
    print(f"  均方误差 (MSE): {mse_speed:.4f}")
    print(f"  均方根误差 (RMSE): {rmse_speed:.4f} m/s")
    print(f"  平均绝对误差 (MAE): {mae_speed:.4f} m/s")
    print(f"  平均绝对百分比误差 (MAPE): {mape_speed:.2f}%")
    for i in range(pred_steps):
        print(
            f"  第 {i + 1} 步速度预测 (LSTM-IDM): RMSE: {rmse_per_step_speed[i]:.4f}, MAE: {mae_per_step_speed[i]:.4f}")

    # 计算位置评估指标
    pos_rmse, pos_mape = np.nan, np.nan  # Initialize
    if true_ego_positions_m_np.shape == pred_ego_positions_m_np.shape and N_test > 0:
        true_f_pos_mape_denom = np.where(np.abs(true_ego_positions_m_np) < 1e-5, 1e-5, true_ego_positions_m_np)
        mse_pos = np.mean((pred_ego_positions_m_np - true_ego_positions_m_np) ** 2)
        pos_rmse = np.sqrt(mse_pos)
        mae_pos = np.mean(np.abs(pred_ego_positions_m_np - true_ego_positions_m_np))
        pos_mape = np.mean(np.abs((pred_ego_positions_m_np - true_ego_positions_m_np) / true_f_pos_mape_denom)) * 100
        rmse_per_step_pos = np.sqrt(np.mean((pred_ego_positions_m_np - true_ego_positions_m_np) ** 2, axis=0))
        mae_per_step_pos = np.mean(np.abs(pred_ego_positions_m_np - true_ego_positions_m_np), axis=0)

        print(f"\n最终 LSTM-IDM 模型推断结果 ({pred_steps} 步) - 自车位置:")
        print(f"  均方误差 (MSE): {mse_pos:.4f}")
        print(f"  均方根误差 (RMSE): {pos_rmse:.4f} m")
        print(f"  平均绝对误差 (MAE): {mae_pos:.4f} m")
        print(f"  平均绝对百分比误差 (MAPE): {pos_mape:.2f}%")
        for i in range(pred_steps):
            print(
                f"  第 {i + 1} 步位置推断 (LSTM-IDM): RMSE: {rmse_per_step_pos[i]:.4f}, MAE: {mae_per_step_pos[i]:.4f}")
    else:
        print("\n警告: 无法计算位置评估指标，真实位置与预测位置数据不匹配或为空。")

    # 保存预测速度、推断位置以及对应的真实值到Excel
    data_to_save_dict = {}
    for k_step_idx in range(pred_steps):
        data_to_save_dict[f"Pred_Ego_Speed_step{k_step_idx + 1}(m/s)"] = y_lstm_idm_pred_speeds_np[:, k_step_idx]
        data_to_save_dict[f"Pred_Ego_Pos_step{k_step_idx + 1}(m)"] = pred_ego_positions_m_np[:, k_step_idx]
        data_to_save_dict[f"True_Ego_Speed_step{k_step_idx + 1}(m/s)"] = true_f_speeds_np[:, k_step_idx]
        if true_ego_positions_m_np.shape[1] > k_step_idx:
            data_to_save_dict[f"True_Ego_Pos_step{k_step_idx + 1}(m)"] = true_ego_positions_m_np[:, k_step_idx]
        else:
            data_to_save_dict[f"True_Ego_Pos_step{k_step_idx + 1}(m)"] = np.full(N_test, np.nan)

    df_predictions = pd.DataFrame(data_to_save_dict)
    try:
        mode_for_writer = 'a' if os.path.exists(output_excel_filepath) else 'w'
        with pd.ExcelWriter(output_excel_filepath, engine="openpyxl", mode=mode_for_writer,
                            if_sheet_exists='replace') as writer:
            df_predictions.to_excel(writer, sheet_name=excel_sheet_name, index=False)
        print(
            f"\nLSTM-IDM模型在测试集上的详细预测（速度和推断位置）已保存至 '{output_excel_filepath}' 的 '{excel_sheet_name}' 工作表。")
    except Exception as e:
        try:
            print(f"以追加模式写入Excel失败 ({e})，尝试以覆盖模式写入...")
            with pd.ExcelWriter(output_excel_filepath, engine="openpyxl", mode="w") as writer:
                df_predictions.to_excel(writer, sheet_name=excel_sheet_name, index=False)
            print(
                f"\nLSTM-IDM模型在测试集上的详细预测（速度和推断位置）已通过覆盖模式保存至 '{output_excel_filepath}' 的 '{excel_sheet_name}' 工作表。")
        except Exception as e2:
            print(f"\n错误：无法将预测结果保存到Excel文件 '{output_excel_filepath}'。错误信息: {e2}")

    # 绘图：自车速度
    num_plot_samples = min(30, N_test)
    plot_interval = max(1, N_test // num_plot_samples if num_plot_samples > 0 else 1)
    true_concat_plot_speed, pred_concat_plot_speed = [], []
    for i in range(0, N_test, plot_interval):
        if len(true_concat_plot_speed) / pred_steps >= num_plot_samples: break
        true_concat_plot_speed.extend(true_f_speeds_np[i, :])
        pred_concat_plot_speed.extend(y_lstm_idm_pred_speeds_np[i, :])

    if true_concat_plot_speed:
        plt.figure(figsize=(12, 6))
        plt.plot(np.array(true_concat_plot_speed), linestyle='--', marker='o', markersize=3,
                 label='真实自车速度 (LSTM-IDM评估)')
        plt.plot(np.array(pred_concat_plot_speed), linestyle='-', marker='x', markersize=3,
                 label='预测LSTM-IDM速度 (LSTM-IDM评估)')
        plt.title(f'最终LSTM-IDM模型自车速度多步预测 (样本拼接)')
        plt.xlabel('预测范围中的时间步 (拼接后)')
        plt.ylabel('速度 (m/s)')
        plt.legend()
        plt.grid(True)
    else:
        print("LSTM-IDM评估中没有足够的测试数据用于绘制速度曲线。")

    # 绘图：自车位置
    if N_test > 0 and pred_ego_positions_m_np.shape == true_ego_positions_m_np.shape:
        true_concat_plot_pos, pred_concat_plot_pos = [], []
        for i in range(0, N_test, plot_interval):
            if len(true_concat_plot_pos) / pred_steps >= num_plot_samples: break
            true_concat_plot_pos.extend(true_ego_positions_m_np[i, :])
            pred_concat_plot_pos.extend(pred_ego_positions_m_np[i, :])

        if true_concat_plot_pos:
            plt.figure(figsize=(12, 6))
            plt.plot(np.array(true_concat_plot_pos), linestyle='--', marker='o', markersize=3,
                     label='真实自车位置 (LSTM-IDM评估)')
            plt.plot(np.array(pred_concat_plot_pos), linestyle='-', marker='x', markersize=3,
                     label='推断LSTM-IDM位置 (LSTM-IDM评估)')
            plt.title(f'最终LSTM-IDM模型自车位置多步推断 (样本拼接)')
            plt.xlabel('预测范围中的时间步 (拼接后)')
            plt.ylabel('位置 (m)')
            plt.legend()
            plt.grid(True)
        else:
            print("LSTM-IDM评估中没有足够的测试数据用于绘制位置曲线。")

    return mse_speed, rmse_speed, mae_speed, mape_speed, pos_rmse, pos_mape


all_datasets_metrics_summary = []  # 用于存储所有数据集的评估指标字典列表


def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse, pos_mape):
    # 将单个数据集的评估指标存入列表
    metrics = {
        "数据集 (Dataset)": dataset_name,
        "速度MSE (Speed_MSE)": speed_mse,
        "速度RMSE (Speed_RMSE)": speed_rmse,
        "速度MAE (Speed_MAE)": speed_mae,
        "速度MAPE (%) (Speed_MAPE)": speed_mape,  # 修正键名
        "位置RMSE (m) (Position_RMSE_m)": pos_rmse,
        "位置MAPE (%) (Position_MAPE_percent)": pos_mape
    }
    all_datasets_metrics_summary.append(metrics)


def save_all_metrics_to_csv(filepath="evaluation_summary_lstm_idm.csv"):
    # 将所有数据集的评估指标汇总保存到CSV文件
    if not all_datasets_metrics_summary:
        print("没有评估指标可以保存。")
        return
    df_metrics = pd.DataFrame(all_datasets_metrics_summary)
    df_metrics.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"所有数据集的评估指标汇总已保存至 {filepath}")


# =========================
# 主函数 (脚本执行入口)
# =========================
if __name__ == "__main__":
    torch.manual_seed(42)  # 设置随机种子保证结果可复现
    np.random.seed(42)
    # torch.autograd.set_detect_anomaly(True) # 调试时开启

    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    print(f"找到的.mat文件: {data_files}")  # 打印找到的文件列表
    if not data_files:
        print(f"在目录 {DATA_DIR} 中未找到 .mat 文件。")
        exit()

    print(f"找到以下数据集文件: {data_files}")

    # 遍历每个找到的数据文件
    for data_file_path in data_files:
        dataset_filename = os.path.basename(data_file_path)
        dataset_name_clean = dataset_filename.replace(".mat", "")

        print(f"\n==================== 开始处理数据集: {dataset_filename} ====================")

        data = sio.loadmat(data_file_path)
        # --- 定义超参数和配置 ---
        DT = 0.1  # 时间步长 (秒)
        HIST_LEN = 50  # LSTM-IDM 和 LNN 输入的历史序列长度

        IDM_INPUT_DIM = 5  # LSTM-IDM的输入特征维度 (v_f, s, dv, a_f, v_l)
        LEADER_LNN_INPUT_DIM = 2  # 前车LNN模型的输入特征维度 (v_l, a_l)

        # 模型隐藏层维度
        HIDDEN_DIM_IDM = 64
        HIDDEN_DIM_LNN_LEADER = 64

        # 模型层数
        NUM_LAYERS_IDM = 1
        NUM_LAYERS_LNN_LEADER = 1

        # 训练轮数
        LEADER_LNN_EPOCHS = 50 # 原为50
        IDM_MULTISTEP_EPOCHS = 50  # 原为50

        BATCH_SIZE = 32

        # --- 加载数据 ---
        if 'train_data' not in data or ('lable_data' not in data and 'label_data' not in data):
            print("错误: .mat 文件中未找到 'train_data' 或 'lable_data'/'label_data'。请检查数据文件。")
            # exit() #  如果希望在单个文件错误时继续处理其他文件，可以注释掉exit()
            continue  # 跳过当前文件，处理下一个

        label_key = 'lable_data' if 'lable_data' in data else 'label_data'
        raw_data_full = torch.tensor(data['train_data'], dtype=torch.float32)
        label_data_full = torch.tensor(data[label_key], dtype=torch.float32)

        # --- 数据子集选择与划分 ---
        total_samples_full = raw_data_full.shape[0]
        num_samples_to_use = int(total_samples_full * 0.2)  # 使用100%数据

        raw_data_all = raw_data_full[:num_samples_to_use]
        label_data_all = label_data_full[:num_samples_to_use]
        print(f"使用 {num_samples_to_use} 个样本进行处理 (总样本数: {total_samples_full})")
        if num_samples_to_use == 0: print("错误: 没有样本可供使用。"); continue  # 跳过当前文件

        train_ratio_main = 0.8
        num_total_main = raw_data_all.shape[0]
        num_train_main = int(num_total_main * train_ratio_main)
        if num_train_main == 0 and num_total_main > 0: num_train_main = max(1,
                                                                            num_total_main - 1 if num_total_main > 1 else 1)
        if num_train_main == num_total_main and num_total_main > 1: num_train_main = num_total_main - 1
        num_test_main = num_total_main - num_train_main

        print(f"主拆分: 总样本 {num_total_main}, 训练样本 {num_train_main}, 测试样本 {num_test_main}")
        if num_train_main == 0 or num_test_main == 0:
            print(f"警告: 数据集 {dataset_filename} 划分后训练集或测试集为空，跳过此数据集。")
            continue

        raw_train_data = raw_data_all[:num_train_main]
        label_train_data = label_data_all[:num_train_main]
        raw_test_data = raw_data_all[num_train_main:]
        label_test_data = label_data_all[num_train_main:]

        # --- 1. 训练前车 LNN 模型 (Leader LNN) ---
        print("\n--- 1. 训练前车 LNN 模型 (Leader LNN) ---")
        leader_lnn_input_hist_train = raw_train_data[:, -HIST_LEN:, [5, 6]].clone() * 0.3048
        leader_lnn_target_speeds_train = label_train_data[:, :, 4].clone() * 0.3048
        PRED_STEPS_K = leader_lnn_target_speeds_train.shape[1]  # 从标签数据动态获取预测步长

        leader_lnn_input_hist_test = raw_test_data[:, -HIST_LEN:, [5, 6]].clone() * 0.3048
        leader_lnn_target_speeds_test = label_test_data[:, :PRED_STEPS_K, 4].clone() * 0.3048

        leader_lnn_train_loader, leader_lnn_test_loader = None, None
        if leader_lnn_input_hist_train.shape[0] > 0:
            leader_lnn_train_dataset = torch.utils.data.TensorDataset(leader_lnn_input_hist_train,
                                                                      leader_lnn_target_speeds_train)
            leader_lnn_train_loader = torch.utils.data.DataLoader(leader_lnn_train_dataset, batch_size=BATCH_SIZE,
                                                                  shuffle=True)
        if leader_lnn_input_hist_test.shape[0] > 0:
            leader_lnn_test_dataset = torch.utils.data.TensorDataset(leader_lnn_input_hist_test,
                                                                     leader_lnn_target_speeds_test)
            leader_lnn_test_loader = torch.utils.data.DataLoader(leader_lnn_test_dataset, batch_size=BATCH_SIZE,
                                                                 shuffle=False)

        leader_model = LiquidNeuralNetworkMultiStep(
            LEADER_LNN_INPUT_DIM, HIDDEN_DIM_LNN_LEADER, PRED_STEPS_K, NUM_LAYERS_LNN_LEADER, HIST_LEN, DT
        ).to(device)
        initialize_weights(leader_model)
        optimizer_lead = optim.Adam(leader_model.parameters(), lr=1e-3)
        criterion_mse = nn.MSELoss()

        if leader_lnn_train_loader:
            train_generic_model(leader_model, leader_lnn_train_loader, optimizer_lead, criterion_mse,
                                LEADER_LNN_EPOCHS, "前车LNN (Leader)", clip_value=1.0)
        if leader_lnn_test_loader:
            evaluate_generic_model(leader_model, leader_lnn_test_loader, PRED_STEPS_K, "前车LNN (Leader)",
                                   device_eval=device)

        # --- 2. 准备并训练 LSTM-IDM 模型 ---
        print("\n--- 2. 训练 LSTM-IDM 模型 ---")
        idm_multistep_train_loader = None
        if raw_train_data.shape[0] > 0:
            (initial_idm_seq_train, initial_f_pos_train, initial_s_safe_train,
             pred_l_speeds_K_train, pred_l_pos_K_train, d1_train) = \
                precompute_leader_trajectories_for_idm_training(
                    leader_model, raw_train_data, PRED_STEPS_K, DT, device, HIST_LEN
                )
            true_f_speeds_K_train_for_loss = label_train_data[:, :PRED_STEPS_K, 0].clone() * 0.3048
            true_v_f_K = label_train_data[:, :PRED_STEPS_K, 0].clone()
            true_s_K = label_train_data[:, :PRED_STEPS_K, 1].clone()
            true_a_f_K = label_train_data[:, :PRED_STEPS_K, 2].clone()
            true_v_l_K = label_train_data[:, :PRED_STEPS_K, 4].clone()
            true_dv_K = true_v_l_K - true_v_f_K
            true_f_all_features_K_train = torch.stack([true_v_f_K, true_s_K, true_dv_K, true_a_f_K], dim=2) * 0.3048
            true_f_pos_K_train = label_train_data[:, :PRED_STEPS_K, 3].clone() * 0.3048

            if initial_idm_seq_train.shape[0] > 0:
                idm_multistep_train_dataset = torch.utils.data.TensorDataset(
                    initial_idm_seq_train, true_f_speeds_K_train_for_loss.to(device),
                    initial_f_pos_train, initial_s_safe_train,
                    pred_l_speeds_K_train, pred_l_pos_K_train, d1_train,
                    true_f_all_features_K_train.to(device), true_f_pos_K_train.to(device)
                )
                idm_multistep_train_loader = torch.utils.data.DataLoader(idm_multistep_train_dataset,
                                                                         batch_size=BATCH_SIZE, shuffle=True)
                print(f"LSTM-IDM: 训练数据样本数 {initial_idm_seq_train.shape[0]}")
            else:
                print("LSTM-IDM: 预计算后训练数据为空。")

        idm_model = HybridIDMModel(IDM_INPUT_DIM, HIDDEN_DIM_IDM, NUM_LAYERS_IDM, DT).to(device)
        initialize_weights(idm_model)
        optimizer_idm = optim.Adam(idm_model.parameters(), lr=2e-4, weight_decay=1e-5)
        if idm_multistep_train_loader:
            train_idm_model_multistep(
                idm_model, idm_multistep_train_loader, optimizer_idm,
                IDM_MULTISTEP_EPOCHS, PRED_STEPS_K, DT, alpha_decay=0.05,
                teacher_forcing_initial_ratio=1.0, min_teacher_forcing_ratio=0.0,
                teacher_forcing_decay_epochs_ratio=0.75, clip_value=1.0
            )
        else:
            print("LSTM-IDM: 由于训练数据加载器为空，跳过训练。")

        # --- 3. 最终 LSTM-IDM 模型评估与结果保存 ---
        print("\n--- 3. 最终 LSTM-IDM 模型评估与结果保存 ---")

        output_excel_filename = os.path.join(RESULTS_DIR, f"lstm_idm_predictions_{dataset_name_clean}.xlsx")
        dataset_basename_for_sheet = dataset_name_clean  # 使用清理后的数据集名作为sheet名

        if raw_test_data.shape[0] > 0 and label_test_data.shape[0] > 0:
            speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse, pos_mape = evaluate_final_lstm_idm_model(
                idm_model, leader_model,  # 传入训练好的模型
                raw_test_data, label_test_data,  # 测试数据
                DT, PRED_STEPS_K, HIST_LEN, device,  # 相关参数
                output_excel_filepath=output_excel_filename,
                excel_sheet_name=dataset_basename_for_sheet
            )
            if speed_mse is not None:  # 检查是否有有效指标返回
                store_dataset_metrics(dataset_name_clean, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse,
                                      pos_mape)
        else:
            print(f"数据集 {dataset_name_clean} 没有足够的测试数据用于最终的LSTM-IDM模型评估和保存。")

        print(f"\n==================== 数据集: {dataset_filename} 处理完毕 ====================")

    # 在所有数据集处理完毕后，保存汇总的评估指标
    summary_metrics_csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_all_datasets_lstm_idm1128.csv")
    save_all_metrics_to_csv(summary_metrics_csv_path)

    # 显示所有绘图 (如果需要一次性显示所有图)
    # plt.show()
    # 如果希望每个数据集的图表在处理完该数据集后立即显示，可以将 plt.show() 放在循环内部的绘图函数之后。
    # 但通常，如果图很多，脚本结束时统一显示或保存到文件更常见。当前代码是保存图到文件。

    print("\n--- 所有流程执行完毕 ---")

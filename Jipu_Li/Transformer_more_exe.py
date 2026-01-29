import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import glob  # 用于查找文件路径
import math  # 导入 math 模块以使用 pi

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许多个OpenMP库存在，避免某些环境下的冲突
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备是: {device}")  # 打印当前使用的设备
# --- 全局路径定义 ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # 数据集存放目录
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim"  # 实验结果保存目录

# 确保结果目录存在 (Ensure results directory exists)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------
# 数据检查函数
def check_data(data, name="data"):
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# ------------------------------
# 位置编码 (Positional Encoding)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # 创建一个形状为 (max_len, d_model) 的零张量，用于存储位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 创建一个从 0 到 max_len-1 的位置索引张量
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算位置编码的频率项
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数索引位置的正弦编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数索引位置的余弦编码
        pe = pe.unsqueeze(0).transpose(0, 1)  # 调整张量形状以匹配输入 (seq_len, batch_size, d_model)
        self.register_buffer('pe', pe)  # 将 pe 注册为模型缓冲区，这样它会被保存和加载，但不会被视为模型参数

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]  # 将位置编码添加到输入张量中
        return self.dropout(x)  # 应用 dropout


# ------------------------------
# Transformer 模型 (Transformer Model)
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 prediction_steps, dropout=0.1, num_steps=50):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.input_dim = input_dim  # 输入特征的维度
        self.model_dim = model_dim  # Transformer模型的内部维度 (d_model)
        self.num_steps = num_steps  # 输入序列的长度

        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_len=num_steps)  # 位置编码器
        self.embedding = nn.Linear(input_dim, model_dim)  # 输入线性嵌入层，将输入维度映射到模型维度

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout, batch_first=True)  # 定义Transformer编码器层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers,
                                                         num_layers=num_encoder_layers)  # 定义完整的Transformer编码器

        # 输出层，将Transformer的输出映射到预测步长
        # Transformer编码器的输出是 (batch_size, seq_len, model_dim)
        # 我们需要将其展平或取最后一个时间步的输出，然后通过线性层
        self.fc_out = nn.Linear(model_dim * num_steps, prediction_steps)  # 方案1: 展平所有时间步的输出
        # self.fc_out = nn.Linear(model_dim, prediction_steps) # 方案2: 只使用最后一个时间步的输出 (需要调整forward)

        self.init_weights()  # 初始化权重

    def init_weights(self):
        initrange = 0.1  # 初始化权重的范围
        self.embedding.weight.data.uniform_(-initrange, initrange)  # 初始化嵌入层权重
        self.embedding.bias.data.zero_()  # 初始化嵌入层偏置
        self.fc_out.weight.data.uniform_(-initrange, initrange)  # 初始化输出层权重
        self.fc_out.bias.data.zero_()  # 初始化输出层偏置

    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        src = self.embedding(src) * math.sqrt(self.model_dim)  # 输入嵌入并缩放
        # TransformerEncoderLayer 默认期望 (seq_len, batch_size, model_dim) 如果 batch_first=False
        # 如果 batch_first=True, 则期望 (batch_size, seq_len, model_dim)
        # 我们的 PositionalEncoding 设计为 (seq_len, batch_size, model_dim), 需要适配
        # 为了简单起见，这里我们让 PositionalEncoding 适配 batch_first=True 的输入

        # 调整 PositionalEncoding 的适配方式
        # src (batch_size, seq_len, model_dim)
        # PositionalEncoding expects (seq_len, batch_size, d_model) internally for its buffer 'pe'
        # but its forward takes (seq_len, batch_size, d_model) and returns it.
        # We need to transpose src for pos_encoder and then transpose back.
        # 或者修改 PositionalEncoding 的 forward 或 TransformerEncoder 的 batch_first 设置
        # 当前 TransformerEncoderLayer 设置了 batch_first=True，所以输入是 (batch_size, seq_len, model_dim)

        # 如果 PositionalEncoding 的 forward 接受 (batch_size, seq_len, d_model)
        # 那么 PositionalEncoding 需要修改如下：
        # pe = pe.transpose(0,1) # 在 __init__ 中，使得 pe 为 (batch_size, max_len, d_model)
        # def forward(self, x): # x: (batch_size, seq_len, d_model)
        #    x = x + self.pe[:, :x.size(1), :]
        #    return self.dropout(x)
        # 为了保持 PositionalEncoding 原样，我们在这里调整：
        src = src.transpose(0, 1)  # (seq_len, batch_size, model_dim)
        src = self.pos_encoder(src)  # 应用位置编码
        src = src.transpose(0, 1)  # (batch_size, seq_len, model_dim)

        output = self.transformer_encoder(src)  # 通过Transformer编码器
        # output: (batch_size, seq_len, model_dim)

        # 使用展平的输出
        output = output.reshape(output.size(0), -1)  # 将输出展平为 (batch_size, seq_len * model_dim)
        output = self.fc_out(output)  # 通过输出线性层得到预测结果
        # output: (batch_size, prediction_steps)

        # 如果只使用最后一个时间步的输出 (需要修改 self.fc_out 的输入维度)
        # output = output[:, -1, :]  # (batch_size, model_dim)
        # output = self.fc_out(output) # (batch_size, prediction_steps)

        return output

    def predict_speed(self, x):
        return self.forward(x)  # 预测速度


# ------------------------------
# 液态单元（Liquid Cell）- 保留以防万一，但未使用
class LiquidCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        super(LiquidCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.activation = nn.Tanh()

    def forward(self, u, h):
        dh = -h + self.activation(self.W_h(h) + self.W_u(u) + self.bias)
        h_new = h + self.dt * dh
        return h_new


# ------------------------------
# 多步预测液态神经网络 - 保留以防万一，但未使用
class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1, num_steps=50):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.prediction_steps = prediction_steps

        self.liquid_cells = nn.ModuleList([
            LiquidCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, prediction_steps)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        T = min(seq_len, self.num_steps)
        h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        for t in range(T):
            inp = x[:, t, :]
            for i, cell in enumerate(self.liquid_cells):
                h[i] = cell(inp if i == 0 else h[i - 1], h[i])
        out = self.fc(h[-1])
        return out

    def predict_speed(self, x):
        return self.forward(x)


# ------------------------------
# 权重初始化 (通用，但 TransformerModel 有自己的初始化)
def initialize_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)


# ------------------------------
# 训练函数
def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed in train_loader:
            optimizer.zero_grad()
            pred = model.predict_speed(batch_data)  # 使用模型的 predict_speed 方法
            loss = criterion(pred, batch_speed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    return model


# ------------------------------
# 评估函数
def evaluate_model(model, train_real_speed, test_loader, dataset_name="", results_dir=""):
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    all_predicted = []
    all_true = []

    with torch.no_grad():
        for batch_idx, (batch_data, batch_speed) in enumerate(test_loader):
            predicted_speed = model.predict_speed(batch_data)  # 使用模型的 predict_speed 方法
            loss = mse_loss(predicted_speed, batch_speed)
            total_mse += loss.item()
            all_predicted.append(predicted_speed)
            all_true.append(batch_speed)

    # 合并所有批次的预测值与真实值
    all_predicted = torch.cat(all_predicted).cpu().numpy()  # shape: (N, prediction_steps)
    all_true = torch.cat(all_true).cpu().numpy()  # shape: (N, prediction_steps)

    # 计算整体评估指标
    mse_val = np.mean((all_predicted - all_true) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted - all_true))
    # 避免除以零的 MAPE 计算
    true_non_zero = all_true[all_true != 0]
    pred_non_zero = all_predicted[all_true != 0]
    if len(true_non_zero) == 0:
        mape_p = float('inf')  # 如果所有真实值都是0，MAPE是未定义的或无穷大
    else:
        mape_p = np.mean(np.abs((pred_non_zero - true_non_zero) / true_non_zero)) * 100
    print(f"Evaluation Metrics:\nMSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_p:.2f}%")

    # 拼接所有样本的数据，步长为 prediction_steps
    # train_real_speed.shape[1] 是 prediction_steps
    num_pred_steps = all_true.shape[1]
    num_samples_to_plot = 30
    true_concat = []
    pred_concat = []

    # 我们有 num_test_samples * num_pred_steps 个点
    # 我们要绘制的是每个样本的 num_pred_steps 个预测点，连续绘制 num_samples_to_plot 个样本
    for i in range(min(num_samples_to_plot, all_true.shape[0])):
        true_concat.extend(all_true[i])  # 添加第 i 个样本的真实多步预测
        pred_concat.extend(all_predicted[i])  # 添加第 i 个样本的预测多步预测

    # 转换为数组以便绘制
    true_concat = np.array(true_concat)
    pred_concat = np.array(pred_concat)

    # 绘制拼接后的对比曲线
    plt.figure(figsize=(12, 8))
    plt.plot(true_concat, linestyle='--', marker='o', label='True')
    plt.plot(pred_concat, linestyle='-', marker='x', label='Predicted')

    # 设置图形标题和标签
    plt.title(f'True vs Predicted Speed for {num_samples_to_plot} Samples (Multi-step)')
    plt.xlabel(f'Time Step (concatenated over {num_pred_steps} steps per sample)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()

    plot_filename = os.path.join(results_dir, f"{dataset_name}_transformer_speed_comparison.png")
    plt.savefig(plot_filename)  # 保存图像
    print(f"速度对比图已保存至 {plot_filename}")
    plt.close()  # 关闭图像，释放内存
    return mse_val, rmse_val, mae_val, mape_p  # 返回评估指标


# ------------------------------
# 保存预测速度到 CSV
def save_predictions_to_csv(model, test_loader, output_file="predictions.csv"):
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for bd, bt in test_loader:
            p = model.predict_speed(bd)  # 使用模型的 predict_speed 方法
            all_t.append(bt.cpu())
            all_p.append(p.cpu())
    all_t = torch.cat(all_t).numpy()
    all_p = torch.cat(all_p).numpy()
    df = pd.DataFrame({
        **{f"True_Step{j + 1}": all_t[:, j] for j in range(all_t.shape[1])},
        **{f"Pred_Step{j + 1}": all_p[:, j] for j in range(all_p.shape[1])}
    })
    df.to_csv(output_file, index=False)
    print(f"Speeds saved to {output_file}")


# ------------------------------
# 多步预测前车未来位置计算并保存
def compute_future_positions_and_save(model,
                                      test_data,  # (N_test, seq_len, features)
                                      raw_data,  # 原始未划分的数据集部分
                                      label_data,  # 原始未划分的标签部分
                                      train_size,  # 训练集大小，用于定位测试集在原始数据中的索引起点
                                      dt=0.1,
                                      output_file="pred_positions.xlsx",
                                      dataset_name=""):
    model.eval()
    with torch.no_grad():
        pred_speeds = model.predict_speed(test_data).cpu().numpy()  # (N_test, prediction_steps)

    N_test, steps = pred_speeds.shape  # steps 是 prediction_steps
    # 测试集在原始数据中的索引起始点
    test_start_idx_in_raw = train_size
    test_end_idx_in_raw = train_size + N_test

    # 当前速度（m/s）: test_data 是已经处理过的输入特征，其最后一个时间步的第0个特征是当前速度
    # 注意：test_data 已经是 m/s 单位了 (因为 train_data 在主函数中转换了)
    curr_speed_m_s = test_data[:, -1, 0].cpu().numpy()  # (N_test,)

    # 当前前车位置（ft -> m）
    # raw_data 中的第8列 (索引7) 是原始的前车全局位置 Preceding_Global_Y (ft)
    # 我们需要测试集样本在 raw_data 中对应的最后一个时间步的前车位置
    # raw_data[index, time_step, feature_index]
    # test_data 的每个样本对应 raw_data 中的一个样本，取其最后一个时间步
    curr_pos_ft = raw_data[test_start_idx_in_raw:test_end_idx_in_raw, -1, 7].cpu().numpy()  # (N_test,)
    curr_pos_m = curr_pos_ft * 0.3048  # (N_test,)

    # 真值前车未来位置（ft -> m）
    # label_data 中的第6列 (索引5) 是未来多步的前车全局位置 Global_Y_prec (ft)
    # label_data[index, step, feature_index]
    true_pos_ft = label_data[test_start_idx_in_raw:test_end_idx_in_raw, :,
                  5].cpu().numpy()  # (N_test, prediction_steps)
    true_pos_m = true_pos_ft * 0.3048  # (N_test, prediction_steps)

    # 真值车速（m/s）
    # label_data 中的第5列 (索引4) 是未来多步的前车车速 v_prec (ft/s)
    # train_real_speed_all = label_data[:, :, 0] -> 这个是label_data第0列，需要确认是哪个速度
    # 根据原代码，train_real_speed = train_real_speed_all.clone() * 0.3048
    # 而 train_real_speed_all = label_data[:, :, 0]
    # 这意味着 label_data 的第0列是我们要预测的速度 (ft/s)
    # 我们这里需要的 true_speeds 是用于对比预测位置的，应该是 *预测的目标车辆* 的真实未来速度
    # pred_speeds 是模型预测的目标车辆的未来速度
    # 因此，这里的 true_speeds 应该是 test_real_speed (已经是 m/s)
    # test_real_speed 是从 train_real_speed 切分出来的，而 train_real_speed 是 label_data[:, :, 0] * 0.3048
    true_speeds_m_s = label_data[test_start_idx_in_raw:test_end_idx_in_raw, :,
                      0].cpu().numpy() * 0.3048  # (N_test, prediction_steps)

    # 初始化存储所有样本的预测位置
    pred_pos_m = np.zeros((N_test, steps))  # (N_test, prediction_steps)

    # 对每个样本单独进行位置递推计算
    for i in range(N_test):
        # 每个样本的初始车速和位置
        # 这里的 prev_speed 应该是目标车辆在 t 时刻的速度，即 curr_speed_m_s[i]
        # 而 pred_speeds[i, k] 是目标车辆在 t+1, t+2 ... t+steps 时刻的预测速度
        prev_speed_for_disp = curr_speed_m_s[i]  # 目标车辆在 t 时刻的速度 (m/s)
        prev_pos_for_disp = curr_pos_m[i]  # 前车在 t 时刻的位置 (m)

        # 计算当前样本的未来位置 (这里是前车的位置)
        # 我们预测的是目标车辆的速度 pred_speeds[i,k]
        # 如果要预测 *前车* 的未来位置，我们需要 *前车* 的预测速度
        # 但模型是根据历史数据预测 *目标车辆* 的未来速度
        # 这里的逻辑似乎是：用 *目标车辆的预测速度* 来推算 *前车的未来位置* ？这有点不直观。
        # 假设这里的意图是：如果知道前车在 t 时刻的位置和速度，并 *假设前车以我们模型预测出的（目标车的）速度行驶*，它的位置会如何变化。
        # 或者，更合理的解释是，pred_speeds 实际上是预测的 *前车* 的速度。
        # 从 LNN 代码的 train_real_speed_all = label_data[:, :, 0] 来看，label_data 的第0列是 'v_Vel_Pred' 即前车速度。
        # 所以 pred_speeds 是预测的 *前车* 的未来多步速度。

        # 确认: raw_data[:, -50:, [0, 1, 2, 3, 5]]
        # [0]: 'v_Vel_Pred' - 前车速度 (ft/s)
        # [1]: 'v_Follow_Vel' - 跟随车（目标车）速度 (ft/s)
        # [2]: 'v_Delta_Vel' - 速度差 (ft/s)
        # [3]: 'v_Space' - 间距 (ft)
        # [5]: 'v_Preceding_Acc' - 前车加速度 (ft/s^2)
        # label_data[:, :, 0] 是 'v_Vel_Pred' (前车速度)
        # 所以模型是根据历史的各种信息，预测未来多步的 *前车速度*。

        # 因此，pred_speeds[i, k] 是第 i 个样本的第 k+1 个未来时间步的 *前车预测速度* (m/s)
        # curr_speed_m_s[i] 是 test_data[:, -1, 0]，而 test_data 的第0列是 'v_Vel_Pred' (前车速度)
        # 所以 curr_speed_m_s[i] 是第 i 个样本在 t 时刻的 *前车实际速度* (m/s)
        # curr_pos_m[i] 是第 i 个样本在 t 时刻的 *前车实际位置* (m)

        current_prec_speed_m_s = curr_speed_m_s[i]  # 前车在 t 的速度
        current_prec_pos_m = curr_pos_m[i]  # 前车在 t 的位置

        for k in range(steps):  # 遍历未来时间步 0 to steps-1 (对应 t+1 to t+steps)
            v_pred_prec = pred_speeds[i, k]  # 前车在 t+k+1 时刻的预测速度 (m/s)

            # 使用前车在 t+k 的速度 (current_prec_speed_m_s) 和 t+k+1 的预测速度 (v_pred_prec)
            # 来计算 t+k 到 t+k+1 之间的位移
            # 平均速度 = (current_prec_speed_m_s + v_pred_prec) / 2
            # 位移 = 平均速度 * dt
            # 或者用牛顿运动学：假设 t+k 到 t+k+1 之间是匀加速运动
            # a_prec = (v_pred_prec - current_prec_speed_m_s) / dt # 前车在 (t+k, t+k+1) 区间的平均加速度
            # disp = current_prec_speed_m_s * dt + 0.5 * a_prec * dt ** 2 # 前车在 (t+k, t+k+1) 区间的位移
            # current_prec_pos_m = current_prec_pos_m + disp # 前车在 t+k+1 的预测位置

            # 简化：假设在 (t+k, t+k+1) 区间内，前车以 current_prec_speed_m_s 匀速行驶，然后瞬间变为 v_pred_prec
            # 或者，更常见的是，用 t+k 时刻的速度来计算 t+k 到 t+k+1 的位移 (即假设速度在小区间内恒定)
            # disp = current_prec_speed_m_s * dt
            # 这种方式会导致滞后。

            # 采用原LNN代码中的逻辑：
            # a = (v_pred - prev_speed) / dt  # 计算加速度
            # disp = prev_speed * dt + 0.5 * a * dt ** 2  # 计算位移
            # pos = prev_pos + disp  # 更新位置
            # prev_speed = v_pred
            # prev_pos = pos
            # 这里的 prev_speed 是上一个时间步的速度，v_pred 是当前时间步的预测速度
            # 对于第k步 (预测 t+k+1 时刻的状态)
            # prev_speed 就是 current_prec_speed_m_s (t+k 时刻的速度)
            # v_pred 就是 pred_speeds[i, k] (t+k+1 时刻的预测速度)
            # prev_pos 就是 current_prec_pos_m (t+k 时刻的位置)

            a_prec_step = (v_pred_prec - current_prec_speed_m_s) / dt
            disp_step = current_prec_speed_m_s * dt + 0.5 * a_prec_step * dt ** 2
            current_prec_pos_m += disp_step  # 更新到 t+k+1 时刻的位置
            pred_pos_m[i, k] = current_prec_pos_m  # 存储 t+k+1 时刻的预测位置

            current_prec_speed_m_s = v_pred_prec  # 更新 t+k+1 时刻的速度，用于下一个循环 (计算t+k+2的位移)

    # 误差评估
    # true_pos_m 是 (N_test, prediction_steps)，对应 t+1 到 t+steps 的真实位置
    rmse_p = np.sqrt(np.mean((pred_pos_m - true_pos_m) ** 2))
    # 避免除以零
    true_pos_m_flat_non_zero = true_pos_m[true_pos_m != 0]
    pred_pos_m_flat_for_mape = pred_pos_m[true_pos_m != 0]
    if len(true_pos_m_flat_non_zero) == 0:
        mape_p = float('inf')
    else:
        mape_p = np.mean(np.abs((pred_pos_m_flat_for_mape - true_pos_m_flat_non_zero) / true_pos_m_flat_non_zero)) * 100
    print(f"Future Position Error -- RMSE: {rmse_p:.4f} m, MAPE: {mape_p:.2f}%")

    # 保存到 CSV
    data_dict = {}
    for i in range(steps):  # steps is prediction_steps
        data_dict[f"Pred_Speed_step{i + 1}(m/s)"] = pred_speeds[:, i]  # 前车预测速度
        data_dict[f"Pred_Pos_step{i + 1}(m)"] = pred_pos_m[:, i]  # 前车预测位置
        data_dict[f"True_Pos_step{i + 1}(m)"] = true_pos_m[:, i]  # 前车真实位置
        # true_speeds_m_s 是前车的真实未来速度
        data_dict[f"True_Speed_step{i + 1}(m/s)"] = true_speeds_m_s[:, i]

    df_pos = pd.DataFrame(data_dict)
    sheet_name = dataset_name

    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            df_pos.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df_pos.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"{dataset_name} 的位置预测结果已保存至 '{output_file}' 的 '{sheet_name}' 工作表。")
    return rmse_p, mape_p


all_datasets_metrics_summary = []


def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse, pos_mape):
    metrics = {
        "数据集 (Dataset)": dataset_name,
        "速度MSE (Speed_MSE)": speed_mse,
        "速度RMSE (Speed_RMSE)": speed_rmse,
        "速度MAE (Speed_MAE)": speed_mae,
        "速度MAPE (%) (Speed_MAPE)": speed_mape,  # 修正了key中的重复MAE
        "位置RMSE (m) (Position_RMSE_m)": pos_rmse,
        "位置MAPE (%) (Position_MAPE_percent)": pos_mape
    }
    all_datasets_metrics_summary.append(metrics)


def save_all_metrics_to_csv(filepath="evaluation_summary.csv"):
    if not all_datasets_metrics_summary:
        print("没有评估指标可以保存。")
        return
    df_metrics = pd.DataFrame(all_datasets_metrics_summary)
    df_metrics.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"所有数据集的评估指标汇总已保存至 {filepath}")


# ------------------------------
# 主函数
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)  # 也为 numpy 设置随机种子

    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    if not data_files:
        print(f"在目录 {DATA_DIR} 中未找到 .mat 文件。")
        exit()

    # 为 Transformer 模型定义新的Excel文件名和CSV文件名
    position_predictions_excel_path = os.path.join(RESULTS_DIR, "pred_positions_all_datasets_Transformer1128.xlsx")
    summary_metrics_csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_all_datasets_Transformer1128.csv")

    # 如果旧的汇总Excel/CSV文件存在，可以选择删除
    if os.path.exists(position_predictions_excel_path):
        os.remove(position_predictions_excel_path)
        print(f"已删除旧的 Transformer 位置预测Excel文件: {position_predictions_excel_path}")
    if os.path.exists(summary_metrics_csv_path):
        os.remove(summary_metrics_csv_path)
        print(f"已删除旧的 Transformer 评估汇总CSV文件: {summary_metrics_csv_path}")

    print(f"找到以下数据集文件: {data_files}")

    for data_file_path in data_files:
        dataset_filename = os.path.basename(data_file_path)
        dataset_name_clean = dataset_filename.replace(".mat", "")

        print(f"\n==================== 开始处理数据集: {dataset_filename} (Transformer Model) ====================")

        data = sio.loadmat(data_file_path)
        raw_data_np = data['train_data']  # (num_samples, total_seq_len, num_features_raw)
        label_data_np = data['lable_data']  # (num_samples, prediction_steps, num_label_features)

        raw_data = torch.tensor(raw_data_np, dtype=torch.float32)
        label_data = torch.tensor(label_data_np, dtype=torch.float32)

        # 提取多步速度标签（前车速度在第 1 列，索引 0）
        # lable_data columns: ['v_Vel_Pred', 'v_Follow_Vel', 'v_Delta_Vel', 'v_Space', 'v_Preceding_Acc', 'Global_Y_prec']
        # 我们要预测的是前车未来多步速度 'v_Vel_Pred', 即 label_data 的第0列
        train_real_speed_all_ft_s = label_data[:, :, 0]  # (num_samples, prediction_steps) in ft/s
        print("train_real_speed_all (ft/s) shape:", train_real_speed_all_ft_s.shape)

        # 构造多步输入：取原始数据最后50步的特征
        # train_data columns from raw_data: ['v_Vel_Pred'(0), 'v_Follow_Vel'(1), 'v_Delta_Vel'(2), 'v_Space'(3), 'Lane_ID'(4), 'v_Preceding_Acc'(5), 'Vehicle_ID'(6), 'Global_Y_prec'(7)]
        # 原 LNN 代码选择: raw_data[:, -50:, [0, 1, 2, 3, 5]]
        # 即: 前车速度, 跟随车速度, 速度差, 间距, 前车加速度
        input_features_indices = [0, 1, 2, 3, 5]
        train_data_ft = raw_data[:, -50:,
                        input_features_indices].clone()  # (num_samples, 50, 5) in ft or ft/s or ft/s^2

        # 单位转换：ft/s -> m/s, ft -> m
        train_data_m = train_data_ft.clone()
        train_data_m[:, :, 0] *= 0.3048  # 前车速度 v_Vel_Pred (m/s)
        train_data_m[:, :, 1] *= 0.3048  # 跟随车速度 v_Follow_Vel (m/s)
        train_data_m[:, :, 2] *= 0.3048  # 速度差 v_Delta_Vel (m/s)
        train_data_m[:, :, 3] *= 0.3048  # 间距 v_Space (m)
        train_data_m[:, :, 4] *= 0.3048  # 前车加速度 v_Preceding_Acc (m/s^2) - ft/s^2 to m/s^2

        train_real_speed_m_s = train_real_speed_all_ft_s.clone() * 0.3048  # (num_samples, prediction_steps) in m/s

        print("train_data_m (input features in meters) shape:", train_data_m.shape)
        print("train_real_speed_m_s (target speeds in m/s) shape:", train_real_speed_m_s.shape)

        # 抽样 (如果需要)
        sample_fraction = 0.2  # 使用全部数据
        total_samples = train_data_m.shape[0]
        sample_size = int(total_samples * sample_fraction)

        # 打乱数据顺序 (可选，但如果 DataLoader shuffle=True，则此处非必须)
        # perm = torch.randperm(total_samples)
        # train_data_sampled = train_data_m[perm][:sample_size]
        # train_real_speed_sampled = train_real_speed_m_s[perm][:sample_size]
        # raw_data_sampled_for_pos = raw_data[perm][:sample_size] # 如果打乱，raw_data 和 label_data 也要对应打乱
        # label_data_sampled_for_pos = label_data[perm][:sample_size]

        train_data_sampled = train_data_m[:sample_size]
        train_real_speed_sampled = train_real_speed_m_s[:sample_size]
        # 在 compute_future_positions_and_save 中，raw_data 和 label_data 用于根据 train_size 索引测试集部分
        # 所以这里不应该对 raw_data 和 label_data 进行抽样或打乱，除非相应调整索引逻辑

        print(f"Using {sample_size} samples after sampling ({sample_fraction * 100}%).")

        check_data(train_data_sampled, "train_data_sampled (m)")
        check_data(train_real_speed_sampled, "train_real_speed_sampled (m/s)")

        dataset_size = train_data_sampled.shape[0]
        train_size = int(dataset_size * 0.8)

        # 划分训练/测试集
        # 注意：如果之前打乱了，这里的划分是在打乱后的数据上进行的
        # 如果没有打乱，就是按原始顺序划分
        train_X = train_data_sampled[:train_size]
        train_Y = train_real_speed_sampled[:train_size]
        test_X = train_data_sampled[train_size:]
        test_Y = train_real_speed_sampled[train_size:]

        # 用于位置计算的 test_raw_data 和 test_label_data 不需要单独创建，
        # compute_future_positions_and_save 会使用完整的 raw_data, label_data 和 train_size 来定位测试集部分

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_X, train_Y),
            batch_size=32, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_X, test_Y),
            batch_size=32, shuffle=False
        )

        # Transformer 模型配置
        input_dim = train_X.shape[2]  # 输入特征维度 (5)
        model_dim = 128  # Transformer 内部维度 (d_model)
        nhead = 4  # 多头注意力头数 (应能被 model_dim 整除)
        num_encoder_layers = 3  # Transformer编码器层数
        num_decoder_layers = 0  # 我们只使用编码器
        dim_feedforward = 256  # 前馈网络维度
        prediction_steps = train_Y.shape[1]  # 预测步长 (e.g., 5)
        dropout = 0.1  # Dropout率
        seq_length = train_X.shape[1]  # 输入序列长度 (50)

        # 确保 nhead 能被 model_dim 整除
        if model_dim % nhead != 0:
            # 调整 nhead 或 model_dim
            # 例如，选择一个能被 model_dim 整除的 nhead，或者调整 model_dim
            print(f"Warning: model_dim ({model_dim}) is not divisible by nhead ({nhead}). Adjusting nhead.")
            # 简单的调整策略：找到 model_dim 的最大因子且 <= 原 nhead
            possible_nheads = [h for h in range(1, nhead + 1) if model_dim % h == 0]
            if not possible_nheads:  # 如果没有合适的因子，可能需要重新考虑 model_dim
                nhead = 1  # 或者抛出错误
            else:
                nhead = max(possible_nheads)
            print(f"Adjusted nhead to: {nhead}")

        model = TransformerModel(
            input_dim=input_dim,
            model_dim=model_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,  # 未使用
            dim_feedforward=dim_feedforward,
            prediction_steps=prediction_steps,
            dropout=dropout,
            num_steps=seq_length  # 传递序列长度给 TransformerModel
        )
        # TransformerModel 有自己的 init_weights，所以 initialize_weights(model) 不是必须的，除非你想覆盖它
        # initialize_weights(model) # 可以注释掉

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Transformer 通常用较小的学习率

        print("Starting Transformer model training...")
        model = train_model(model, train_loader, optimizer, criterion, num_epochs=30)  # 可调整 epoch 数

        print("Evaluating Transformer model...")
        # train_real_speed (用于绘图的 train_Y) 已经是 m/s
        speed_mse, speed_rmse, speed_mae, speed_mape = evaluate_model(
            model,
            train_Y,
            # Pass train_Y for consistency in plot reference if needed, though not directly used by evaluate_model logic now
            test_loader,
            dataset_name=dataset_name_clean,
            results_dir=RESULTS_DIR
        )

        # 多步预测前车未来位置计算并保存
        # test_X 是测试集的输入特征 (N_test, seq_len, features) in meters/m/s
        # raw_data 是完整的原始数据 (torch tensor, ft/s units)
        # label_data 是完整的原始标签 (torch tensor, ft/s units)
        # train_size 是训练集样本数
        print("Computing future positions with Transformer model...")
        pos_rmse, pos_mape = compute_future_positions_and_save(
            model,
            test_X,  # 测试集输入数据 (m/s)
            raw_data,  # 原始完整数据 (ft/s) - 用于提取初始位置等
            label_data,  # 原始完整标签 (ft/s) - 用于提取真实未来位置
            train_size,  # 训练集大小，用于在 raw_data/label_data 中定位测试集
            dt=0.1,  # 时间步长
            output_file=position_predictions_excel_path,  # 新的 Excel 文件
            dataset_name=dataset_name_clean
        )
        store_dataset_metrics(dataset_name_clean, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse, pos_mape)

    # 保存所有数据集的评估指标到新的CSV文件
    save_all_metrics_to_csv(summary_metrics_csv_path)
    print("\n所有数据集使用 Transformer 模型处理完毕。")

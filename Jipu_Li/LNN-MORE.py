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
# # ------------------------------
# # 数据检查函数
# def check_data(data, name="data"):
#     print(f"Checking {name} for NaN or Inf values...")
#     print(f"Has NaN: {torch.isnan(data).any().item()}")
#     print(f"Has Inf: {torch.isinf(data).any().item()}")
#
# # ------------------------------
# # 液态单元（Liquid Cell）
# class LiquidCell(nn.Module):
#     def __init__(self, input_dim, hidden_dim, dt=0.1):
#         super(LiquidCell, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.dt = dt
#         self.W_h = nn.Linear(hidden_dim, hidden_dim)
#         self.W_u = nn.Linear(input_dim, hidden_dim)
#         self.bias = nn.Parameter(torch.zeros(hidden_dim))
#         self.activation = nn.Tanh()
#
#     def forward(self, u, h):
#         dh = -h + self.activation(self.W_h(h) + self.W_u(u) + self.bias)
#         h_new = h + self.dt * dh
#         return h_new
#
# # ------------------------------
# # 多步预测液态神经网络
# class LiquidNeuralNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1, num_steps=50):
#         super(LiquidNeuralNetwork, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.num_steps = num_steps
#         self.prediction_steps = prediction_steps
#
#         self.liquid_cells = nn.ModuleList([
#             LiquidCell(input_dim if i == 0 else hidden_dim, hidden_dim)
#             for i in range(num_layers)
#         ])
#         self.fc = nn.Linear(hidden_dim, prediction_steps)
#
#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
#         T = min(seq_len, self.num_steps)
#         h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
#         for t in range(T):
#             inp = x[:, t, :]
#             for i, cell in enumerate(self.liquid_cells):
#                 h[i] = cell(inp if i == 0 else h[i-1], h[i])
#         out = self.fc(h[-1])
#         return out
#
#     def predict_speed(self, x):
#         return self.forward(x)
#
# # ------------------------------
# # 权重初始化
# def initialize_weights(model):
#     for name, param in model.named_parameters():
#         if "weight" in name:
#             if param.dim() >= 2:
#                 nn.init.xavier_uniform_(param)
#             else:
#                 nn.init.uniform_(param)
#         elif "bias" in name:
#             nn.init.constant_(param, 0)
#
# # ------------------------------
# # 训练函数
# def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch_data, batch_speed in train_loader:
#             optimizer.zero_grad()
#             pred = model.predict_speed(batch_data)
#             loss = criterion(pred, batch_speed)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
#     return model
#
# # ------------------------------
# # 评估函数
# def evaluate_model(model, test_loader):
#     model.eval()
#     mse_loss = nn.MSELoss()
#     total_mse = 0
#     all_predicted = []
#     all_true = []
#
#     with torch.no_grad():
#         for batch_idx, (batch_data, batch_speed) in enumerate(test_loader):
#             predicted_speed = model.predict_speed(batch_data)
#             loss = mse_loss(predicted_speed, batch_speed)
#             total_mse += loss.item()
#             all_predicted.append(predicted_speed)
#             all_true.append(batch_speed)
#
#     # 合并所有批次的预测值与真实值
#     all_predicted = torch.cat(all_predicted).cpu().numpy()  # shape: (N, prediction_steps)
#     all_true = torch.cat(all_true).cpu().numpy()  # shape: (N, prediction_steps)
#
#     # 计算整体评估指标
#     mse_val = np.mean((all_predicted - all_true) ** 2)
#     rmse_val = np.sqrt(mse_val)
#     mae_val = np.mean(np.abs(all_predicted - all_true))
#     print(f"Evaluation Metrics:\nMSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")
#
#     # 拼接所有样本的数据，步长为5
#     step = 5
#     num_samples = 30
#     true_concat = []
#     pred_concat = []
#
#     for i in range(num_samples):
#         idx = i * step
#         if idx >= all_true.shape[0]:
#             break
#         # 拼接每个样本的真实值和预测值
#         true_concat.extend(all_true[idx])
#         pred_concat.extend(all_predicted[idx])
#
#     # 转换为数组以便绘制
#     true_concat = np.array(true_concat)
#     pred_concat = np.array(pred_concat)
#
#     # 绘制拼接后的对比曲线
#     plt.figure(figsize=(12, 8))
#     plt.plot(true_concat, linestyle='--', marker='o', label='True')
#     plt.plot(pred_concat, linestyle='-', marker='x', label='Predicted')
#
#     # 设置图形标题和标签
#     plt.title('True vs Predicted Speed for 30 Samples (Step 1-150)')
#     plt.xlabel('Time Step')
#     plt.ylabel('Speed (m/s)')
#     plt.legend()
#     plt.grid()
#     plt.show()
#
# # ------------------------------
# # 保存预测速度到 CSV
# def save_predictions_to_csv(model, test_loader, output_file="predictions.csv"):
#     model.eval()
#     all_t, all_p = [], []
#     with torch.no_grad():
#         for bd, bt in test_loader:
#             p = model.predict_speed(bd)
#             all_t.append(bt.cpu())
#             all_p.append(p.cpu())
#     all_t = torch.cat(all_t).numpy()
#     all_p = torch.cat(all_p).numpy()
#     df = pd.DataFrame({
#         **{f"True_Step{j+1}": all_t[:, j] for j in range(all_t.shape[1])},
#         **{f"Pred_Step{j+1}": all_p[:, j] for j in range(all_p.shape[1])}
#     })
#     df.to_csv(output_file, index=False)
#     print(f"Speeds saved to {output_file}")
#
# # ------------------------------
# # 多步预测前车未来位置计算并保存
# def compute_future_positions_and_save(model,
#                                       test_data,
#                                       raw_data,
#                                       label_data,
#                                       train_size,
#                                       dt=0.1,
#                                       output_file="pred_positions..xlsx"):
#     model.eval()
#     with torch.no_grad():
#         pred_speeds = model.predict_speed(test_data).cpu().numpy()  # (N_test, steps)
#     N_test, steps = pred_speeds.shape
#     idx = np.arange(train_size, train_size + N_test)
#
#     # 当前速度（m/s）
#     curr_speed = test_data[:, -1, 0].cpu().numpy()
#     # 当前前车位置（ft -> m）
#     curr_pos_ft = raw_data[idx, -1, 7].cpu().numpy()
#     curr_pos_m = curr_pos_ft * 0.3048
#
#     # 真值前车未来位置（ft -> m），label_data [:, :, 5] 存储前车未来位置
#     true_pos_ft = label_data[idx, :, 5].cpu().numpy()
#     true_pos_m = true_pos_ft * 0.3048
#
#     # 真值车速（m/s），label_data[:, :, 4] 存储车速
#     true_speeds = label_data[idx, :, 4].cpu().numpy()
#
#     # 初始化存储所有样本的预测位置
#     pred_pos_m = np.zeros((N_test, steps))
#
#     # 对每个样本单独进行位置递推计算
#     for i in range(N_test):
#         # 每个样本的初始车速和位置
#         prev_speed = curr_speed[i]
#         prev_pos = curr_pos_m[i]
#
#         # 计算当前样本的未来位置
#         for k in range(steps):
#             v_pred = pred_speeds[i, k]  # 当前样本在第k步的预测车速
#             # 使用牛顿运动学公式计算位移
#             a = (v_pred - prev_speed) / dt  # 计算加速度
#             disp = prev_speed * dt + 0.5 * a * dt ** 2  # 计算位移
#             pos = prev_pos + disp  # 更新位置
#             pred_pos_m[i, k] = pos  # 更新预测位置
#             prev_speed = v_pred  # 更新车速
#             prev_pos = pos  # 更新位置
#
#     # 误差评估
#     rmse_p = np.sqrt(np.mean((pred_pos_m - true_pos_m) ** 2))
#     mape_p = np.mean(np.abs((pred_pos_m - true_pos_m) / true_pos_m)) * 100
#     print(f"Future Position Error -- RMSE: {rmse_p:.4f} m, MAPE: {mape_p:.2f}%")
#
#     # 保存到 CSV
#     data_dict = {}
#     for i in range(steps):
#         data_dict[f"Pred_Speed_step{i + 1}(m/s)"] = pred_speeds[:, i]
#         data_dict[f"Pred_Pos_step{i + 1}(m)"] = pred_pos_m[:, i]
#         data_dict[f"True_Pos_step{i + 1}(m)"] = true_pos_m[:, i]
#         data_dict[f"True_Speed_step{i + 1}(m/s)"] = true_speeds[:, i]  # 添加真实速度
#
#     df_pos = pd.DataFrame(data_dict)
#     # df_pos.to_csv(output_file, index=False)
#     # print(f"Predicted positions and true speeds saved to {output_file}")
#     sheet_name = "LNN_1"  # 指定新工作表名称
#
#     # 如果文件存在则在其基础上添加新 sheet，否则新建文件
#     with pd.ExcelWriter(output_file, engine="openpyxl", mode="a" if os.path.exists(output_file) else "w") as writer:
#         df_pos.to_excel(writer, sheet_name=sheet_name, index=False)
#         print(f"New model results saved to '{output_file}' in sheet '{sheet_name}'.")
#
#
# # ------------------------------
# # 主函数
# if __name__ == "__main__":
#     torch.manual_seed(42)
#
#     # 加载数据
#     data = sio.loadmat('E:\\pythonProject1\\data_ngsim\\data_5.mat')
#     raw_data = torch.tensor(data['train_data'], dtype=torch.float32)
#     label_data = torch.tensor(data['lable_data'], dtype=torch.float32)  # 保留全部 6 列
#
#     # 提取多步速度标签（前车速度在第 5 列索引 4）
#     train_real_speed_all = label_data[:, :, 4]
#     print("train_real_speed_all shape:", train_real_speed_all.shape)
#
#     # 构造多步输入：取最后 50 步的前车速度(5)和加速度(6)
#     train_data = raw_data[:, -50:, [5, 6]]
#     train_real_speed = train_real_speed_all.clone()
#     print("train_data shape:", train_data.shape)
#
#     # 单位转换：ft/s -> m/s
#     train_data[:, :, 0] *= 0.3048
#     train_data[:, :, 1] *= 0.3048
#     train_real_speed *= 0.3048
#
#     # 抽样 10%
#     total_samples = train_data.shape[0]
#     sample_size = int(total_samples * 0.1)
#     train_data = train_data[:sample_size]
#     train_real_speed = train_real_speed[:sample_size]
#
#     # 检查
#     check_data(train_data, "train_data")
#     check_data(train_real_speed, "train_real_speed")
#
#     # 划分训练/测试集
#     dataset_size = train_data.shape[0]
#     train_size = int(dataset_size * 0.8)
#     test_data = train_data[train_size:]
#     test_real_speed = train_real_speed[train_size:]
#     train_data = train_data[:train_size]
#     train_real_speed = train_real_speed[:train_size]
#
#     train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(train_data, train_real_speed),
#         batch_size=32, shuffle=True
#     )
#     test_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(test_data, test_real_speed),
#         batch_size=32, shuffle=False
#     )
#
#     # 模型配置
#     input_dim = train_data.shape[2]
#     hidden_dim = 128
#     prediction_steps = train_real_speed.shape[1]  # 5
#     model = LiquidNeuralNetwork(input_dim, hidden_dim, prediction_steps,
#                                 num_layers=1, num_steps=train_data.shape[1])
#     initialize_weights(model)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=5e-4)
#
#     # 训练 & 评估
#     model = train_model(model, train_loader, optimizer, criterion, num_epochs=100)
#     evaluate_model(model, test_loader)
#
#     # 保存速度预测
#     # save_predictions_to_csv(model, test_loader, output_file="predictions..xlsx")
#
#     # **新增**：多步预测前车未来位置计算并保存
#     compute_future_positions_and_save(
#         model,
#         test_data,
#         raw_data,
#         label_data,
#         train_size,
#         dt=0.1,
#         output_file="pred_positions.xlsx"
#     )

import os
import glob  # 用于查找文件路径
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许多个OpenMP库存在，避免某些环境下的冲突

# --- 全局路径定义 ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # 数据集存放目录
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim"  # 实验结果保存目录

# 确保结果目录存在 (Ensure results directory exists)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------
# 数据检查函数 (Data checking function)
def check_data(data, name="data"):
    # 检查数据中是否包含 NaN 或 Inf 值
    print(f"正在检查 {name} 是否包含 NaN 或 Inf 值...")  # Checking {name} for NaN or Inf values...
    print(f"包含 NaN: {torch.isnan(data).any().item()}")  # Has NaN
    print(f"包含 Inf: {torch.isinf(data).any().item()}")  # Has Inf


# ------------------------------
# 液态单元（Liquid Cell）
class LiquidCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        super(LiquidCell, self).__init__()
        self.hidden_dim = hidden_dim  # 隐层维度
        self.dt = dt  # 时间步长 (微分方程中的dt)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)  # 隐层到隐层的权重
        self.W_u = nn.Linear(input_dim, hidden_dim)  # 输入到隐层的权重
        self.bias = nn.Parameter(torch.zeros(hidden_dim))  # 偏置项
        self.activation = nn.Tanh()  # 激活函数

    def forward(self, u, h):
        # 液态神经元的前向传播
        # u: 当前时间步的输入
        # h: 上一时间步的隐状态
        dh = -h + self.activation(self.W_h(h) + self.W_u(u) + self.bias)  # 隐状态的微分
        h_new = h + self.dt * dh  # 更新隐状态 (欧拉法近似)
        return h_new


# ------------------------------
# 多步预测液态神经网络 (Multi-step Prediction Liquid Neural Network)
class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1, num_steps=50):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim  # 隐层维度
        self.num_layers = num_layers  # 液态层数量
        self.num_steps = num_steps  # 输入序列的时间步长度 (用于内部循环)
        self.prediction_steps = prediction_steps  # 输出的预测步长

        # 创建多个液态层
        self.liquid_cells = nn.ModuleList([
            LiquidCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, prediction_steps)  # 全连接层，从最后一个隐状态预测多个未来步长

    def forward(self, x):
        # x: 输入数据，形状为 (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        T = min(seq_len, self.num_steps)  # 实际处理的时间步数

        # 初始化每个液态层的隐状态
        h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

        # 遍历时间步
        for t in range(T):
            inp = x[:, t, :]  # 当前时间步的输入
            for i, cell in enumerate(self.liquid_cells):
                # 传递给第一层的输入是原始输入，后续层的输入是前一层的隐状态
                h[i] = cell(inp if i == 0 else h[i - 1], h[i])

        out = self.fc(h[-1])  # 使用最后一层的最后一个时间步的隐状态进行预测
        return out

    def predict_speed(self, x):
        # 预测速度的接口
        return self.forward(x)


# ------------------------------
# 权重初始化 (Weight Initialization)
def initialize_weights(model):
    # 对模型的权重进行初始化
    for name, param in model.named_parameters():
        if "weight" in name:  # 如果是权重参数
            if param.dim() >= 2:  # 对于二维及以上的权重矩阵
                nn.init.xavier_uniform_(param)  # 使用 Xavier 均匀分布初始化
            else:  # 对于一维权重 (例如某些偏置被误认为权重时)
                nn.init.uniform_(param)  # 使用均匀分布初始化
        elif "bias" in name:  # 如果是偏置参数
            nn.init.constant_(param, 0)  # 初始化为 0


# ------------------------------
# 训练函数 (Training function)
def train_model(model, train_loader, optimizer, criterion, num_epochs=30, dataset_name=""):
    model.train()  # 设置模型为训练模式
    print(f"--- 开始为数据集: {dataset_name} 训练模型 ---")  # Starting to train model for dataset: ...
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed in train_loader:
            optimizer.zero_grad()  # 清空梯度
            pred = model.predict_speed(batch_data)  # 模型预测
            loss = criterion(pred, batch_speed)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            total_loss += loss.item()
        print(
            f"数据集 {dataset_name} - 轮次 [{epoch + 1}/{num_epochs}], 平均损失: {total_loss / len(train_loader):.4f}")
    print(f"--- 数据集: {dataset_name} 模型训练完成 ---")  # Model training completed for dataset: ...
    return model


# ------------------------------
# 评估函数 (Evaluation function)
def evaluate_model(model, test_loader,current_prediction_steps ,dataset_name="", results_dir=""):
    model.eval()  # 设置模型为评估模式
    mse_loss = nn.MSELoss()  # 均方误差损失
    total_mse = 0
    all_predicted = []  # 存储所有预测值
    all_true = []  # 存储所有真实值

    with torch.no_grad():  # 不计算梯度
        for batch_idx, (batch_data, batch_speed) in enumerate(test_loader):
            predicted_speed = model.predict_speed(batch_data)  # 模型预测
            loss = mse_loss(predicted_speed, batch_speed)  # 计算损失
            total_mse += loss.item()
            all_predicted.append(predicted_speed)
            all_true.append(batch_speed)

    # 合并所有批次的预测值与真实值
    all_predicted = torch.cat(all_predicted).cpu().numpy()  # shape: (N, prediction_steps)
    all_true = torch.cat(all_true).cpu().numpy()  # shape: (N, prediction_steps)

    # 计算整体评估指标
    mse_val = np.mean((all_predicted - all_true) ** 2)  # 均方误差
    rmse_val = np.sqrt(mse_val)  # 均方根误差
    mae_val = np.mean(np.abs(all_predicted - all_true))  # 平均绝对误差

    print(f"--- 数据集: {dataset_name} 评估结果 ---")  # Evaluation results for dataset: ...
    print(f"速度预测评估指标:\nMSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")

    # 绘制部分样本的对比曲线
    step = current_prediction_steps  # 每隔 step 个样本取一个用于绘图
    num_samples_to_plot = 30  # 总共绘制的样本数量
    true_concat = []
    pred_concat = []

    for i in range(num_samples_to_plot):
        idx = i * step
        if idx >= all_true.shape[0]:  # 防止索引越界
            break
        # 拼接每个样本的真实值和预测值序列
        true_concat.extend(all_true[idx])
        pred_concat.extend(all_predicted[idx])

    true_concat = np.array(true_concat)
    pred_concat = np.array(pred_concat)

    plt.figure(figsize=(12, 8))
    plt.plot(true_concat, linestyle='--', marker='o', label='真实值 (True)')
    plt.plot(pred_concat, linestyle='-', marker='x', label='预测值 (Predicted)')

    title_string = f'真实值 vs 预测值 (速度) - 数据集: {dataset_name}\n(绘制{min(num_samples_to_plot, (all_true.shape[0] + step - 1) // step)}个样本, 采样间隔{step}, 总计{len(true_concat)}个时间点)'
    plt.title(title_string)
    plt.xlabel('时间点索引 (Time Point Index)')
    plt.ylabel('速度 (m/s) (Speed (m/s))')
    plt.legend()
    plt.grid()

    plot_filename = os.path.join(results_dir, f"{dataset_name}_speed_comparison.png")
    plt.savefig(plot_filename)  # 保存图像
    print(f"速度对比图已保存至 {plot_filename}")  # Speed comparison plot saved to ...
    plt.close()  # 关闭图像，释放内存

    return mse_val, rmse_val, mae_val  # 返回评估指标


# ------------------------------
# 多步预测前车未来位置计算并保存 (Compute future positions of preceding vehicle and save)
def compute_future_positions_and_save(model,
                                      test_model_input_data,  # 用于模型预测的测试集输入特征
                                      original_raw_data_for_file,  # 当前数据集的完整原始raw_data
                                      original_label_data_for_file,  # 当前数据集的完整原始label_data
                                      idx_start_of_test_in_sampled,  # 测试集在采样数据中的起始索引
                                      dt=0.1,  # 时间间隔
                                      output_excel_file="pred_positions.xlsx",
                                      dataset_name=""):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        # pred_speeds 形状: (N_test, prediction_steps)
        pred_speeds = model.predict_speed(test_model_input_data).cpu().numpy()

    N_test, steps_predicted = pred_speeds.shape  # N_test: 测试样本数, steps_predicted: 预测的步长数

    # 这些索引是针对 original_raw_data_for_file 和 original_label_data_for_file 的
    # 它们指向采样数据中的测试集部分所对应的原始数据位置
    indices_for_original_data = np.arange(idx_start_of_test_in_sampled, idx_start_of_test_in_sampled + N_test)

    # 当前时刻速度 (m/s)，从模型输入 test_model_input_data 的最后一个时间步获取
    # test_model_input_data 的特征0是速度
    curr_speed_mps = test_model_input_data[:, -1, 0].cpu().numpy()

    # 当前时刻前车位置 (m)，从 original_raw_data_for_file 获取 (原始特征索引7是前车位置，单位ft)
    curr_pos_ft = original_raw_data_for_file[indices_for_original_data, -1, 7].cpu().numpy()
    curr_pos_m = curr_pos_ft * 0.3048  # ft -> m

    # 真实的前车未来位置 (m)，从 original_label_data_for_file 获取 (标签特征索引5是前车未来位置，单位ft)
    true_future_pos_ft = original_label_data_for_file[indices_for_original_data, :, 5].cpu().numpy()
    true_future_pos_m = true_future_pos_ft * 0.3048  # ft -> m

    # 真实的未来车速 (m/s)，从 original_label_data_for_file 获取 (标签特征索引4是未来车速，单位ft/s)
    # 注意：这里单位转换在主函数中已对 target_speeds (即 train_real_speed_all_steps) 做过，所以label已经是m/s
    # 但如果label_data[:,:,4]原始单位是ft/s，则需转换。代码中是 train_real_speed_all_steps = original_label_data[:, :, 4]
    # 然后 train_real_speed_all_steps *= 0.3048。所以这里 original_label_data[:,:,4] 若未被修改，仍是ft/s
    # 为了安全，假设 original_label_data[:,:,4] 是原始 ft/s 单位
    true_future_speeds_fps = original_label_data_for_file[indices_for_original_data, :, 4].cpu().numpy()
    true_future_speeds_mps = true_future_speeds_fps * 0.3048  # ft/s -> m/s

    # 初始化存储所有样本的预测位置
    pred_future_pos_m = np.zeros((N_test, steps_predicted))

    # 对每个测试样本单独进行未来位置的递推计算
    for i in range(N_test):
        prev_speed = curr_speed_mps[i]  # 当前样本的初始车速 (t=0)
        prev_pos = curr_pos_m[i]  # 当前样本的初始位置 (t=0)

        for k in range(steps_predicted):  # 遍历预测的每一个未来时间步
            v_pred_step_k = pred_speeds[i, k]  # 模型对当前样本在未来第k步的预测车速

            # 使用牛顿运动学公式: pos_new = pos_old + v_old * dt + 0.5 * a * dt^2
            # 加速度 a = (v_pred_step_k - prev_speed) / dt
            acceleration = (v_pred_step_k - prev_speed) / dt
            displacement = prev_speed * dt + 0.5 * acceleration * dt ** 2  # 位移

            current_pos = prev_pos + displacement  # 更新位置
            pred_future_pos_m[i, k] = current_pos  # 存储预测位置

            prev_speed = v_pred_step_k  # 更新车速以用于下一个子步的计算
            prev_pos = current_pos  # 更新位置以用于下一个子步的计算

    # 误差评估 (位置预测)
    # 确保 true_future_pos_m 和 pred_future_pos_m 的步长维度一致
    # true_future_pos_m 可能比 pred_future_pos_m 的步长更多或更少，取决于label_data的原始构造
    # 我们应该只比较到 steps_predicted 那么多步
    if true_future_pos_m.shape[1] < steps_predicted:
        print(f"警告: 真实位置标签的步长 ({true_future_pos_m.shape[1]}) 小于预测步长 ({steps_predicted})。将截断比较。")
        current_steps_to_compare = true_future_pos_m.shape[1]
    else:
        current_steps_to_compare = steps_predicted

    rmse_p = np.sqrt(np.mean(
        (pred_future_pos_m[:, :current_steps_to_compare] - true_future_pos_m[:, :current_steps_to_compare]) ** 2))
    # 避免除以零或非常小的值导致MAPE过大
    mask = true_future_pos_m[:, :current_steps_to_compare] != 0
    mape_p = np.mean(np.abs((pred_future_pos_m[:, :current_steps_to_compare][mask] -
                             true_future_pos_m[:, :current_steps_to_compare][mask]) /
                            true_future_pos_m[:, :current_steps_to_compare][mask])) * 100

    print(f"--- 数据集: {dataset_name} 未来位置预测评估 ---")  # Future Position Prediction Evaluation for dataset: ...
    print(f"位置预测误差 -- RMSE: {rmse_p:.4f} m, MAPE: {mape_p:.2f}% (比较了 {current_steps_to_compare} 步)")

    # 保存到 Excel
    data_dict = {}
    for k in range(steps_predicted):  # 保存所有模型预测的步长
        data_dict[f"Pred_Speed_step{k + 1}(m/s)"] = pred_speeds[:, k]
        data_dict[f"Pred_Pos_step{k + 1}(m)"] = pred_future_pos_m[:, k]
        if k < true_future_pos_m.shape[1]:  # 确保真实值索引不越界
            data_dict[f"True_Pos_step{k + 1}(m)"] = true_future_pos_m[:, k]
        if k < true_future_speeds_mps.shape[1]:
            data_dict[f"True_Speed_step{k + 1}(m/s)"] = true_future_speeds_mps[:, k]

    df_pos = pd.DataFrame(data_dict)
    sheet_name = dataset_name  # 使用数据集名作为 sheet 名称 (例如 "data_5")

    # 如果文件存在则在其基础上添加新 sheet，否则新建文件
    # 'a'模式表示追加, 'w'模式表示写入(会覆盖)
    # if_sheet_exists='replace' 可以用来替换同名sheet，需要 pandas >= 1.3.0 和 openpyxl
    try:
        with pd.ExcelWriter(output_excel_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            df_pos.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:  # 如果文件不存在，首次以 'w' 模式创建
        with pd.ExcelWriter(output_excel_file, engine="openpyxl", mode="w") as writer:
            df_pos.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"{dataset_name} 的位置预测结果已保存至 '{output_excel_file}' 的 '{sheet_name}' 工作表。")
    # Position predictions for {dataset_name} saved to '{output_excel_file}' in sheet '{sheet_name}'.

    return rmse_p, mape_p  # 返回位置预测的评估指标


# ------------------------------
# 存储和保存所有数据集的评估指标 (Store and save evaluation metrics for all datasets)
all_datasets_metrics_summary = []  # 用于存储所有数据集的评估指标字典列表


def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae, pos_rmse, pos_mape):
    # 将单个数据集的评估指标存入列表
    metrics = {
        "数据集 (Dataset)": dataset_name,
        "速度MSE (Speed_MSE)": speed_mse,
        "速度RMSE (Speed_RMSE)": speed_rmse,
        "速度MAE (Speed_MAE)": speed_mae,
        "位置RMSE (m) (Position_RMSE_m)": pos_rmse,
        "位置MAPE (%) (Position_MAPE_percent)": pos_mape
    }
    all_datasets_metrics_summary.append(metrics)


def save_all_metrics_to_csv(filepath="evaluation_summary.csv"):
    # 将所有数据集的评估指标汇总保存到CSV文件
    if not all_datasets_metrics_summary:
        print("没有评估指标可以保存。")  # No metrics to save.
        return
    df_metrics = pd.DataFrame(all_datasets_metrics_summary)
    df_metrics.to_csv(filepath, index=False, encoding='utf-8-sig')  # utf-8-sig 确保中文在Excel中正确显示
    print(f"所有数据集的评估指标汇总已保存至 {filepath}")  # All evaluation metrics saved to {filepath}


# ------------------------------
# 主函数 (Main function)
if __name__ == "__main__":
    torch.manual_seed(42)  # 设置 PyTorch 随机种子以保证结果可复现
    # np.random.seed(42)  # 设置 NumPy 随机种子

    # 获取指定目录下所有的 .mat 文件
    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    print(data_files)
    if not data_files:
        print(f"在目录 {DATA_DIR} 中未找到 .mat 文件。")  # No .mat files found in the directory.
        exit()

    # 定义位置预测结果的Excel文件路径
    position_predictions_excel_path = os.path.join(RESULTS_DIR, "pred_positions_all_datasets1128.xlsx")
    # 如果旧的汇总Excel文件存在，可以选择删除以重新开始，或者让程序追加/替换sheet
    # 当前的 compute_future_positions_and_save 实现会替换同名sheet或追加新sheet

    print(f"找到以下数据集文件: {data_files}")  # Found the following dataset files:

    # 遍历每个找到的数据文件
    for data_file_path in data_files:
        dataset_filename = os.path.basename(data_file_path)  # 获取文件名，例如 "data_5.mat"
        dataset_name_clean = dataset_filename.replace(".mat", "")  # 去掉 .mat 后缀，例如 "data_5"

        print(f"\n==================== 开始处理数据集: {dataset_filename} ====================")
        # Processing dataset: ...

        # 1. 加载数据 (Load data)
        try:
            data_mat = sio.loadmat(data_file_path)
            # 原始的、未经处理的完整数据
            original_raw_data = torch.tensor(data_mat['train_data'], dtype=torch.float32)
            original_label_data = torch.tensor(data_mat['lable_data'], dtype=torch.float32)
        except Exception as e:
            print(f"加载数据文件 {dataset_filename} 失败: {e}")  # Failed to load data file ...
            continue  # 跳过此文件

        # 2. 提取多步速度标签并确定预测步长 (Extract multi-step speed labels and determine prediction steps)
        # 假设标签数据的第5列 (索引4) 是车辆速度
        train_real_speed_all_steps = original_label_data[:, :, 4]
        current_prediction_steps = train_real_speed_all_steps.shape[1]  # 预测步长由标签数据的第二维决定
        print(f"数据集 {dataset_filename}: 预测步长 = {current_prediction_steps}")  # Prediction steps = ...

        # 3. 构造模型输入特征 (Construct model input features)
        # 取原始数据最后50个时间步的第6和第7列特征 (索引5和6) 作为模型输入
        # 假设这些特征是前车速度和加速度，并且所有数据集结构一致
        if original_raw_data.shape[1] < 50:
            print(
                f"数据集 {dataset_filename} 的时间步数 ({original_raw_data.shape[1]}) 小于50，无法提取足够的输入序列。跳过此数据集。")
            continue
        model_input_features = original_raw_data[:, -50:, [0, 1, 2, 3, 5]].clone() * 0.3048
        target_speeds = train_real_speed_all_steps.clone()  # 模型要预测的目标速度

        # 4. 单位转换 (Unit conversion): ft/s -> m/s (假设原始单位是英尺/秒)
        # model_input_features[:, :, 0] *= 0.3048  # 特征1 (例如，速度)
        # model_input_features[:, :, 1] *= 0.3048  # 特征2 (例如，加速度)
        target_speeds *= 0.3048  # 目标速度

        # 5. 数据抽样 (Data sampling): 取前10%的样本用于实验
        total_samples_in_file = model_input_features.shape[0]
        sample_size = int(total_samples_in_file * 1)  # 10%的样本量

        if sample_size == 0 and total_samples_in_file > 0:
            sample_size = 1  # 如果有数据但10%为0，至少取1个样本
        if sample_size == 0:
            print(
                f"数据集 {dataset_filename} 数据量不足以进行抽样。跳过此数据集。")  # Not enough data to sample. Skipping.
            continue

        # 经过抽样的数据，用于后续的训练和测试
        sampled_input_features = model_input_features[:sample_size]
        sampled_target_speeds = target_speeds[:sample_size]
        print(f"数据集 {dataset_filename}: 原始样本数 {total_samples_in_file}, 抽取样本数 {sample_size}")

        # 6. 数据检查 (Check data for NaN/Inf)
        check_data(sampled_input_features, f"采样后的输入特征 ({dataset_filename})")
        check_data(sampled_target_speeds, f"采样后的目标速度 ({dataset_filename})")

        # 7. 划分训练集和测试集 (Split into training and test sets) - 从抽样数据中划分
        num_sampled_data = sampled_input_features.shape[0]
        train_split_size = int(num_sampled_data * 0.2)  # 80% 作为训练集

        if train_split_size == 0 and num_sampled_data > 0:
            train_split_size = 1  # 确保训练集至少有一个样本

        # 确保测试集也有数据（如果可能）
        if train_split_size == num_sampled_data and num_sampled_data > 1:
            train_split_size = num_sampled_data - 1

        if num_sampled_data <= 1:  # 如果总样本太少，无法有效划分
            print(
                f"数据集 {dataset_filename} 抽样后数据过少 ({num_sampled_data})，无法有效划分训练/测试集。跳过此数据集。")
            # Insufficient data after sampling for train/test split. Skipping.
            continue

        X_train = sampled_input_features[:train_split_size]
        y_train = sampled_target_speeds[:train_split_size]
        X_test = sampled_input_features[train_split_size:]
        y_test = sampled_target_speeds[train_split_size:]

        print(
            f"数据集 {dataset_filename}: 抽样数据 {num_sampled_data} -> 训练集 {X_train.shape[0]}, 测试集 {X_test.shape[0]}")

        # 创建 DataLoader
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=32, shuffle=True  # 训练时打乱数据
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, y_test),
            batch_size=32, shuffle=False  # 测试时不打乱
        )

        # 8. 模型配置 (Model configuration)
        input_dim = X_train.shape[2]  # 模型输入特征的数量
        hidden_dim = 128  # 液态神经元隐层维度
        model_input_seq_len = X_train.shape[1]  # 模型输入的序列长度 (这里是50)

        model = LiquidNeuralNetwork(input_dim, hidden_dim, current_prediction_steps,
                                    num_layers=1, num_steps=model_input_seq_len)
        initialize_weights(model)  # 初始化模型权重
        criterion = nn.MSELoss()  # 损失函数：均方误差
        optimizer = optim.Adam(model.parameters(), lr=5e-4)  # 优化器：Adam

        # 9. 训练模型 (Train model)
        model = train_model(model, train_loader, optimizer, criterion, num_epochs=20, dataset_name=dataset_name_clean)

        # 10. 评估模型并保存结果 (Evaluate model and save results)
        if X_test.shape[0] > 0:  # 只有当测试集非空时才进行评估
            speed_mse, speed_rmse, speed_mae = evaluate_model(model, test_loader, current_prediction_steps, dataset_name=dataset_name_clean,
                                                              results_dir=RESULTS_DIR)

            # 计算并保存未来位置预测
            # 注意: original_raw_data 和 original_label_data 是当前文件完整的原始数据
            # train_split_size 是在 *抽样后* 的数据中，训练集的样本数量，也即测试集在抽样数据中的起始索引
            pos_rmse, pos_mape = compute_future_positions_and_save(
                model,
                X_test,  # 测试集的模型输入特征
                original_raw_data,  # 整个文件的原始 raw_data
                original_label_data,  # 整个文件的原始 label_data
                train_split_size,  # 测试集在抽样数据中的起始索引
                dt=0.1,
                output_excel_file=position_predictions_excel_path,
                dataset_name=dataset_name_clean
            )
            # 存储当前数据集的评估指标
            store_dataset_metrics(dataset_name_clean, speed_mse, speed_rmse, speed_mae, pos_rmse, pos_mape)
        else:
            print(f"数据集 {dataset_filename} 的测试集为空，跳过评估和位置预测。")
            # Test set is empty, skipping evaluation and position prediction.
            store_dataset_metrics(dataset_name_clean, float('nan'), float('nan'), float('nan'), float('nan'),
                                  float('nan'))

        print(f"==================== 数据集: {dataset_filename} 处理完成 ====================")
        # Dataset processing completed.

    # 11. 所有数据集处理完毕后，保存汇总的评估指标 (After processing all datasets, save summary metrics)
    summary_metrics_csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_all_datasets1128.csv")
    save_all_metrics_to_csv(summary_metrics_csv_path)

    print("\n所有数据集处理完毕。")  # All datasets processed.

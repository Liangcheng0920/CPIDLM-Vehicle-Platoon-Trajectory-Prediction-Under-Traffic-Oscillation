import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import seaborn as sns  # 导入更美观的绘图库
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --- 数据检查函数 ---
def check_data(data, name="data"):
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# --- 定义混合模型 ---
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(HybridIDMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 6)  # IDM参数：[v_desired, T, a_max, b_safe, delta, s0]
        self.softplus = nn.Softplus()
        self.delta_t = 0.1  # 时间步长

    def forward(self, x):
        out, _ = self.lstm(x)
        params = self.fc(out[:, -1, :])
        params = self.softplus(params)  # 确保参数为正
        return params

    def predict_speed(self, x, s_safe):
        params = self.forward(x)
        v_n = x[:, -1, 0]
        delta_v = x[:, -1, 2]
        v_desired, T, a_max, b_safe, delta, s0 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:,
                                                                                                         4], params[:,
                                                                                                             5]
        s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe))
        s_star = torch.clamp(s_star, min=0)
        v_follow = v_n + self.delta_t * a_max * (1 - (v_n / v_desired) ** delta - (s_star / s_safe) ** 2)
        predicted_speed = torch.clamp(v_follow, min=0)
        return predicted_speed, params


# --- 初始化权重和偏置 ---
def initialize_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)


# --- 模型训练 ---
def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed, batch_s_safe in train_loader:
            optimizer.zero_grad()
            predicted_speed, _ = model.predict_speed(batch_data, batch_s_safe)
            loss = criterion(predicted_speed, batch_speed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    return model


# --- 模型评估 ---
# --- 模型评估 ---
def evaluate_model(model, test_loader):
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    all_predicted = []
    all_true = []
    all_predicted_params = [] # <--- Add this list to store all predicted parameters

    with torch.no_grad():
        for batch_idx, (batch_data, batch_speed, batch_s_safe) in enumerate(test_loader):
            predicted_speed, params = model.predict_speed(batch_data, batch_s_safe)
            loss = mse_loss(predicted_speed, batch_speed)
            total_mse += loss.item()
            all_predicted.append(predicted_speed)
            all_true.append(batch_speed)
            all_predicted_params.append(params) # <--- Append parameters for each batch

            # ... (rest of your print statements for parameters, if desired)

    mse = total_mse / len(test_loader)
    rmse = torch.sqrt(torch.tensor(mse))
    mae = torch.mean(torch.abs(torch.cat(all_predicted) - torch.cat(all_true))).item()
    print(f"Evaluation Metrics:\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.plot(torch.cat(all_true)[:100].cpu().numpy(), label='True Speed', linestyle='--', marker='o')
    plt.plot(torch.cat(all_predicted)[:100].cpu().numpy(), label='Predicted Speed', linestyle='-', marker='x')
    plt.title('True vs Predicted Speed (Hybrid IDM-LSTM)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()

    # Concatenate all predicted parameters and return them
    return torch.cat(all_predicted_params, dim=0).cpu().numpy() # <--- Return the concatenated parameters


def save_predictions_to_csv(model, test_loader, output_file="predictions_param.csv"):
    """
    保存测试数据的预测结果到CSV文件。

    :param model: 训练好的模型
    :param test_loader: 测试数据加载器
    :param output_file: 输出CSV文件的路径
    """
    model.eval()
    all_true_speeds = []
    all_predicted_speeds = []
    all_params = []

    with torch.no_grad():
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
        "True Speed": all_true_speeds,
        "Predicted Speed": all_predicted_speeds,
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

# 基于预测车速，推算下一时刻 Y 坐标与车头间距，并保存结果
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
    model.eval()
    with torch.no_grad():
        # pred_speed = model.predict_speed(test_data).squeeze().cpu().numpy()
        pred_speed_tensor, params_tensor = model.predict_speed(test_data, test_s_safe)
        # 对速度张量进行 squeeze 并转到 CPU numpy
        pred_speed = pred_speed_tensor.squeeze().cpu().numpy()
    true_speed = test_real_speed.cpu().numpy()

    N_test = test_data.shape[0]
    idx = np.arange(train_size, train_size + N_test)

    current_Y_ft = raw_data[idx, -1, 4].numpy()  # 第四维索引
    current_Y_ft *= 0.3048
    current_speed_m = test_data[:, -1, 0].numpy()

    true_Y_ft = label_data[idx, -1, 3].numpy()
    true_Y_ft *= 0.3048
    true_spacing_m = label_data[idx, -1, 1].numpy()
    true_spacing_m *= 0.3048
    # 根据预测车速计算位移（这里采用等加速度运动公式近似）
    disp_m = current_speed_m * dt + 0.5 * ((pred_speed - current_speed_m)/dt) * dt**2
    disp_ft = disp_m
    pred_Y_ft = current_Y_ft + disp_ft

    pred_Y_m = pred_Y_ft
    true_Y_m = true_Y_ft
    pred_spacing_m = (true_Y_ft - pred_Y_ft) + true_spacing_m

    rmse_Y = np.sqrt(np.mean((pred_Y_m - true_Y_m) ** 2))
    mape_Y = np.mean(np.abs((pred_Y_m - true_Y_m) / true_Y_m)) * 100
    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m) ** 2))
    mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m) / true_spacing_m)) * 100

    print(f"Position Error    -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
    print(f"Spacing  Error    -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")

    df = pd.DataFrame({
        "Pred Speed (m/s)":      pred_speed,
        "True Speed (m/s)":      true_speed,
        "Predicted Y (m)":       pred_Y_m,
        "True Y (m)":            true_Y_m,
        "Predicted Spacing (m)": pred_spacing_m,
        "True Spacing (m)":      true_spacing_m,
    })
    sheet_name = "LSTM-IDM_1"  # 指定新工作表名称

    # 如果文件存在则在其基础上添加新 sheet，否则新建文件
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a" if os.path.exists(output_file) else "w") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"New model results saved to '{output_file}' in sheet '{sheet_name}'.")


# def plot_parameter_distributions(predicted_params):
#     """
#     绘制LSTM-IDM模型预测的六个参数的分布图。
#
#     :param predicted_params: 包含所有测试样本预测参数的numpy数组，形状为 (num_samples, 6)。
#     """
#     sns.set_palette("viridis") # 设置一个美观的调色板
#     sns.set_style("whitegrid") # 设置白色网格的风格
#     params_names = [r'$v_{desired}$ (m/s)', r'$T$ (s)', r'$a_{max}$ (m/s$^2$)',
#                     r'$b_{safe}$ (m/s$^2$)', r'$\delta$', r'$s_0$ (m)']
#     num_params = predicted_params.shape[1]
#
#     fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
#     axes = axes.flatten() # 将2x3的axes数组展平，方便索引
#
#     for i in range(num_params):
#         sns.histplot(predicted_params[:, i], ax=axes[i], kde=True)
#         axes[i].set_title(f'Distribution of {params_names[i]}', fontsize=14)
#         axes[i].set_xlabel(params_names[i], fontsize=12)
#         axes[i].set_ylabel('Frequency', fontsize=12)
#         axes[i].tick_params(axis='both', which='major', labelsize=10)
#
#     plt.tight_layout()
#     plt.suptitle('Distribution of Predicted IDM Parameters on Test Set', fontsize=16, y=1.02)
#     plt.show()



# def plot_parameter_distributions(predicted_params):
#     """
#     绘制LSTM-IDM模型预测的六个参数的分布图。
#
#     :param predicted_params: numpy数组，形状 (num_samples, 6)
#     """
#     # 期刊级别设置
#     sns.set_style("whitegrid")
#     plt.rcParams.update({
#         "figure.dpi": 300,
#         "font.size": 12,
#         "axes.titlesize": 14,
#         "axes.labelsize": 12,
#         "legend.fontsize": 12,
#         "xtick.labelsize": 10,
#         "ytick.labelsize": 10,
#     })
#
#     params_names = [
#         r'$v_{\mathrm{desired}}\,(\mathrm{m/s})$',
#         r'$T\,(\mathrm{s})$',
#         r'$a_{\max}\,(\mathrm{m/s^2})$',
#         r'$b_{\mathrm{safe}}\,(\mathrm{m/s^2})$',
#         r'$\delta$',
#         r'$s_0\,(\mathrm{m})$'
#     ]
#     num_params = predicted_params.shape[1]
#
#     fig, axes = plt.subplots(2, 3, figsize=(18, 10))
#     axes = axes.flatten()
#
#     for i in range(num_params):
#         ax = axes[i]
#         # 直方图：灰色填充，半透明；边缘黑线
#         sns.histplot(
#             predicted_params[:, i],
#             bins=30,
#             stat='density',
#             ax=ax,
#             color='lightgray',
#             edgecolor='black',
#             alpha=0.6
#         )
#         # 密度曲线：深蓝色，宽线
#         sns.kdeplot(
#             predicted_params[:, i],
#             ax=ax,
#             lw=2,
#             linestyle='-',
#             color='#1f3b73'
#         )
#         ax.set_title(f'Distribution of {params_names[i]}')
#         ax.set_xlabel(params_names[i])
#         ax.set_ylabel('Density')
#         ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
#
#     # 全局布局调整和保存
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.suptitle('Distribution of Predicted IDM Parameters on Test Set', fontsize=16, y=1.02)
#     plt.savefig('idm_params_distribution.png', dpi=300, bbox_inches='tight')
#     plt.show()

def plot_parameter_distributions(predicted_params):
    """
    绘制 LSTM-IDM 模型对测试集预测的六个 IDM 参数分布：
    [v_desired, T, a_max, b_safe, delta, s0]
    """
    # ---- 样式设置 ----
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11

    # 参数名称（含单位）
    params_names = [
        r'$v_{\mathrm{desired}}$ (m/s)',
        r'$T$ (s)',
        r'$a_{\mathrm{max}}$ (m/s$^2$)',
        r'$b_{\mathrm{safe}}$ (m/s$^2$)',
        r'$\delta$',
        r'$s_{0}$ (m)'
    ]

    # 配色：直方图蓝色（半透明），密度曲线橙色
    # hist_color = "#4C72B0"   # 蓝色调
    # kde_color  = "#DD8452"   # 橙色调
    hist_color = "#1f77b4"   # 蓝色调
    kde_color  = "#ff7f0e"   # 橙色调
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        sns.histplot(
            predicted_params[:, i],
            ax=ax,
            stat='density',
            bins=30,
            color=hist_color,
            alpha=0.6,
            edgecolor='none'
        )
        sns.kdeplot(
            predicted_params[:, i],
            ax=ax,
            color=kde_color,
            lw=2
        )
        ax.set_title(f'Distribution of {params_names[i]}')
        ax.set_xlabel(params_names[i])
        ax.set_ylabel('Density')
        # 显示全比例，无截断
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', which='major', labelsize=10)

    # 整体标题
    # fig.suptitle('Distribution of Predicted IDM Parameters on Test Set', fontsize=16)
    # plt.show()

        # 全局布局调整和保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Distribution of Predicted IDM Parameters on Test Set', fontsize=16, y=1.02)
    plt.savefig('idm_params_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- 主函数 ---
if __name__ == "__main__":
    torch.manual_seed(42)

    # 加载数据
    data = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')  # 请替换为你的实际数据文件名
    raw_data = torch.tensor(data['train_data'], dtype=torch.float32)  # 假设形状为 (样本数, 时间步长, 3)
    label_data = torch.tensor(data['lable_data'], dtype=torch.float32)
    # 注意：原代码采用 train_data = raw_data[:, -5:, :] 仅取最后5个时间步，
    # 若希望τ的取值范围达到0~50，则历史数据步长应足够长。这里暂保持原代码结构。
    print(raw_data.shape)
    train_data = raw_data[:, -50:, [0, 1, 2, 3, -1]]  # 第一列为自车车速、第二列为前车距离、第三列为速度差、第四列为自车加速度、第五列为前车速度

    train_real_speed1 = torch.tensor(data['lable_data'], dtype=torch.float32)

    print(train_real_speed1.shape)
    train_real_speed = train_real_speed1[:, -1, 0]

    # # 加载数据
    # data = sio.loadmat('NG_data_onesec.mat')  # 替换为你的实际文件名
    #
    # raw_data = torch.tensor(data['train_data'], dtype=torch.float32)  # shape: (1000, 20, 3)
    # #修改观测步长
    # # 原始数据切片方式（保留时间顺序）
    # train_data = raw_data[:, -5:, :]  # 直接选取最后两个时间步（形状变为[3670,2,3]）
    # train_real_speed = torch.tensor(data['train_real_speed'].flatten(), dtype=torch.float32)  # shape: (1000,)

    # 修改部分：从train_data中提取最后一帧的第2列数据（索引1），并保持形状一致
    train_s_safe = torch.tensor(train_data[:, -1, 1].clone(), dtype=torch.float32)  # 关键修改点

    # 验证处理后的时序顺序
    print("处理后时序示例（第一个样本）:")
    print(train_data[0, :, 0])  # 假设第一列为速度特征
    # 打印输出，检查数据形状
    print(f"train_data shape: {train_data.shape}")
    print(f"train_real_speed shape: {train_real_speed.shape}")
    print(f"train_s_safe shape: {train_s_safe.shape}")

    # 数据预处理：单位转换（例如ft/s转为m/s，ft转为m）
    train_data[:, :, 0] *= 0.3048
    train_data[:, :, 1] *= 0.3048
    train_data[:, :, 2] *= -0.3048
    train_data[:, :, 3] *= 0.3048
    train_data[:, :, 4] *= 0.3048
    train_real_speed *= 0.3048
    train_s_safe *= 0.3048  # 距离从 ft 转换为 m
    # 选取前 10% 的数据
    total_samples = train_data.shape[0]
    sample_size = int(total_samples * 0.1)
    train_data = train_data[:sample_size]
    train_real_speed = train_real_speed[:sample_size]
    train_s_safe = train_s_safe[:sample_size]

    # 检查数据
    check_data(train_data, "train_data")
    check_data(train_real_speed, "train_real_speed")
    check_data(train_s_safe, "train_s_safe")

    # 划分训练集和测试集
    dataset_size = train_data.shape[0]
    train_size = int(dataset_size * 0.8)
    train_x = train_data[:train_size]
    test_x = train_data[train_size:]
    train_y = train_real_speed[:train_size]
    test_y = train_real_speed[train_size:]
    train_data, test_data = train_data[:train_size], train_data[train_size:]
    train_real_speed, test_real_speed = train_real_speed[:train_size], train_real_speed[train_size:]
    train_s_safe, test_s_safe = train_s_safe[:train_size], train_s_safe[train_size:]

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_data, train_real_speed, train_s_safe)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_real_speed, test_s_safe)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型、权重、损失函数、优化器
    input_dim = train_data.shape[2]  # 输入特征数（3：速度、距离、速度差）
    hidden_dim = 128  # 增大隐藏单元数
    num_layers = 1  # 增加LSTM层数
    model = HybridIDMModel(input_dim, hidden_dim, num_layers)
    initialize_weights(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 适当调整学习率

    # 训练模型
    num_epochs = 100 # 增加训练轮数
    model = train_model(model, train_loader, optimizer, criterion, num_epochs)

    # 评估模型并获取预测参数
    predicted_params = evaluate_model(model, test_loader)

    # 绘制预测参数的分布图
    plot_parameter_distributions(predicted_params)

    output_file = "test_predictions_param.csv"
    save_predictions_to_csv(model, test_loader, output_file=output_file)
    # compute_position_and_spacing_and_save(
    #     model,
    #     test_x,
    #     test_y,
    #     raw_data,
    #     label_data,
    #     train_size,
    #     test_s_safe,
    #     dt=0.1,
    #     output_file="predictions_extended.xlsx"
    # )


import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ------------------------------
# 数据检查函数
def check_data(data, name="data"):
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# ------------------------------
# 液态单元（Liquid Cell）
# 该单元模拟连续时间动态，通过欧拉积分进行状态更新：
# h_new = h + dt * (-h + tanh(W_h * h + W_u * u + b))
class LiquidCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        """
        :param input_dim: 当前输入特征数（对于第一层为原始输入，对于后续层为上一层的隐藏状态）
        :param hidden_dim: 隐藏状态维度
        :param dt: 时间步长，用于欧拉积分更新
        """
        super(LiquidCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_u = nn.Linear(input_dim, hidden_dim)
        # 使用单独的偏置参数
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.activation = nn.Tanh()

    def forward(self, u, h):
        # u: (batch, input_dim); h: (batch, hidden_dim)
        dh = -h + self.activation(self.W_h(h) + self.W_u(u) + self.bias)
        h_new = h + self.dt * dh
        return h_new


# ------------------------------
# 液态神经网络模型（Liquid Neural Network）
# 利用液态单元沿时间步迭代更新隐藏状态，最终输出预测结果。
class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_steps=50, output_dim=1):
        """
        :param input_dim: 输入特征数
        :param hidden_dim: 隐藏状态维度
        :param num_layers: 液态层数（堆叠多个液态单元）
        :param num_steps: 序列中用于更新的时间步数（一般设为序列长度）
        :param output_dim: 输出维度（例如预测车速为1维）
        """
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_steps

        # 构建多个液态单元，第一个单元输入为原始输入，其余单元输入为前一层的隐藏状态
        self.liquid_cells = nn.ModuleList([
            LiquidCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # 最后通过全连接层预测输出
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        :param x: 输入数据，形状为 (batch_size, seq_len, input_dim)
        :return: 预测结果，形状为 (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        T = min(seq_len, self.num_steps)  # 使用序列中的前T个时间步进行状态更新

        # 初始化每一层的隐藏状态为零向量
        h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

        # 沿时间步迭代更新隐藏状态
        for t in range(T):
            input_t = x[:, t, :]  # 当前时间步输入
            for i, cell in enumerate(self.liquid_cells):
                # 第一层的输入为原始输入，后续层输入为上一层的输出
                if i == 0:
                    h[i] = cell(input_t, h[i])
                else:
                    h[i] = cell(h[i-1], h[i])
        # 使用最后一层的隐藏状态作为整体表征，并经过全连接层预测输出
        out = self.fc(h[-1])
        return out

    def predict_speed(self, x):
        return self.forward(x)


# ------------------------------
# 权重初始化函数
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
# 模型训练函数
def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed in train_loader:
            optimizer.zero_grad()
            predicted_speed = model.predict_speed(batch_data)
            loss = criterion(predicted_speed, batch_speed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    return model


# ------------------------------
# 模型评估函数
def evaluate_model(model, test_loader):
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    all_predicted = []
    all_true = []
    with torch.no_grad():
        for batch_idx, (batch_data, batch_speed) in enumerate(test_loader):
            predicted_speed = model.predict_speed(batch_data)
            loss = mse_loss(predicted_speed, batch_speed)
            total_mse += loss.item()
            all_predicted.append(predicted_speed)
            all_true.append(batch_speed)
    # 合并所有批次的预测值与真实值
    all_predicted = torch.cat(all_predicted).numpy()
    all_true = torch.cat(all_true).numpy()

    mse_val = np.mean((all_predicted - all_true) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted - all_true))
    print(f"Evaluation Metrics:\nMSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(all_true[:100], label='True Speed', linestyle='--', marker='o')
    plt.plot(all_predicted[:100], label='Predicted Speed', linestyle='-', marker='x')
    plt.title('True vs Predicted Speed (Liquid Neural Network)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()


# ------------------------------
# 保存预测结果到CSV
def save_predictions_to_csv(model, test_loader, output_file="predictions.csv"):
    model.eval()
    all_true_speeds = []
    all_predicted_speeds = []

    with torch.no_grad():
        for batch_data, batch_speed in test_loader:
            predicted_speed = model.predict_speed(batch_data)
            all_true_speeds.append(batch_speed.cpu())
            all_predicted_speeds.append(predicted_speed.cpu())

    all_true_speeds = torch.cat(all_true_speeds).numpy()
    all_predicted_speeds = torch.cat(all_predicted_speeds).numpy()

    result_data = pd.DataFrame({
        "True Speed": all_true_speeds.flatten(),
        "Predicted Speed": all_predicted_speeds.flatten(),
    })

    result_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# --- 基于预测车速，推算下一时刻 Y 坐标与车头间距，并保存结果 ---
def compute_position_and_spacing_and_save(model,
                                           test_data,
                                           test_real_speed,
                                           raw_data,
                                           label_data,
                                           train_size,
                                           dt=0.1,
                                           output_file="predictions_extended.xlsx"):
    model.eval()
    with torch.no_grad():
        pred_speed = model.predict_speed(test_data).squeeze().cpu().numpy()
    true_speed = test_real_speed.cpu().numpy()

    N_test = test_data.shape[0]
    idx = np.arange(train_size, train_size + N_test)

    current_Y_ft = raw_data[idx, -1, 4].numpy() #第四维索引
    current_Y_ft *= 0.3048
    current_speed_m = test_data[:, -1, 0].numpy()

    true_Y_ft = label_data[idx, -1, 3].numpy()
    true_Y_ft *= 0.3048
    true_spacing_m = label_data[idx, -1, 1].numpy()
    true_spacing_m *= 0.3048
    # disp_m = (pred_speed + current_speed_m) / 2 * dt
    disp_m = current_speed_m * dt + 0.5* ((pred_speed - current_speed_m)/dt) * dt**2
    disp_ft = disp_m
    pred_Y_ft = current_Y_ft + disp_ft

    pred_Y_m = pred_Y_ft
    true_Y_m = true_Y_ft
    pred_spacing_m = (true_Y_ft- pred_Y_ft)  + true_spacing_m

    rmse_Y = np.sqrt(np.mean((pred_Y_m - true_Y_m) ** 2))
    mape_Y = np.mean(np.abs((pred_Y_m - true_Y_m) / true_Y_m)) * 100
    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m) ** 2))
    mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m) / true_spacing_m)) * 100

    print(f"Position Error    -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
    print(f"Spacing  Error    -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")

    df = pd.DataFrame({
        "Pred Speed (m/s)": pred_speed.flatten(),
        "True Speed m(m/s)": true_speed.flatten(),
        "Predicted Y (m)": pred_Y_m.flatten(),
        "True Y (m)": true_Y_m.flatten(),
        "Predicted Spacing (m)": pred_spacing_m.flatten(),
        "True Spacing (m)": true_spacing_m.flatten(),
    })
    sheet_name = "LNN_1"  # 指定新工作表名称

    # 如果文件存在则在其基础上添加新 sheet，否则新建文件
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a" if os.path.exists(output_file) else "w") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"New model results saved to '{output_file}' in sheet '{sheet_name}'.")
# ------------------------------
# 主函数
if __name__ == "__main__":
    torch.manual_seed(42)
    mat = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')
    raw_data = torch.tensor(mat['train_data'], dtype=torch.float32)
    label_data = torch.tensor(mat['lable_data'], dtype=torch.float32)

    # 构造训练输入
    train_data = raw_data[:, -50:, [0, 1, 2, 3, -1]]
    train_real_speed = label_data[:, :, 0]
    # 单位转换（例如 ft/s 转为 m/s）
    train_data[:, :, 0] *= 0.3048
    train_data[:, :, 1] *= 0.3048
    train_data[:, :, 2] *= 0.3048
    train_data[:, :, 3] *= 0.3048
    train_real_speed *= 0.3048

    # --- **抽样 10%**，加快实验速度 ---
    total_samples = train_data.shape[0]
    sample_size = int(total_samples * 0.1)
    print(f"Total samples: {total_samples}, using only first {sample_size} samples for quick run.")
    train_data = train_data[:sample_size]
    train_real_speed = train_real_speed[:sample_size]

    # 划分训练/测试集（80% / 20%），此时 N=sample_size
    N = train_data.shape[0]
    train_size = int(N * 0.8)
    test_size = N - train_size

    train_x = train_data[:train_size]
    test_x = train_data[train_size:]
    train_y = train_real_speed[:train_size]
    test_y = train_real_speed[train_size:]

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=batch_size, shuffle=False
    )


    # 模型参数设置
    input_dim = train_data.shape[2]  # 输入特征数
    hidden_dim = 128  # 隐藏状态维度
    num_layers = 1  # 可根据需要调整液态层数
    num_steps = train_data.shape[1]  # 使用整个序列的时间步进行状态更新

    # 初始化液态神经网络模型
    model = LiquidNeuralNetwork(input_dim, hidden_dim, num_layers=num_layers, num_steps=num_steps, output_dim=1)
    initialize_weights(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # 训练模型
    num_epochs = 100
    model = train_model(model, train_loader, optimizer, criterion, num_epochs)

    # 评估模型
    evaluate_model(model, test_loader)
    # 若需要将预测结果保存到CSV文件，可调用：
    # save_predictions_to_csv(model, test_loader, output_file="predictions.csv")
    # --- 加载数据 ---

    # # # 位置 & 间距预测并保存
    compute_position_and_spacing_and_save(
        model,
        test_x,
        test_y,
        raw_data,
        label_data,
        train_size,
        dt=0.1,
        output_file="predictions_extended.xlsx"
    )


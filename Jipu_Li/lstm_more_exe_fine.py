import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import glob  # 用于查找文件路径
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 设置环境变量，防止因重复加载库而报错
# --- 全局路径定义 ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # 数据集存放目录
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim"  # 实验结果保存目录

# 确保结果目录存在 (Ensure results directory exists)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备是: {device}")  # 打印当前使用的设备
# ------------------------------
# 数据检查函数
def check_data(data, name="data"):
    """
    检查数据中是否包含 NaN 或 Inf 值。
    Args:
        data (torch.Tensor): 需要检查的数据。
        name (str): 数据名称，用于打印信息。
    """
    print(f"正在检查 {name} 是否包含 NaN 或 Inf 值...")
    print(f"包含 NaN: {torch.isnan(data).any().item()}")
    print(f"包含 Inf: {torch.isinf(data).any().item()}")


# ------------------------------
# 基于 LSTM 的多步预测模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1):
        """
        LSTM 模型初始化。
        Args:
            input_dim (int): 输入特征的维度。
            hidden_dim (int): LSTM 隐藏层的维度。
            prediction_steps (int): 需要预测的未来时间步数量。
            num_layers (int): LSTM 网络的层数。
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim  # LSTM隐藏层维度
        self.num_layers = num_layers  # LSTM层数
        self.prediction_steps = prediction_steps  # 预测的时间步长

        # LSTM层
        # batch_first=True 表示输入和输出张量的维度格式为 (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # 全连接层，将LSTM最后一个时间步的输出映射到预测的步数
        self.fc = nn.Linear(hidden_dim, prediction_steps)

    def forward(self, x):
        """
        模型的前向传播。
        Args:
            x (torch.Tensor): 输入数据，形状为 (batch_size, seq_len, input_dim)。
        Returns:
            torch.Tensor: 预测结果，形状为 (batch_size, prediction_steps)。
        """
        batch_size = x.size(0)  # 获取批量大小

        # 初始化LSTM的隐藏状态和细胞状态
        # h_0 的形状为 (num_layers, batch_size, hidden_dim)
        # c_0 的形状为 (num_layers, batch_size, hidden_dim)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        # LSTM前向传播
        # lstm_out 的形状为 (batch_size, seq_len, hidden_dim)
        # (hn, cn) 分别是最后一个时间步的隐藏状态和细胞状态
        lstm_out, (hn, cn) = self.lstm(x, (h_0, c_0))

        # 我们使用最后一个时间步的最后一个LSTM层的隐藏状态 hn[-1] 进行预测
        # hn 的形状是 (num_layers, batch_size, hidden_dim)，所以 hn[-1] 取的是最后一层的隐藏状态
        # hn[-1] 的形状是 (batch_size, hidden_dim)
        out = self.fc(hn[-1])  # 或者使用 lstm_out[:, -1, :] 也是一样的效果，表示取序列最后一个时间步的输出

        return out

    def predict_speed(self, x):
        """
        预测速度的辅助函数，直接调用 forward 方法。
        Args:
            x (torch.Tensor): 输入数据。
        Returns:
            torch.Tensor: 预测的速度。
        """
        return self.forward(x)


# ------------------------------
# 权重初始化
def initialize_weights(model):
    """
    初始化模型权重。
    Args:
        model (nn.Module): 需要初始化权重的模型。
    """
    for name, param in model.named_parameters():
        if "weight" in name:  # 如果参数名包含 "weight"
            if param.dim() >= 2:  # 如果是多维权重（例如全连接层或LSTM的权重矩阵）
                nn.init.xavier_uniform_(param)  # 使用 Xavier 均匀分布初始化
            else:  # 如果是一维权重（较少见，但以防万一）
                nn.init.uniform_(param)  # 使用均匀分布初始化
        elif "bias" in name:  # 如果参数名包含 "bias"
            nn.init.constant_(param, 0)  # 将偏置初始化为 0


# ------------------------------
# 训练函数
def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
    """
    训练模型。
    Args:
        model (nn.Module): 待训练的模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        criterion (nn.Module): 损失函数。
        num_epochs (int): 训练轮数。
    Returns:
        nn.Module: 训练好的模型。
    """
    model.train()  # 设置模型为训练模式
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed in train_loader:  # 遍历训练数据批次
            optimizer.zero_grad()  # 清空梯度
            pred = model.predict_speed(batch_data)  # 模型预测
            loss = criterion(pred, batch_speed)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            total_loss += loss.item()  # 累积损失
        print(f"轮次 [{epoch + 1}/{num_epochs}], 损失: {total_loss / len(train_loader):.4f}")
    return model


# ------------------------------
# 评估函数
def evaluate_model(model, train_real_speed,test_loader,dataset_name="", results_dir=""):
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
    all_predicted = torch.cat(all_predicted).cpu().numpy()  # shape: (N, prediction_steps)
    all_true = torch.cat(all_true).cpu().numpy()  # shape: (N, prediction_steps)

    # 计算整体评估指标
    mse_val = np.mean((all_predicted - all_true) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted - all_true))
    mape_p = np.mean(np.abs((all_predicted - all_true) / all_true)) * 100
    print(f"Evaluation Metrics:\nMSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_p:.2f}%")

    # 拼接所有样本的数据，步长为5
    step = train_real_speed.shape[1]
    num_samples = 30
    true_concat = []
    pred_concat = []

    for i in range(num_samples):
        idx = i * step
        if idx >= all_true.shape[0]:
            break
        # 拼接每个样本的真实值和预测值
        true_concat.extend(all_true[idx])
        pred_concat.extend(all_predicted[idx])

    # 转换为数组以便绘制
    true_concat = np.array(true_concat)
    pred_concat = np.array(pred_concat)

    # 绘制拼接后的对比曲线
    plt.figure(figsize=(12, 8))
    plt.plot(true_concat, linestyle='--', marker='o', label='True')
    plt.plot(pred_concat, linestyle='-', marker='x', label='Predicted')

    # 设置图形标题和标签
    plt.title('True vs Predicted Speed for 30 Samples (Step 1-150)')
    plt.xlabel('Time Step')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    # plt.show()

    plot_filename = os.path.join(results_dir, f"{dataset_name}_speed_comparison_LSTM.png")
    plt.savefig(plot_filename)  # 保存图像
    print(f"速度对比图已保存至 {plot_filename}")  # Speed comparison plot saved to ...
    plt.close()  # 关闭图像，释放内存
    return mse_val, rmse_val, mae_val, mape_p  # 返回评估指标


# ------------------------------
# 保存预测速度到 CSV
def save_predictions_to_csv(model, test_loader, output_file="predictions.csv"):
    """
    将模型的预测速度和真实速度保存到 CSV 文件。
    Args:
        model (nn.Module): 训练好的模型。
        test_loader (torch.utils.data.DataLoader): 测试数据加载器。
        output_file (str): 输出 CSV 文件名。
    """
    model.eval()  # 设置为评估模式
    all_t, all_p = [], []  # 存储真实值和预测值
    with torch.no_grad():  # 不计算梯度
        for bd, bt in test_loader:  # 遍历测试数据
            p = model.predict_speed(bd)  # 预测速度
            all_t.append(bt.cpu())  # 收集真实值
            all_p.append(p.cpu())  # 收集预测值
    all_t = torch.cat(all_t).numpy()  # 拼接并转为 NumPy 数组
    all_p = torch.cat(all_p).numpy()  # 拼接并转为 NumPy 数组

    # 构建 DataFrame
    df_data = {}
    for j in range(all_t.shape[1]):  # 遍历预测的每一个时间步
        df_data[f"True_Step{j + 1}"] = all_t[:, j]
    for j in range(all_p.shape[1]):  # 遍历预测的每一个时间步
        df_data[f"Pred_Step{j + 1}"] = all_p[:, j]
    df = pd.DataFrame(df_data)

    df.to_csv(output_file, index=False)  # 保存到 CSV
    print(f"速度预测结果已保存到 {output_file}")


# ------------------------------
# 多步预测前车未来位置计算并保存
def compute_future_positions_and_save(model,
                                      test_data,
                                      raw_data,
                                      label_data,
                                      train_size,
                                      dt=0.1,
                                      output_file="pred_positions.xlsx",
                                      dataset_name=""):
    model.eval()
    with torch.no_grad():
        pred_speeds = model.predict_speed(test_data).cpu().numpy()  # (N_test, steps)
    N_test, steps = pred_speeds.shape
    idx = np.arange(train_size, train_size + N_test)

    # 当前速度（m/s）
    curr_speed = test_data[:, -1, 0].cpu().numpy()
    # 当前前车位置（ft -> m）
    curr_pos_ft = raw_data[idx, -1, 7].cpu().numpy()
    curr_pos_m = curr_pos_ft * 0.3048

    # 真值前车未来位置（ft -> m），label_data [:, :, 5] 存储前车未来位置
    true_pos_ft = label_data[idx, :, 5].cpu().numpy()
    true_pos_m = true_pos_ft * 0.3048

    # 真值车速（m/s），label_data[:, :, 4] 存储车速
    true_speeds = label_data[idx, :, 4].cpu().numpy()
    true_speeds *= 0.3048
    # 初始化存储所有样本的预测位置
    pred_pos_m = np.zeros((N_test, steps))

    # 对每个样本单独进行位置递推计算
    for i in range(N_test):
        # 每个样本的初始车速和位置
        prev_speed = curr_speed[i]
        prev_pos = curr_pos_m[i]

        # 计算当前样本的未来位置
        for k in range(steps):
            v_pred = pred_speeds[i, k]  # 当前样本在第k步的预测车速
            # 使用牛顿运动学公式计算位移
            a = (v_pred - prev_speed) / dt  # 计算加速度
            disp = prev_speed * dt + 0.5 * a * dt ** 2  # 计算位移
            pos = prev_pos + disp  # 更新位置
            pred_pos_m[i, k] = pos  # 更新预测位置
            prev_speed = v_pred  # 更新车速
            prev_pos = pos  # 更新位置

    # 误差评估
    rmse_p = np.sqrt(np.mean((pred_pos_m - true_pos_m) ** 2))
    mape_p = np.mean(np.abs((pred_pos_m - true_pos_m) / true_pos_m)) * 100
    print(f"Future Position Error -- RMSE: {rmse_p:.4f} m, MAPE: {mape_p:.2f}%")

    # 保存到 CSV
    data_dict = {}
    for i in range(steps):
        data_dict[f"Pred_Speed_step{i + 1}(m/s)"] = pred_speeds[:, i]
        data_dict[f"Pred_Pos_step{i + 1}(m)"] = pred_pos_m[:, i]
        data_dict[f"True_Pos_step{i + 1}(m)"] = true_pos_m[:, i]
        data_dict[f"True_Speed_step{i + 1}(m/s)"] = true_speeds[:, i]  # 添加真实速度

    df_pos = pd.DataFrame(data_dict)
    sheet_name = dataset_name  # 使用数据集名作为 sheet 名称 (例如 "data_5")

    # 如果文件存在则在其基础上添加新 sheet，否则新建文件
    # 'a'模式表示追加, 'w'模式表示写入(会覆盖)
    # if_sheet_exists='replace' 可以用来替换同名sheet，需要 pandas >= 1.3.0 和 openpyxl
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            df_pos.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:  # 如果文件不存在，首次以 'w' 模式创建
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df_pos.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"{dataset_name} 的位置预测结果已保存至 '{output_file}' 的 '{sheet_name}' 工作表。")
    # Position predictions for {dataset_name} saved to '{output_excel_file}' in sheet '{sheet_name}'.

    return rmse_p, mape_p  # 返回位置预测的评估指标

all_datasets_metrics_summary = []  # 用于存储所有数据集的评估指标字典列表

def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae,speed_mape ,pos_rmse, pos_mape):
    # 将单个数据集的评估指标存入列表
    metrics = {
        "数据集 (Dataset)": dataset_name,
        "速度MSE (Speed_MSE)": speed_mse,
        "速度RMSE (Speed_RMSE)": speed_rmse,
        "速度MAE (Speed_MAE)": speed_mae,
        "速度MAE (Speed_MAPE)": speed_mape,
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
# 主函数
if __name__ == "__main__":
    torch.manual_seed(42)  # 设置随机种子以确保结果可复现
    # torch.manual_seed(42)


    # 获取指定目录下所有的 .mat 文件
    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    print(data_files)
    if not data_files:
        print(f"在目录 {DATA_DIR} 中未找到 .mat 文件。")  # No .mat files found in the directory.
        exit()

    # 定义位置预测结果的Excel文件路径
    position_predictions_excel_path = os.path.join(RESULTS_DIR, "pred_positions_all_datasets_LSTM_1129.xlsx")
    # 如果旧的汇总Excel文件存在，可以选择删除以重新开始，或者让程序追加/替换sheet
    # 当前的 compute_future_positions_and_save 实现会替换同名sheet或追加新sheet

    print(f"找到以下数据集文件: {data_files}")  # Found the following dataset files:

    # 遍历每个找到的数据文件
    for data_file_path in data_files:
        dataset_filename = os.path.basename(data_file_path)  # 获取文件名，例如 "data_5.mat"
        dataset_name_clean = dataset_filename.replace(".mat", "")  # 去掉 .mat 后缀，例如 "data_5"

        print(f"\n==================== 开始处理数据集: {dataset_filename} ====================")
        # Processing dataset: ...

        data = sio.loadmat(data_file_path)
        raw_data = torch.tensor(data['train_data'], dtype=torch.float32)
        label_data = torch.tensor(data['lable_data'], dtype=torch.float32)  # 保留全部 6 列

        # 提取多步速度标签（前车速度在第 5 列索引 4）
        train_real_speed_all = label_data[:, :, 0]
        print("train_real_speed_all shape:", train_real_speed_all.shape)

        # 构造多步输入：取最后 50 步的前车速度(5)和加速度(6)
        train_data = raw_data[:, -50:, [0, 1, 2, 3, 5]].clone() * 0.3048
        train_real_speed = train_real_speed_all.clone()
        print("train_data shape:", train_data.shape)

        # 单位转换：ft/s -> m/s
        # train_data[:, :, 0] *= 0.3048  错误
        # train_data[:, :, 1] *= 0.3048
        train_real_speed *= 0.3048

        # 抽样 10%
        total_samples = train_data.shape[0]
        sample_size = int(total_samples * 0.2)
        train_data = train_data[:sample_size]
        train_real_speed = train_real_speed[:sample_size]

        # 检查
        check_data(train_data, "train_data")
        check_data(train_real_speed, "train_real_speed")

        # 划分训练/测试集
        dataset_size = train_data.shape[0]
        train_size = int(dataset_size * 0.8)
        test_data = train_data[train_size:]
        test_real_speed = train_real_speed[train_size:]
        train_data = train_data[:train_size]
        train_real_speed = train_real_speed[:train_size]

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data, train_real_speed),
            batch_size=32, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_data, test_real_speed),
            batch_size=32, shuffle=False
        )

        # 模型配置
        input_dim = train_data.shape[2]
        hidden_dim = 128
        prediction_steps = train_real_speed.shape[1]  # 5
        model = LSTMModel(input_dim, hidden_dim, prediction_steps,
                                    num_layers=1)
        initialize_weights(model)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-4)

        # 训练 & 评估
        model = train_model(model, train_loader, optimizer, criterion, num_epochs=30)
        speed_mse, speed_rmse, speed_mae, speed_mape = evaluate_model(model, train_real_speed, test_loader,
                                                                      dataset_name=dataset_name_clean,
                                                                      results_dir=RESULTS_DIR)

        # 保存速度预测
        # save_predictions_to_csv(model, test_loader, output_file="predictions..xlsx")

        # **新增**：多步预测前车未来位置计算并保存
        pos_rmse, pos_mape = compute_future_positions_and_save(
            model,
            test_data,
            raw_data,
            label_data,
            train_size,
            dt=0.1,
            output_file=position_predictions_excel_path,
            dataset_name=dataset_name_clean
        )
        store_dataset_metrics(dataset_name_clean, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse, pos_mape)
    summary_metrics_csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_all_datasets_LSTM_1129.csv")
    save_all_metrics_to_csv(summary_metrics_csv_path)
    print("\n所有数据集处理完毕。")  # All datasets processed.

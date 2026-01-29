import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import os
# --- 定义 IDM 模型 ---
def idm_model_calibration(v_n, s_safe, delta_v, params):
    """
    IDM 模型计算下一时刻速度，用于标定。
    :param v_n: 自车当前速度 (numpy array)
    :param s_safe: 当前跟车距离 (numpy array)
    :param delta_v: 自车与前车的速度差 (numpy array)
    :param params: [v_desired, T, a_max, b_safe, delta, s0], 需要标定的 IDM 模型参数
    :return: 下一时刻速度 (numpy array)
    """
    v_desired, T, a_max, b_safe, delta, s0 = params
    delta_t = 0.1  # 时间步长

    # 确保参数值合理，避免非法计算
    a_max = max(a_max, 1e-6)
    b_safe = max(b_safe, 1e-6)
    s_safe = np.maximum(s_safe, 1e-6)

    # 理想安全距离
    s_star = s0 + v_n * T + (v_n * delta_v) / (2 * np.sqrt(a_max * b_safe))
    s_star = np.maximum(s_star, 0)  # 确保 s_star 非负

    # 预测速度
    v_follow = v_n + delta_t * a_max * (1 - (v_n / v_desired) ** delta - (s_star / s_safe) ** 2)
    return np.maximum(v_follow, 0)  # 确保速度非负


# --- 损失函数 (RMSE) ---
def loss_function_idm(params, v_n, s_safe, delta_v, real_speed):
    """
    损失函数：计算预测速度与真实速度的 RMSE
    :param params: [v_desired, T, a_max, b_safe, delta, s0], 需要标定的 IDM 参数
    :param v_n: 自车当前速度 (numpy array)
    :param s_safe: 当前跟车距离 (numpy array)
    :param delta_v: 自车与前车的速度差 (numpy array)
    :param real_speed: 下一时刻真实速度 (numpy array)
    :return: RMSE 值
    """
    predicted_speed = idm_model_calibration(v_n, s_safe, delta_v, params)
    rmse = np.sqrt(np.mean((predicted_speed - real_speed) ** 2))
    return rmse  # 返回 RMSE


# --- 使用 minimize 标定 IDM 模型参数 ---
def calibrate_idm(v_n, s_safe, delta_v, real_speed):
    """
    标定 IDM 模型参数
    :param v_n: 自车当前速度 (numpy array)
    :param s_safe: 当前跟车距离 (numpy array)
    :param delta_v: 自车与前车的速度差 (numpy array)
    :param real_speed: 下一时刻真实速度 (numpy array)
    :return: 标定后的 IDM 模型参数 [v_desired, T, a_max, b_safe, delta, s0]
    """
    # 初始化参数 [v_desired, T, a_max, b_safe, delta, s0]
    initial_params = [30.0, 1.5, 2.0, 3.0, 4.0, 2.0]

    # 参数范围 (bounds)
    param_bounds = [
        (10, 40),   # v_desired (目标速度)，范围 10 - 40 m/s
        (0.5, 2.5), # T (反应时间)，范围 0.5 - 2.5 秒
        (0.1, 5),   # a_max (最大加速度)，范围 0.1 - 5 m/s²
        (0.1, 5),   # b_safe (最大减速度)，范围 0.1 - 5 m/s²
        (1, 10),    # delta (加速度指数)，范围 1 - 10
        (0.1, 5)    # s0 (最小安全距离)，范围 0.1 - 5 m
    ]

    # 优化参数
    result = minimize(
        loss_function_idm,
        initial_params,
        args=(v_n, s_safe, delta_v, real_speed),
        bounds=param_bounds,  # 参数范围约束
        method="L-BFGS-B"
    )

    if result.success:
        print(f"Optimization Successful! Found Parameters:")
        print(f"v_desired={result.x[0]:.4f}, T={result.x[1]:.4f}, a_max={result.x[2]:.4f}, b_safe={result.x[3]:.4f}, delta={result.x[4]:.4f}, s0={result.x[5]:.4f}")
    else:
        print("Optimization Failed. Using Initial Parameters.")

    return result.x


# --- 对比结果并输出评价指标 ---
def compare_results_idm(v_n, s_safe, delta_v, real_speed, calibrated_params):
    """
    对比真实速度、标定后的 IDM 模型预测速度，并输出评价指标
    :param v_n: 自车当前速度 (numpy array)
    :param s_safe: 当前跟车距离 (numpy array)
    :param delta_v: 自车与前车的速度差 (numpy array)
    :param real_speed: 下一时刻真实速度 (numpy array)
    :param calibrated_params: 标定后的 IDM 参数
    """
    # 标定后的预测速度
    calibrated_speed = idm_model_calibration(v_n, s_safe, delta_v, calibrated_params)

    # 计算评价指标
    mse = np.mean((real_speed - calibrated_speed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(real_speed - calibrated_speed))
    print(f"Evaluation Metrics for Calibrated IDM Model:\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.plot(real_speed[:100], label='True Speed', linestyle='--', marker='o')
    plt.plot(calibrated_speed[:100], label='Calibrated Speed', linestyle='-', marker='x')
    plt.title('True Speed vs Calibrated Speed (IDM Model)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid()
    plt.show()

def compute_position_and_spacing_and_save(calibrated_params,
                                           raw_data,
                                           label_data,
                                           test_idx,
                                           dt=0.1,
                                           output_file='predictions_extended.xlsx'):
    """
    基于标定后的 IDM 模型参数计算下一时刻车辆纵向位置（Y）和车头间距，并保存到 Excel。
    :param calibrated_params: 标定后的 IDM 参数列表 [v_desired, T, a_max, b_safe, delta, s0]
    :param raw_data: 原始特征数据，单位 ft / ft/s，shape=(N, T, D_raw)
    :param label_data: 标签数据，单位 ft，shape=(N, T, D_label)
    :param test_idx: 测试集样本索引 array
    :param dt: 时间步长（s），代码2 中默认 1.0
    :param output_file: 输出文件路径
    """
    # —— 提取测试集末帧的速度、距离、速度差 (转换到 m/s, m, m/s)
    v_n     = raw_data[test_idx, -1, 0] * 0.3048
    s_safe  = raw_data[test_idx, -1, 1] * 0.3048
    delta_v = raw_data[test_idx, -1, 2] * 0.3048

    # —— IDM 计算预测速度 (m/s)
    pred_v = idm_model_calibration(v_n, s_safe, delta_v, calibrated_params)

    # —— 当前与真实 Y 坐标 (ft) 以及真实间距 (标签中 spacing 为 ft)
    current_Y_ft   = raw_data[test_idx, -1, 4]* 0.3048
    true_Y_ft      = label_data[test_idx, -1, 3]* 0.3048
    true_spacing_m = label_data[test_idx, -1, 1] * 0.3048  # 转为 m

    # —— 预测位移：disp = v_prev*dt + 0.5*a*dt^2,  a=(pred_v - v_prev)/dt
    v_prev_ft  = v_n
    pred_v_ft  = pred_v
    disp_ft    = v_prev_ft * dt + 0.5 * ((pred_v_ft - v_prev_ft) / dt) * dt**2
    pred_Y_ft  = current_Y_ft + disp_ft

    # —— 转换到米并计算预测间距
    pred_Y_m       = pred_Y_ft
    true_Y_m       = true_Y_ft
    pred_spacing_m = (true_Y_ft -pred_Y_m)  + true_spacing_m

    # —— 误差指标
    rmse_Y  = np.sqrt(np.mean((pred_Y_m - true_Y_m)**2))
    mape_Y  = np.mean(np.abs((pred_Y_m - true_Y_m) / true_Y_m)) * 100
    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m)**2))
    mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m) / true_spacing_m)) * 100

    print(f'Position Error -- RMSE: {rmse_Y:.3f} m, MAPE: {mape_Y:.2f}%')
    print(f'Spacing  Error -- RMSE: {rmse_sp:.3f} m, MAPE: {mape_sp:.2f}%')

    # —— 保存到 Excel
    df = pd.DataFrame({
        'Pred Speed (m/s)'  : pred_v,
        'Pred Y (m)'        : pred_Y_m,
        'True Y (m)'        : true_Y_m,
        'Pred Spacing (m)'  : pred_spacing_m,
        'True Spacing (m)'  : true_spacing_m
    })
    sheet_name = 'IDM_calibrated'
    mode = 'a' if os.path.exists(output_file) else 'w'
    with pd.ExcelWriter(output_file, engine='openpyxl', mode=mode) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Results saved to '{output_file}' sheet '{sheet_name}'.")


# --- 主函数 ---
if __name__ == "__main__":
    # 读取数据
    data = loadmat('E:\pythonProject1\data_fine_0.1.mat')  # 替换为你的实际文件名

    raw_data = data['train_data']  # shape: (N, T, D_raw)，单位：ft / ft/s
    label_data = data['lable_data']  # shape: (N, T, D_label)，包含真实 spacing（索引1）和 Y 坐标（索引3），单位 ft
    total = raw_data.shape[0]
    subset = int(total * 0.1)
    train_sz = int(subset * 0.8)
    test_idx = np.arange(train_sz, subset)
    # 提取数据
    train_data = data['train_data']  # shape: (样本数, 时间序列长度, 输入特征数)
    train_real_speed1 = data['lable_data']  # shape: (样本数, 1)
    train_real_speed = train_real_speed1[:, -1, 0]  # shape: (样本数, 1)
    train_s_safe =  train_data[:, -1, 1] # shape: (样本数, 1)
    print(train_real_speed.shape)
    # 缩小数据集
    # 数据准备
    total_samples = train_data.shape[0]
    train_split1 = int(total_samples * 0.1)  # 使用 80% 数据作为训练集，其余为测试集
    # 划分训练集和测试集
    train_data = train_data[:train_split1]
    train_real_speed = train_real_speed[:train_split1]
    train_s_safe = train_s_safe[:train_split1]

    print(train_s_safe.shape)
    # 数据准备
    total_samples = train_data.shape[0]
    train_split = int(total_samples * 0.8)  # 使用 80% 数据作为训练集，其余为测试集

    # 划分训练集和测试集
    train_data_train = train_data[:train_split]
    train_real_speed_train = train_real_speed[:train_split].reshape(-1)
    train_s_safe_train = train_s_safe[:train_split].reshape(-1)

    train_data_test = train_data[train_split:]
    train_real_speed_test = train_real_speed[train_split:].reshape(-1)
    train_s_safe_test = train_s_safe[train_split:].reshape(-1)

    # 训练集的自车速度、前车距离和速度差
    v_n_train = train_data_train[:, -1, 0] * 0.3048  # 自车速度 ft/s -> m/s
    delta_v_train = train_data_train[:, -1, 2] * 0.3048  # 速度差 ft/s -> m/s
    s_safe_train = train_data_train[:, -1, 1] * 0.3048   # 距离 ft -> m
    real_speed_train = train_real_speed_train * 0.3048  # 真实速度 ft/s -> m/s

    # 测试集的自车速度、前车距离和速度差
    v_n_test = train_data_test[:, -1, 0] * 0.3048
    delta_v_test = train_data_test[:, -1, 2] * 0.3048
    s_safe_test = train_data_test[:, -1, 1] * 0.3048
    real_speed_test = train_real_speed_test * 0.3048

    # 使用 minimize 标定 IDM 参数（使用训练集）
    calibrated_params = calibrate_idm(v_n_train, s_safe_train, delta_v_train, real_speed_train)

    # 使用测试集评估模型性能
    compare_results_idm(v_n_test, s_safe_test, delta_v_test, real_speed_test, calibrated_params)
    # compute_position_and_spacing_and_save(
    #     calibrated_params,
    #     raw_data,
    #     label_data,
    #     test_idx,
    #     dt=0.1,
    #     output_file='predictions_extended.xlsx'
    # )

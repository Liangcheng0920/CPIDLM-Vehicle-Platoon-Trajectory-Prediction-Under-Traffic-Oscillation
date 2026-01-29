# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import pandas as pd
# import numpy as np
# import glob  # 用于查找文件路径
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许多个OpenMP库存在，避免某些环境下的冲突
#
# # --- 全局路径定义 ---
# DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # 数据集存放目录
# RESULTS_DIR = "E:\\pythonProject1\\results_ngsim"  # 实验结果保存目录
#
# # 确保结果目录存在 (Ensure results directory exists)
# os.makedirs(RESULTS_DIR, exist_ok=True)
#
# # --- 全局参数 ---
# PRED_HORIZON = 5  # 预测未来5个时间步
# DT = 0.1  # 时间步长 (s) - 从原代码推断
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
#               v_desired=12.64798288, T=0.50284384, a_max=0.10033688,
#               b_safe=4.98937183, delta=1.0, s0=0.13082412,
#               delta_t=DT):  # 使用全局DT
#     device = v_n.device
#     # 把常数转成 0-d tensor
#     v_desired = torch.tensor(v_desired, device=device, dtype=v_n.dtype)
#     T = torch.tensor(T, device=device, dtype=v_n.dtype)
#     a_max = torch.tensor(a_max, device=device, dtype=v_n.dtype).clamp(min=1e-6)
#     b_safe = torch.tensor(b_safe, device=device, dtype=v_n.dtype).clamp(min=1e-6)
#     s0 = torch.tensor(s0, device=device, dtype=v_n.dtype)
#     delta = torch.tensor(delta, device=device, dtype=v_n.dtype)
#     delta_t_tensor = torch.tensor(delta_t, device=device, dtype=v_n.dtype)
#
#     s_safe = s_safe.clamp(min=1e-6)  # 确保s_safe是正的
#
#     s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe))
#     s_star = s_star.clamp(min=0.0)  # s_star不能为负
#
#     # 防止v_desired为0导致除零错误
#     v_n_ratio = torch.zeros_like(v_n)
#     mask_v_desired_nonzero = v_desired.abs() > 1e-6
#     if mask_v_desired_nonzero.any():  # 只有当v_desired非零时才计算比率
#         v_n_ratio[mask_v_desired_nonzero] = (v_n[mask_v_desired_nonzero] / v_desired[mask_v_desired_nonzero])
#
#     acceleration_term = a_max * (
#             1 - v_n_ratio ** delta - (s_star / s_safe) ** 2
#     )
#     v_follow = v_n + delta_t_tensor * acceleration_term
#     return v_follow.clamp(min=0.0)
#
#
# # --- 定义新融合模型 ---
# class HybridIDMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim=PRED_HORIZON, num_layers=2):
#         super(HybridIDMModel, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)  # LSTM输出K步预测
#         self.alpha_raw = nn.Parameter(torch.tensor(0.0))  # 可学习的 alpha
#
#         # IDM参数，如果希望它们也是可学习的，可以定义为nn.Parameter
#         # 这里我们保持它们为固定值，通过idm_fixed函数传入
#         self.v_desired_idm = 12.64798288
#         self.T_idm = 0.50284384
#         self.a_max_idm = 0.10033688
#         self.b_safe_idm = 4.98937183
#         self.delta_idm = 1.0
#         self.s0_idm = 0.13082412
#
#     def forward(self, x, s_safe_initial, v_lead_initial):
#         """
#         :param x:  输入序列，shape=(batch, seq_len, input_dim)
#                    x[:, -1, 0] 是当前自车速度 (v_n_t)
#                    x[:, -1, 1] 是当前跟车距离 (s_t)
#                    x[:, -1, 4] 是当前前车速度 (v_lead_t)
#         :param s_safe_initial: 当前安全距离 (真实观测值), shape=(batch,)
#         :param v_lead_initial: 当前前车速度 (真实观测值，用于IDM多步)，shape=(batch,)
#         :return:
#           y_lstm_multistep: LSTM直接输出的K步预测速度，shape=(batch, K)
#           y_idm_multistep: IDM迭代K步的预测速度，shape=(batch, K)
#           alpha: 当前学习到的权重标量
#         """
#         batch_size = x.size(0)
#         device = x.device
#
#         # LSTM K步预测
#         out, _ = self.lstm(x)  # (batch, seq, hidden_dim)
#         y_lstm_multistep = self.fc(out[:, -1, :])  # (batch, K)
#
#         # IDM K步迭代预测
#         y_idm_multistep_list = []
#
#         # IDM迭代的初始状态
#         v_ego_current_idm = x[:, -1, 0].clone()  # 当前自车速度 (m/s)
#         s_current_idm = s_safe_initial.clone()  # 当前实际间距 (m)
#         # v_lead_constant_idm = x[:, -1, 4].clone() # 当前前车速度，假设未来K步不变 (m/s)
#         v_lead_constant_idm = v_lead_initial.clone()  # 使用传入的初始前车速度
#
#         for k_step in range(PRED_HORIZON):
#             delta_v_idm = v_lead_constant_idm - v_ego_current_idm
#
#             # 调用idm_fixed进行单步预测
#             v_ego_next_pred_idm = idm_fixed(
#                 v_ego_current_idm, s_current_idm, delta_v_idm,
#                 v_desired=self.v_desired_idm, T=self.T_idm, a_max=self.a_max_idm,
#                 b_safe=self.b_safe_idm, delta=self.delta_idm, s0=self.s0_idm, delta_t=DT
#             )
#             y_idm_multistep_list.append(v_ego_next_pred_idm.unsqueeze(1))
#
#             # 更新IDM下一次迭代的输入
#             # 下一步的间距 s_t+1 = s_t + (v_lead_t - v_ego_t) * dt
#             # 这里 v_ego_t 是当前迭代步开始时的速度
#             s_current_idm = s_current_idm + (v_lead_constant_idm - v_ego_current_idm) * DT
#             s_current_idm = s_current_idm.clamp(min=1e-6)  # 确保间距不为负或零
#
#             v_ego_current_idm = v_ego_next_pred_idm  # 更新自车速度为预测值
#
#         y_idm_multistep = torch.cat(y_idm_multistep_list, dim=1)  # (batch, K)
#
#         alpha = torch.sigmoid(self.alpha_raw)
#         return y_lstm_multistep, y_idm_multistep, alpha
#
#
# def initialize_weights(model):
#     for name, param in model.named_parameters():
#         if "weight" in name and param.dim() > 1:  # Xavier适用于多维权重
#             nn.init.xavier_uniform_(param)
#         elif "bias" in name:
#             nn.init.constant_(param, 0)
#
#
# # --- 训练函数 ---
# def train_model(model, train_loader, optimizer, num_epochs=30, alpha_decay_loss=0.1):
#     model.train()
#
#     # 生成损失权重 w_k = exp(-alpha_decay * k), k从0到K-1
#     loss_weights = torch.exp(-alpha_decay_loss * torch.arange(PRED_HORIZON, dtype=torch.float32)).to(
#         next(model.parameters()).device)
#     loss_weights = loss_weights / loss_weights.sum() * PRED_HORIZON  # 归一化使得平均权重为1
#
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in train_loader:
#             # batch_y_multistep should have shape (batch, PRED_HORIZON)
#             optimizer.zero_grad()
#
#             y_lstm_multistep, y_idm_multistep, alpha = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)
#
#             # 多步加权MSE损失
#             # L_lstm_true = sum_k w_k * (y_lstm_k - y_true_k)^2
#             # L_lstm_idm = sum_k w_k * (y_lstm_k - y_idm_k)^2
#
#             loss_lstm_vs_true = ((y_lstm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
#             loss_lstm_vs_idm = ((y_lstm_multistep - y_idm_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
#
#             loss = alpha * loss_lstm_vs_true + (1 - alpha) * loss_lstm_vs_idm
#
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         print(f"Epoch {epoch + 1}/{num_epochs}  Loss: {epoch_loss / len(train_loader):.6f}  α={alpha.item():.4f}")
#     return model
#
#
#
# # # --- 测试/评估函数 ---
# def evaluate_model(model, test_loader, alpha_decay_loss=0.1,dataset_name="", results_dir=""):
#     model.eval()
#     all_pred_lstm, all_pred_idm, all_true, all_alpha = [], [], [], []
#
#     loss_weights = torch.exp(-alpha_decay_loss * torch.arange(PRED_HORIZON, dtype=torch.float32)).to(
#         next(model.parameters()).device)
#     loss_weights = loss_weights / loss_weights.sum() * PRED_HORIZON
#
#     total_mse_lstm_vs_true = 0
#     total_mse_lstm_vs_idm = 0
#     total_combined_mse = 0
#
#     with torch.no_grad():
#         for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in test_loader:
#             y_lstm_multistep, y_idm_multistep, alpha = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)
#
#             all_pred_lstm.append(y_lstm_multistep.cpu())
#             all_pred_idm.append(y_idm_multistep.cpu())
#             all_true.append(batch_y_multistep.cpu())
#             all_alpha.append(alpha.cpu().item())
#
#             loss_lstm_vs_true = ((y_lstm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
#             loss_lstm_vs_idm = ((y_lstm_multistep - y_idm_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
#
#             y_combined_multistep = alpha * y_lstm_multistep + (1 - alpha) * y_idm_multistep
#             loss_combined_vs_true = (
#                         (y_combined_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
#
#             total_mse_lstm_vs_true += loss_lstm_vs_true.item() * batch_x.size(0)
#             total_mse_lstm_vs_idm += loss_lstm_vs_idm.item() * batch_x.size(0)
#             total_combined_mse += loss_combined_vs_true.item() * batch_x.size(0)
#
#     num_samples = len(test_loader.dataset)
#     avg_mse_lstm_vs_true = total_mse_lstm_vs_true / num_samples
#     avg_mse_lstm_vs_idm = total_mse_lstm_vs_idm / num_samples
#     avg_combined_mse = total_combined_mse / num_samples
#     final_alpha = np.mean(all_alpha)
#
#     y_pred_lstm_cat = torch.cat(all_pred_lstm)  # (N_test, K)
#     y_pred_idm_cat = torch.cat(all_pred_idm)  # (N_test, K)
#     y_true_cat = torch.cat(all_true)  # (N_test, K)
#
#     y_pred_combined_cat = final_alpha * y_pred_lstm_cat + (1 - final_alpha) * y_pred_idm_cat
#     mse_val=avg_combined_mse
#     rmse_val=np.sqrt(avg_combined_mse)
#     mae_val = torch.mean(torch.abs(y_pred_combined_cat - y_true_cat)).item()
#     mape_p =torch.mean((np.abs((y_pred_combined_cat - y_true_cat) / y_true_cat)) * 100).item()
#
#     print(f"\nTest Results (Averaged over {PRED_HORIZON} steps with decay weights):")
#     print(f"  LSTM vs True   -- MSE: {avg_mse_lstm_vs_true:.4f}, RMSE: {np.sqrt(avg_mse_lstm_vs_true):.4f}")
#     print(f"  LSTM vs IDM    -- MSE: {avg_mse_lstm_vs_idm:.4f}, RMSE: {np.sqrt(avg_mse_lstm_vs_idm):.4f}")
#     print(f"  Combined vs True -- MSE: {avg_combined_mse:.4f}, RMSE: {np.sqrt(avg_combined_mse):.4f}")
#     print(f"  Final α={final_alpha:.4f}")
#
#     # 打印每一步的性能 (使用简单平均MSE，不带权重，仅作参考)
#     for k_step in range(PRED_HORIZON):
#         mse_step_lstm = nn.MSELoss()(y_pred_lstm_cat[:, k_step], y_true_cat[:, k_step]).item()
#         mse_step_idm = nn.MSELoss()(y_pred_idm_cat[:, k_step], y_true_cat[:, k_step]).item()
#         mse_step_combined = nn.MSELoss()(y_pred_combined_cat[:, k_step], y_true_cat[:, k_step]).item()
#         print(
#             f"  Step {k_step + 1} Pred vs True -- LSTM MSE: {mse_step_lstm:.4f}, IDM MSE: {mse_step_idm:.4f}, Combined MSE: {mse_step_combined:.4f}")
#
#     # 绘图 (例如，只绘制第一个预测步 K=0)
#     k_plot = 0
#     plt.figure(figsize=(12, 7))
#     plt.plot(y_true_cat[:100, k_plot].numpy(), '--o', label=f'True (Step {k_plot + 1})')
#     plt.plot(y_pred_lstm_cat[:100, k_plot].numpy(), '-x', label=f'LSTM Pred (Step {k_plot + 1})')
#     plt.plot(y_pred_idm_cat[:100, k_plot].numpy(), '-s', label=f'IDM Pred (Step {k_plot + 1})')
#     plt.plot(y_pred_combined_cat[:100, k_plot].numpy(), '-.',
#              label=f'Combined Pred (Step {k_plot + 1}, α={final_alpha:.2f})')
#     plt.title(f'Speed Prediction Comparison (First 100 samples, Step {k_plot + 1})')
#     plt.xlabel("Sample Index")
#     plt.ylabel("Speed (m/s)")
#     plt.legend();
#     plt.grid();
#     # plt.show()
#     plot_filename = os.path.join(results_dir, f"{dataset_name}_speed_comparison_PLSTM_IDM.png")
#     plt.savefig(plot_filename)  # 保存图像
#     print(f"速度对比图已保存至 {plot_filename}")  # Speed comparison plot saved to ...
#     plt.close()  # 关闭图像，释放内存
#     return mse_val, rmse_val, mae_val, mape_p  # 返回评估指标
#
#
# # === 修改后的 compute_position_and_spacing_and_save ===
# def compute_position_and_spacing_and_save(model,
#                                           test_loader,  # 使用dataloader获取批量数据
#                                           raw_data_all,  # (N_all, seq_len_raw, feat_raw)
#                                           label_data_all,  # (N_all, K_lab, feat_lab)
#                                           train_size,  # 用于定位测试数据在原始数据中的索引
#                                           dt=DT,
#                                           output_file="predictions_multistep_extended.xlsx",
#                                           dataset_name = ""):
#     model.eval()
#
#     # 从 test_loader 中获取所有测试数据对应的原始索引
#     # TensorDataset(test_seq, test_y_multistep, test_s_safe, test_v_lead_initial)
#     # 我们需要 test_seq 的原始索引来对应 raw_data_all 和 label_data_all
#     # 假设test_loader.dataset.tensors[0]是test_seq，我们需要一个方法来获取原始索引
#     # 这里简化处理，直接迭代test_loader，并假设其顺序对应原始数据的测试部分
#
#     all_pred_speeds_k_steps_list = []
#     all_true_speeds_k_steps_list = []
#     # all_initial_ego_pos_ft_list = [] # raw_data[idx, -1, 4]
#     # all_initial_lead_pos_ft_list = [] # raw_data[idx, -1, 7]
#     # all_initial_ego_speed_mps_list = [] # test_data[:, -1, 0]
#     # all_initial_lead_speed_mps_list = [] # test_data[:, -1, 4]
#
#     all_true_ego_pos_k_steps_m_list = []  # label_data[idx, :, 3] * 0.3048
#     all_true_spacing_k_steps_m_list = []  # label_data[idx, :, 1] * 0.3048
#
#     # 需要原始数据中的初始位置信息来推算多步位置
#     # raw_data_test_part = raw_data_all[train_size:] # (N_test, seq_len_raw, feat_raw)
#     # label_data_test_part = label_data_all[train_size:] # (N_test, K_lab, feat_lab)
#
#     # 收集所有批次的预测和真实数据
#     y_lstm_list, y_idm_list, y_true_list = [], [], []
#     initial_ego_pos_ft_collected = []
#     initial_lead_pos_ft_collected = []
#     initial_ego_speed_mps_collected = []
#     initial_lead_speed_mps_collected = []
#     true_future_ego_pos_ft_collected = []  # label_data[idx, k, 3]
#     true_future_spacing_ft_collected = []  # label_data[idx, k, 1]
#
#     test_indices = np.arange(train_size, train_size + len(test_loader.dataset))
#
#     with torch.no_grad():
#         for i, (batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial) in enumerate(test_loader):
#             start_idx_in_batch = i * test_loader.batch_size
#             end_idx_in_batch = start_idx_in_batch + batch_x.size(0)
#             current_batch_indices = test_indices[start_idx_in_batch:end_idx_in_batch]
#
#             y_lstm_k, y_idm_k, alpha = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)
#             y_lstm_list.append(y_lstm_k.cpu())
#             y_idm_list.append(y_idm_k.cpu())
#             y_true_list.append(batch_y_multistep.cpu())
#
#             # 从原始数据中提取初始状态 (单位: ft, ft/s)
#             # batch_x 已经转换成了 m/s. 我们需要原始的 raw_data
#             # x_raw_batch = raw_data_all[current_batch_indices, -50:, :] # 和训练输入一致的历史长度
#
#             # 初始自车位置 (ft) from raw_data (column 4: 自车位置)
#             initial_ego_pos_ft_collected.append(raw_data_all[current_batch_indices, -1, 4].cpu())
#             # 初始前车位置 (ft) from raw_data (column 7: 前车位置)
#             initial_lead_pos_ft_collected.append(raw_data_all[current_batch_indices, -1, 7].cpu())
#
#             # 初始自车速度 (m/s) from batch_x (column 0: 当前车速)
#             initial_ego_speed_mps_collected.append(batch_x[:, -1, 0].cpu())
#             # 初始前车速度 (m/s) from batch_x (column 4: 前车速度)
#             initial_lead_speed_mps_collected.append(batch_x[:, -1, 4].cpu())
#
#             # 真实未来K步自车位置 (ft) from label_data (column 3: 自车位置)
#             true_future_ego_pos_ft_collected.append(label_data_all[current_batch_indices, :PRED_HORIZON, 3].cpu())
#             # 真实未来K步间距 (ft) from label_data (column 1: 前车距离)
#             true_future_spacing_ft_collected.append(label_data_all[current_batch_indices, :PRED_HORIZON, 1].cpu())
#
#     y_lstm_all = torch.cat(y_lstm_list, dim=0)  # (N_test, K) m/s
#     y_idm_all = torch.cat(y_idm_list, dim=0)  # (N_test, K) m/s
#     y_true_speeds_all = torch.cat(y_true_list, dim=0)  # (N_test, K) m/s
#
#     alpha_val = model.alpha_raw.sigmoid().item()  # Get current alpha
#     pred_speeds_fused_mps = alpha_val * y_lstm_all + (1 - alpha_val) * y_idm_all  # (N_test, K) m/s
#
#     initial_ego_pos_ft = torch.cat(initial_ego_pos_ft_collected, dim=0)  # (N_test)
#     initial_lead_pos_ft = torch.cat(initial_lead_pos_ft_collected, dim=0)  # (N_test)
#     initial_ego_speed_mps = torch.cat(initial_ego_speed_mps_collected, dim=0)  # (N_test)
#     initial_lead_speed_mps = torch.cat(initial_lead_speed_mps_collected, dim=0)  # (N_test)
#
#     true_future_ego_pos_ft = torch.cat(true_future_ego_pos_ft_collected, dim=0)  # (N_test, K)
#     true_future_spacing_ft = torch.cat(true_future_spacing_ft_collected, dim=0)  # (N_test, K)
#
#     N_test = pred_speeds_fused_mps.size(0)
#     pred_ego_pos_ft = torch.zeros_like(pred_speeds_fused_mps)  # (N_test, K)
#     pred_lead_pos_ft = torch.zeros_like(pred_speeds_fused_mps)  # (N_test, K)
#     pred_spacing_ft = torch.zeros_like(pred_speeds_fused_mps)  # (N_test, K)
#
#     # 转换为 ft/s 进行位移计算
#     pred_speeds_fused_ftps = pred_speeds_fused_mps / 0.3048
#     initial_ego_speed_ftps = initial_ego_speed_mps / 0.3048
#     initial_lead_speed_ftps = initial_lead_speed_mps / 0.3048  # 这是t时刻的速度，假设未来K步匀速
#
#     current_ego_pos_ft = initial_ego_pos_ft.clone()
#     current_lead_pos_ft = initial_lead_pos_ft.clone()
#
#     for k in range(PRED_HORIZON):
#         # 当前步开始时的速度 (用于计算本步位移)
#         speed_ego_this_step_ftps = pred_speeds_fused_ftps[:, k - 1] if k > 0 else initial_ego_speed_ftps
#         speed_lead_this_step_ftps = initial_lead_speed_ftps  # 前车匀速
#
#         # 计算位移: simple v*dt
#         disp_ego_ft = speed_ego_this_step_ftps * dt
#         disp_lead_ft = speed_lead_this_step_ftps * dt
#
#         current_ego_pos_ft += disp_ego_ft
#         current_lead_pos_ft += disp_lead_ft
#
#         pred_ego_pos_ft[:, k] = current_ego_pos_ft
#         pred_lead_pos_ft[:, k] = current_lead_pos_ft
#         pred_spacing_ft[:, k] = current_lead_pos_ft - current_ego_pos_ft
#
#     # 转换单位为米进行比较和保存
#     pred_ego_pos_m = pred_ego_pos_ft.numpy() * 0.3048
#     true_ego_pos_m = true_future_ego_pos_ft.numpy() * 0.3048
#     pred_spacing_m = pred_spacing_ft.numpy() * 0.3048
#     true_spacing_m = true_future_spacing_ft.numpy() * 0.3048
#
#     # 打印 K 步累积误差或最后一步的误差
#     # 这里我们计算最后一步 (k=PRED_HORIZON-1) 的误差
#     k_eval = PRED_HORIZON - 1
#
#     rmse_Y_last_step = np.sqrt(np.mean((pred_ego_pos_m[:, k_eval] - true_ego_pos_m[:, k_eval]) ** 2))
#     valid_true_Y = np.abs(true_ego_pos_m[:, k_eval]) > 1e-6  # Avoid division by zero
#     mape_Y_last_step = np.mean(np.abs(
#         (pred_ego_pos_m[valid_true_Y, k_eval] - true_ego_pos_m[valid_true_Y, k_eval]) / true_ego_pos_m[
#             valid_true_Y, k_eval])) * 100
#
#     rmse_sp_last_step = np.sqrt(np.mean((pred_spacing_m[:, k_eval] - true_spacing_m[:, k_eval]) ** 2))
#     valid_true_sp = np.abs(true_spacing_m[:, k_eval]) > 1e-6
#     mape_sp_last_step = np.mean(np.abs(
#         (pred_spacing_m[valid_true_sp, k_eval] - true_spacing_m[valid_true_sp, k_eval]) / true_spacing_m[
#             valid_true_sp, k_eval])) * 100
#
#     print(f"\nPosition & Spacing Error at Step K={k_eval + 1}:")
#     print(f"  Position Error -- RMSE: {rmse_Y_last_step:.4f} m, MAPE: {mape_Y_last_step:.2f}%")
#     print(f"  Spacing  Error -- RMSE: {rmse_sp_last_step:.4f} m, MAPE: {mape_sp_last_step:.2f}%")
#     rmse_p= rmse_Y_last_step
#     mape_p= mape_Y_last_step
#     # 保存到 Excel (例如，保存所有K步的预测和真实值)
#     # 为了简化，我们只保存最后一步的比较数据
#     df_data = {
#         f"Pred Speed (m/s) Step{k_eval + 1}": pred_speeds_fused_mps[:, k_eval].numpy(),
#         f"True Speed (m/s) Step{k_eval + 1}": y_true_speeds_all[:, k_eval].numpy(),
#         f"Predicted Ego Y (m) Step{k_eval + 1}": pred_ego_pos_m[:, k_eval],
#         f"True Ego Y (m) Step{k_eval + 1}": true_ego_pos_m[:, k_eval],
#         f"Pred Spacing (m) Step{k_eval + 1}": pred_spacing_m[:, k_eval],
#         f"True Spacing (m) Step{k_eval + 1}": true_spacing_m[:, k_eval],
#     }
#     # 可以扩展为保存所有K步
#     # for k_idx in range(PRED_HORIZON):
#     #     df_data[f"Pred Speed (m/s) Step{k_idx+1}"] = pred_speeds_fused_mps[:, k_idx].numpy()
#     #     df_data[f"True Speed (m/s) Step{k_idx+1}"] = y_true_speeds_all[:, k_idx].numpy()
#     #     # ... add position and spacing for each step
#
#     df_pos = pd.DataFrame(df_data)
#     sheet_name = dataset_name  # 使用数据集名作为 sheet 名称 (例如 "data_5")
#
#     # 如果文件存在则在其基础上添加新 sheet，否则新建文件
#     # 'a'模式表示追加, 'w'模式表示写入(会覆盖)
#     # if_sheet_exists='replace' 可以用来替换同名sheet，需要 pandas >= 1.3.0 和 openpyxl
#     try:
#         with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
#             df_pos.to_excel(writer, sheet_name=sheet_name, index=False)
#     except FileNotFoundError:  # 如果文件不存在，首次以 'w' 模式创建
#         with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
#             df_pos.to_excel(writer, sheet_name=sheet_name, index=False)
#
#     print(f"{dataset_name} 的位置预测结果已保存至 '{output_file}' 的 '{sheet_name}' 工作表。")
#     # Position predictions for {dataset_name} saved to '{output_excel_file}' in sheet '{sheet_name}'.
#
#     return rmse_p, mape_p  # 返回位置预测的评估指标
#
# all_datasets_metrics_summary = []  # 用于存储所有数据集的评估指标字典列表
#
# def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae,speed_mape ,pos_rmse, pos_mape):
#     # 将单个数据集的评估指标存入列表
#     metrics = {
#         "数据集 (Dataset)": dataset_name,
#         "速度MSE (Speed_MSE)": speed_mse,
#         "速度RMSE (Speed_RMSE)": speed_rmse,
#         "速度MAE (Speed_MAE)": speed_mae,
#         "速度MAE (Speed_MAPE)": speed_mape,
#         "位置RMSE (m) (Position_RMSE_m)": pos_rmse,
#         "位置MAPE (%) (Position_MAPE_percent)": pos_mape
#     }
#     all_datasets_metrics_summary.append(metrics)
#
#
# def save_all_metrics_to_csv(filepath="evaluation_summary.csv"):
#     # 将所有数据集的评估指标汇总保存到CSV文件
#     if not all_datasets_metrics_summary:
#         print("没有评估指标可以保存。")  # No metrics to save.
#         return
#     df_metrics = pd.DataFrame(all_datasets_metrics_summary)
#     df_metrics.to_csv(filepath, index=False, encoding='utf-8-sig')  # utf-8-sig 确保中文在Excel中正确显示
#     print(f"所有数据集的评估指标汇总已保存至 {filepath}")  # All evaluation metrics saved to {filepath}
#
#
# # --- 主流程 ---
# if __name__ == "__main__":
#     torch.manual_seed(42)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
#     print(data_files)
#     if not data_files:
#         print(f"在目录 {DATA_DIR} 中未找到 .mat 文件。")  # No .mat files found in the directory.
#         exit()
#
#     # 定义位置预测结果的Excel文件路径
#     position_predictions_excel_path = os.path.join(RESULTS_DIR, "pred_positions_all_datasets.xlsx")
#     # 如果旧的汇总Excel文件存在，可以选择删除以重新开始，或者让程序追加/替换sheet
#     # 当前的 compute_future_positions_and_save 实现会替换同名sheet或追加新sheet
#
#     print(f"找到以下数据集文件: {data_files}")  # Found the following dataset files:
#
#     # 遍历每个找到的数据文件
#     for data_file_path in data_files:
#         dataset_filename = os.path.basename(data_file_path)  # 获取文件名，例如 "data_5.mat"
#         dataset_name_clean = dataset_filename.replace(".mat", "")  # 去掉 .mat 后缀，例如 "data_5"
#
#         print(f"\n==================== 开始处理数据集: {dataset_filename} ====================")
#         # Processing dataset: ...
#
#         data = sio.loadmat(data_file_path)
#         # Trandata（每列长度为50） ：当前车速(0) 、跟车距离(1) 、速度差(2) 、自车加速度(3) 、自车位置(4) 、前车速度(5)、前车加速度(6)、前车位置(7)
#         # lable data（每列长度为5） ：自车车速(0) 、前车距离(1) 、自车加速度(2) 、自车位置(3) 、前车速度(4) 、前车未来位置(5)
#         raw_all_ft = torch.tensor(data['train_data'], dtype=torch.float32)  # (N_all, 50, 8)
#         lab_all_ft = torch.tensor(data['lable_data'], dtype=torch.float32)  # (N_all, 5, 6)
#
#         # LSTM 输入特征选择 (与原代码一致): v_n, s_safe, Δv, acc_n, v_lead
#         # Indices for raw_all_ft: 0, 1, 2, 3, 5
#         seq_ft = raw_all_ft[:, :, [0, 1, 2, 3, 5]].clone()  # (N_all, 50, 5)
#         # 目标: 未来K步自车速度
#         y_multistep_ftps = lab_all_ft[:, :, 0].clone()  # (N_all, K) col 0 is 自车车速
#
#         # 用于IDM多步迭代的初始安全距离 (来自输入序列的最后一步)
#         s_safe_initial_ft = seq_ft[:, -1, 1].clone()  # (N_all,) col 1 is 跟车距离 (s_safe)
#         # 用于IDM多步迭代的初始前车速度 (来自输入序列的最后一步, 假设其在K步内不变)
#         v_lead_initial_ftps = seq_ft[:, -1, 4].clone()  # (N_all,) col 4 is 前车速度
#
#         # --- 单位转换 ft→m, ft/s→m/s ---
#         seq_mps = seq_ft.clone()
#         seq_mps[:, :, [0, 2, 3, 4]] *= 0.3048  # 速度, 速度差, 加速度(假设是ft/s^2), 前车速度
#         seq_mps[:, :, 1] *= 0.3048  # 跟车距离
#
#         y_multistep_mps = y_multistep_ftps * 0.3048
#         s_safe_initial_m = s_safe_initial_ft * 0.3048
#         v_lead_initial_mps = v_lead_initial_ftps * 0.3048
#         PRED_HORIZON =y_multistep_ftps.shape[1]
#
#         raw_all_m = raw_all_ft.clone()
#         raw_all_m[:, :, [0, 2, 3, 5, 6]] *= 0.3048  # Speeds, delta_v, accelerations
#         raw_all_m[:, :, [1, 4, 7]] *= 0.3048  # Distances, positions
#
#         lab_all_m = lab_all_ft.clone()
#         lab_all_m[:, :, [0, 2, 4]] *= 0.3048  # Speeds, accelerations
#         lab_all_m[:, :, [1, 3, 5]] *= 0.3048  # Distances, positions
#
#         # 只用前 10% 加速示例 (与原代码一致)
#         N_total = seq_mps.size(0)
#         N = int(N_total * 1)
#         seq_mps, y_multistep_mps, s_safe_initial_m, v_lead_initial_mps = \
#             seq_mps[:N], y_multistep_mps[:N], s_safe_initial_m[:N], v_lead_initial_mps[:N]
#
#         # 保留原始单位数据用于后续位置计算 (也取前10%)
#         raw_for_pos_calc_m = raw_all_m[:N]
#         lab_for_pos_calc_m = lab_all_m[:N]
#
#         # 划分 80/20
#         split = int(N * 0.8)
#         train_size = split
#
#         train_seq = seq_mps[:split].to(device)
#         test_seq = seq_mps[split:].to(device)
#
#         train_y_multistep = y_multistep_mps[:split].to(device)
#         test_y_multistep = y_multistep_mps[split:].to(device)
#
#         train_s_safe_initial = s_safe_initial_m[:split].to(device)
#         test_s_safe_initial = s_safe_initial_m[split:].to(device)
#
#         train_v_lead_initial = v_lead_initial_mps[:split].to(device)
#         test_v_lead_initial = v_lead_initial_mps[split:].to(device)
#
#         # 创建 TensorDataset 和 DataLoader
#         train_ds = torch.utils.data.TensorDataset(train_seq, train_y_multistep, train_s_safe_initial,
#                                                   train_v_lead_initial)
#         test_ds = torch.utils.data.TensorDataset(test_seq, test_y_multistep, test_s_safe_initial, test_v_lead_initial)
#
#         train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
#         test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
#
#         # 模型/优化器设定
#         input_dim = train_seq.size(2)  # 特征数量
#         hidden_dim = 128
#         model = HybridIDMModel(input_dim, hidden_dim, num_layers=1, output_dim=PRED_HORIZON).to(device)
#         initialize_weights(model)
#         optimizer = optim.Adam(model.parameters(), lr=5e-4)
#
#         # 训练 & 评估
#         model = train_model(model, train_loader, optimizer, num_epochs=100, alpha_decay_loss=0.05)  # 调整 epoch 和 decay
#         speed_mse, speed_rmse, speed_mae, speed_mape= evaluate_model(model, test_loader, alpha_decay_loss=0.05,dataset_name=dataset_name_clean,
#                                                               results_dir=RESULTS_DIR)
#
#         # 将原始单位的、被选择的10%数据传递给位置计算函数
#         # 注意，pos_calc函数内部会根据train_size来划分测试集部分
#         pos_rmse, pos_mape = compute_position_and_spacing_and_save(
#             model,
#             test_loader,  # 传递dataloader
#             raw_all_ft,  # 传递未经tensor selection的完整原始数据 (ft单位)
#             lab_all_ft,  # 传递未经tensor selection的完整标签数据 (ft单位)
#             train_size,  # 告知函数训练集大小，以正确索引测试部分
#             dt=DT,
#             output_file="predictions_multistep_extended_PIDLSTM_IDM.xlsx",
#             dataset_name=dataset_name_clean
#         )
#     summary_metrics_csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_all_datasets_PLSTM_IDM.csv")
#     save_all_metrics_to_csv(summary_metrics_csv_path)
#     print("\n所有数据集处理完毕。")  # All datasets processed..


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import pandas as pd
# import numpy as np
# import glob  # 用于查找文件路径
# import os  # 操作系统接口，用于路径操作和环境变量
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许多个OpenMP库存在，避免某些环境下的冲突
#
# # --- 全局路径定义 ---
# DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # 数据集存放目录
# RESULTS_DIR = "E:\\pythonProject1\\results_ngsim"  # 实验结果保存目录
#
# # 确保结果目录存在
# os.makedirs(RESULTS_DIR, exist_ok=True)
#
# # --- 全局常量 ---
# DT = 0.1  # 时间步长 (s) - 从原代码推断
#
#
# # --- 数据检查函数 ---
# def check_data(data, name="data"):
#     """ 检查数据中是否包含NaN或Inf值 """
#     print(f"正在检查 {name} 是否包含 NaN 或 Inf 值...")  # Checking {name} for NaN or Inf values...
#     print(f"包含 NaN: {torch.isnan(data).any().item()}")  # Has NaN: ...
#     print(f"包含 Inf: {torch.isinf(data).any().item()}")  # Has Inf: ...
#
#
# # --- 固定 IDM 参数预测函数 ---
# def idm_fixed(v_n, s_safe, delta_v,
#               v_desired=12.64798288, T=0.50284384, a_max=0.10033688,
#               b_safe=4.98937183, delta=1.0, s0=0.13082412,
#               delta_t=DT):
#     """
#     使用固定的参数集执行一步IDM（智能驾驶员模型）预测。
#     :param v_n: 当前自车速度 (m/s)
#     :param s_safe: 当前实际间距 (m)
#     :param delta_v: 当前速度差 (前车速度 - 自车速度, m/s)
#     :param v_desired:期望速度 (m/s)
#     :param T: 安全时间余量 (s)
#     :param a_max: 最大加速度 (m/s^2)
#     :param b_safe: 舒适减速度 (m/s^2)
#     :param delta: 加速度指数
#     :param s0: 最小静止间距 (m)
#     :param delta_t: 时间步长 (s)
#     :return: 下一时间步的预测自车速度 (m/s)
#     """
#     device = v_n.device  # 获取输入张量所在的设备
#     # 把常数转换为与输入数据相同设备和类型的0维张量
#     v_desired = torch.tensor(v_desired, device=device, dtype=v_n.dtype)
#     T = torch.tensor(T, device=device, dtype=v_n.dtype)
#     a_max = torch.tensor(a_max, device=device, dtype=v_n.dtype).clamp(min=1e-6)  # 防止a_max为0或负
#     b_safe = torch.tensor(b_safe, device=device, dtype=v_n.dtype).clamp(min=1e-6)  # 防止b_safe为0或负
#     s0 = torch.tensor(s0, device=device, dtype=v_n.dtype)
#     delta_param = torch.tensor(delta, device=device, dtype=v_n.dtype)  # 避免与python关键字delta冲突
#     delta_t_tensor = torch.tensor(delta_t, device=device, dtype=v_n.dtype)
#
#     s_safe = s_safe.clamp(min=1e-6)  # 确保s_safe是正的，防止除零或log(0)等问题
#
#     # IDM模型中的期望间距 s*
#     s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe) + 1e-6)  # 加一个小量避免除零
#     s_star = s_star.clamp(min=0.0)  # s_star不能为负
#
#     # 计算速度比 (v_n / v_desired)，处理v_desired可能为0的情况
#     v_n_ratio = torch.zeros_like(v_n)
#     mask_v_desired_nonzero = v_desired.abs() > 1e-6  # 创建一个掩码，标记v_desired非零的位置
#     if mask_v_desired_nonzero.any():  # 只有当v_desired非零时才计算比率
#         v_n_ratio[mask_v_desired_nonzero] = (v_n[mask_v_desired_nonzero] / v_desired[mask_v_desired_nonzero])
#
#     # IDM加速度项
#     acceleration_term = a_max * (
#             1 - v_n_ratio ** delta_param - (s_star / s_safe) ** 2  # 使用delta_param
#     )
#     # 计算下一时间步的速度
#     v_follow = v_n + delta_t_tensor * acceleration_term
#     return v_follow.clamp(min=0.0)  # 速度不能为负
#
#
# # --- 定义新融合模型 ---
# class HybridIDMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):  # output_dim 即为预测步长 K
#         super(HybridIDMModel, self).__init__()
#         self.pred_horizon = output_dim  # 存储预测步长 K
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, self.pred_horizon)  # LSTM输出K步预测
#         self.alpha_raw = nn.Parameter(torch.tensor(0.0))  # 可学习的 alpha 原始值，后续通过sigmoid激活
#
#         # IDM参数 (固定值)
#         self.v_desired_idm = 12.64798288
#         self.T_idm = 0.50284384
#         self.a_max_idm = 0.10033688
#         self.b_safe_idm = 4.98937183
#         self.delta_idm = 1.0  # IDM模型中的指数delta
#         self.s0_idm = 0.13082412
#
#     def forward(self, x, s_safe_initial, v_lead_initial):
#         """
#         模型的前向传播。
#         :param x:  输入序列，shape=(batch, seq_len, input_dim)
#                    x[:, -1, 0] 是当前自车速度 (v_n_t)
#                    x[:, -1, 1] 是当前跟车距离 (s_t) (注意: IDM使用的是s_safe_initial)
#                    x[:, -1, 4] 是当前前车速度 (v_lead_t) (注意: IDM使用的是v_lead_initial)
#         :param s_safe_initial: 当前安全距离 (真实观测值，用于IDM多步迭代的起始), shape=(batch,)
#         :param v_lead_initial: 当前前车速度 (真实观测值，用于IDM多步迭代，假设未来K步不变), shape=(batch,)
#         :return:
#           y_lstm_multistep: LSTM直接输出的K步预测速度，shape=(batch, K)
#           y_idm_multistep: IDM迭代K步的预测速度，shape=(batch, K)
#           alpha: 当前学习到的权重标量 (经过sigmoid)
#         """
#         batch_size = x.size(0)  # 获取批量大小
#         device = x.device  # 获取输入数据所在的设备
#
#         # LSTM K步预测
#         out, _ = self.lstm(x)  # LSTM输出 shape: (batch, seq_len, hidden_dim)
#         # 取序列最后一个时间步的LSTM输出，通过全连接层得到K步预测
#         y_lstm_multistep = self.fc(out[:, -1, :])  # shape: (batch, K)
#
#         # IDM K步迭代预测
#         y_idm_multistep_list = []  # 用于存储IDM每一步的预测结果
#
#         # IDM迭代的初始状态
#         v_ego_current_idm = x[:, -1, 0].clone()  # 当前自车速度 (m/s)，从输入序列的最后一个时间步获取
#         s_current_idm = s_safe_initial.clone()  # 当前实际间距 (m)，使用传入的初始安全距离
#         v_lead_constant_idm = v_lead_initial.clone()  # 当前前车速度 (m/s)，使用传入的初始前车速度，假设未来K步不变
#
#         # 迭代K步进行IDM预测
#         for k_step in range(self.pred_horizon):  # 使用存储的预测步长 self.pred_horizon
#             delta_v_idm = v_lead_constant_idm - v_ego_current_idm  # 计算当前速度差
#
#             # 调用idm_fixed进行单步预测
#             v_ego_next_pred_idm = idm_fixed(
#                 v_ego_current_idm, s_current_idm, delta_v_idm,
#                 v_desired=self.v_desired_idm, T=self.T_idm, a_max=self.a_max_idm,
#                 b_safe=self.b_safe_idm, delta=self.delta_idm, s0=self.s0_idm, delta_t=DT  # 传入IDM参数
#             )
#             y_idm_multistep_list.append(v_ego_next_pred_idm.unsqueeze(1))  # 存储预测结果并增加维度
#
#             # 更新IDM下一次迭代的输入状态
#             # 下一步的间距 s_t+1 = s_t + (v_lead_t - v_ego_t) * dt
#             # 注意: 这里 v_ego_t 是当前迭代步开始时的速度 v_ego_current_idm
#             s_current_idm = s_current_idm + (v_lead_constant_idm - v_ego_current_idm) * DT
#             s_current_idm = s_current_idm.clamp(min=1e-6)  # 确保间距不为负或零
#
#             v_ego_current_idm = v_ego_next_pred_idm  # 更新自车速度为当前步的预测值
#
#         y_idm_multistep = torch.cat(y_idm_multistep_list, dim=1)  # 将K步预测结果拼接, shape: (batch, K)
#
#         alpha = torch.sigmoid(self.alpha_raw)  # 对可学习参数alpha_raw应用sigmoid函数，使其值在(0,1)之间
#         return y_lstm_multistep, y_idm_multistep, alpha
#
#
# def initialize_weights(model):
#     """ 初始化模型权重 """
#     for name, param in model.named_parameters():
#         if "weight" in name and param.dim() > 1:  # Xavier均匀初始化适用于多维权重
#             nn.init.xavier_uniform_(param)
#         elif "bias" in name:  # 偏置项初始化为0
#             nn.init.constant_(param, 0)
#
#
# # --- 训练函数 ---
# def train_model(model, train_loader, optimizer, pred_horizon, num_epochs=30, alpha_decay_loss=0.1):
#     """
#     训练模型。
#     :param model: 待训练的模型
#     :param train_loader: 训练数据加载器
#     :param optimizer: 优化器
#     :param pred_horizon: 当前数据集的预测步长 K
#     :param num_epochs: 训练轮数
#     :param alpha_decay_loss: 用于计算损失权重的衰减因子
#     :return: 训练好的模型
#     """
#     model.train()  # 设置模型为训练模式
#
#     # 生成损失权重 w_k = exp(-alpha_decay_loss * k), k从0到K-1
#     # 这些权重用于给多步预测中不同时间步的损失赋予不同重要性
#     loss_weights = torch.exp(-alpha_decay_loss * torch.arange(pred_horizon, dtype=torch.float32)).to(
#         next(model.parameters()).device)  # 将权重张量移至模型参数所在的设备
#     loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * pred_horizon  # 归一化使得平均权重为1, 加1e-9防止除零
#
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0  # 初始化当前轮的总损失
#         current_alpha = 0.0  # 用于记录最后一个批次的alpha值
#         for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in train_loader:
#             # batch_y_multistep 的形状应为 (batch, pred_horizon)
#             optimizer.zero_grad()  # 清空之前的梯度
#
#             # 模型前向传播
#             y_lstm_multistep, y_idm_multistep, alpha = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)
#             current_alpha = alpha.item()  # 获取当前alpha值
#
#             # 多步加权MSE损失
#             # L_lstm_true = sum_k w_k * (y_lstm_k - y_true_k)^2
#             # L_lstm_idm = sum_k w_k * (y_lstm_k - y_idm_k)^2
#             # LSTM预测与真实值之间的损失
#             loss_lstm_vs_true = ((y_lstm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
#             # LSTM预测与IDM预测之间的损失 (物理一致性损失)
#             loss_lstm_vs_idm = ((y_lstm_multistep - y_idm_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
#
#             # 总损失是上述两部分损失的加权和
#             loss = alpha * loss_lstm_vs_true + (1 - alpha) * loss_lstm_vs_idm
#
#             loss.backward()  # 反向传播计算梯度
#             optimizer.step()  # 更新模型参数
#             epoch_loss += loss.item()  # 累加批次损失
#         print(f"轮次 {epoch + 1}/{num_epochs}  损失: {epoch_loss / len(train_loader):.6f}  α={current_alpha:.4f}")
#         # Epoch {epoch + 1}/{num_epochs}  Loss: ...
#     return model
#
#
# # --- 测试/评估函数 ---
# def evaluate_model(model, test_loader, pred_horizon, alpha_decay_loss=0.1, dataset_name="", results_dir=""):
#     """
#     评估模型性能。
#     :param model: 待评估的模型
#     :param test_loader: 测试数据加载器
#     :param pred_horizon: 当前数据集的预测步长 K
#     :param alpha_decay_loss: 用于计算损失权重的衰减因子
#     :param dataset_name: 数据集名称，用于保存结果
#     :param results_dir: 结果保存目录
#     :return: mse_val, rmse_val, mae_val, mape_p (速度预测的综合评估指标)
#     """
#     model.eval()  # 设置模型为评估模式
#     all_pred_lstm, all_pred_idm, all_true, all_alpha_values = [], [], [], []  # 初始化列表存储所有预测和真实值
#
#     # 生成损失权重，与训练时一致
#     loss_weights = torch.exp(-alpha_decay_loss * torch.arange(pred_horizon, dtype=torch.float32)).to(
#         next(model.parameters()).device)
#     loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * pred_horizon  # 加1e-9防止除零
#
#     total_mse_lstm_vs_true = 0  # 初始化总MSE (LSTM vs True)
#     total_mse_lstm_vs_idm = 0  # 初始化总MSE (LSTM vs IDM)
#     total_combined_mse_weighted = 0  # 初始化加权总MSE (Combined vs True)
#
#     with torch.no_grad():  # 在评估模式下不计算梯度
#         for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in test_loader:
#             y_lstm_multistep, y_idm_multistep, alpha_tensor = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)
#
#             all_pred_lstm.append(y_lstm_multistep.cpu())  # 收集LSTM预测
#             all_pred_idm.append(y_idm_multistep.cpu())  # 收集IDM预测
#             all_true.append(batch_y_multistep.cpu())  # 收集真实值
#             all_alpha_values.append(alpha_tensor.cpu().item())  # 收集alpha值
#
#             # 计算加权MSE损失 (用于报告综合指标)
#             loss_lstm_vs_true_weighted = (
#                         (y_lstm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
#             loss_lstm_vs_idm_weighted = ((y_lstm_multistep - y_idm_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
#
#             # 计算融合预测的加权损失
#             y_combined_multistep = alpha_tensor * y_lstm_multistep + (1 - alpha_tensor) * y_idm_multistep
#             loss_combined_vs_true_weighted = (
#                     (y_combined_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
#
#             # 累加损失 (乘以批量大小以正确计算平均值)
#             total_mse_lstm_vs_true += loss_lstm_vs_true_weighted.item() * batch_x.size(0)
#             total_mse_lstm_vs_idm += loss_lstm_vs_idm_weighted.item() * batch_x.size(0)
#             total_combined_mse_weighted += loss_combined_vs_true_weighted.item() * batch_x.size(0)
#
#     num_samples = len(test_loader.dataset)  # 测试样本总数
#     avg_mse_lstm_vs_true_weighted = total_mse_lstm_vs_true / num_samples
#     avg_mse_lstm_vs_idm_weighted = total_mse_lstm_vs_idm / num_samples
#     avg_combined_mse_summary = total_combined_mse_weighted / num_samples  # 这是用于总结的加权MSE
#     final_alpha_mean = np.mean(all_alpha_values)  # 计算平均alpha值
#
#     # 将所有批次的预测和真实值拼接起来
#     y_pred_lstm_cat = torch.cat(all_pred_lstm)  # shape: (N_test, K)
#     y_pred_idm_cat = torch.cat(all_pred_idm)  # shape: (N_test, K)
#     y_true_cat = torch.cat(all_true)  # shape: (N_test, K)
#
#     # 使用最终的平均alpha进行融合预测 (主要用于绘图和单步指标)
#     y_pred_combined_cat = final_alpha_mean * y_pred_lstm_cat + (1 - final_alpha_mean) * y_pred_idm_cat
#
#     # 计算综合评估指标 (基于所有样本和所有时间步的简单平均，非加权)
#     mse_val_overall = torch.mean((y_pred_combined_cat - y_true_cat).pow(2)).item()  # 整体MSE (简单平均)
#     rmse_val_overall = np.sqrt(mse_val_overall)  # 整体RMSE
#     mae_val_overall = torch.mean(torch.abs(y_pred_combined_cat - y_true_cat)).item()  # 整体MAE
#
#     # 整体MAPE，处理分母为0的情况
#     abs_error_overall = torch.abs(y_pred_combined_cat - y_true_cat)
#     abs_true_overall = torch.abs(y_true_cat)
#     valid_mape_mask_overall = abs_true_overall > 1e-6
#     if torch.sum(valid_mape_mask_overall) > 0:
#         mape_p_overall = torch.mean(
#             abs_error_overall[valid_mape_mask_overall] / abs_true_overall[valid_mape_mask_overall]) * 100
#         mape_p_overall = mape_p_overall.item()
#     else:
#         mape_p_overall = float('nan')
#
#     print(f"\n--- 测试结果摘要 (综合指标，简单平均所有步) ---")
#     # Test Results Summary (Overall metrics, simple average over all steps):
#     print(
#         f"  融合预测 vs 真实 -- MSE: {mse_val_overall:.4f}, RMSE: {rmse_val_overall:.4f}, MAE: {mae_val_overall:.4f}, MAPE: {mape_p_overall if not np.isnan(mape_p_overall) else 'N/A'}%")
#     print(
#         f"  (用于损失的加权MSE -- LSTM vs True: {avg_mse_lstm_vs_true_weighted:.4f}, LSTM vs IDM: {avg_mse_lstm_vs_idm_weighted:.4f}, Combined vs True: {avg_combined_mse_summary:.4f})")
#     print(f"  最终平均 α={final_alpha_mean:.4f}")  # Final mean α=...
#
#     print(f"\n--- 每一步的详细评估指标 (速度预测) ---")
#
#     # Per-step detailed evaluation metrics (speed prediction)
#     for k_step in range(pred_horizon):
#         y_pred_lstm_step_k = y_pred_lstm_cat[:, k_step]
#         y_pred_idm_step_k = y_pred_idm_cat[:, k_step]
#         y_pred_combined_step_k = y_pred_combined_cat[:, k_step]
#         y_true_step_k = y_true_cat[:, k_step]
#
#         mse_step_lstm = nn.MSELoss()(y_pred_lstm_step_k, y_true_step_k).item()
#         mse_step_idm = nn.MSELoss()(y_pred_idm_step_k, y_true_step_k).item()
#         mse_step_combined = nn.MSELoss()(y_pred_combined_step_k, y_true_step_k).item()
#         rmse_step_combined = np.sqrt(mse_step_combined)
#         mae_step_combined = torch.mean(torch.abs(y_pred_combined_step_k - y_true_step_k)).item()
#
#         abs_error_step = torch.abs(y_pred_combined_step_k - y_true_step_k)
#         abs_true_step = torch.abs(y_true_step_k)
#         valid_mape_mask_step = abs_true_step > 1e-6
#         if torch.sum(valid_mape_mask_step) > 0:
#             mape_step_combined = torch.mean(
#                 abs_error_step[valid_mape_mask_step] / abs_true_step[valid_mape_mask_step]) * 100
#             mape_step_combined = mape_step_combined.item()
#         else:
#             mape_step_combined = float('nan')
#
#         print(
#             f"  步骤 {k_step + 1} 预测 vs 真实 -- LSTM MSE: {mse_step_lstm:.4f}, IDM MSE: {mse_step_idm:.4f}")
#         print(
#             f"    融合 MSE: {mse_step_combined:.4f}, RMSE: {rmse_step_combined:.4f}, MAE: {mae_step_combined:.4f}, MAPE: {mape_step_combined if not np.isnan(mape_step_combined) else 'N/A'}%")
#
#     # 绘图 (例如，只绘制第一个预测步 K=0)
#     k_plot = 0  # 选择绘制的预测步索引
#     plt.figure(figsize=(12, 7))
#     plt.plot(y_true_cat[:100, k_plot].numpy(), '--o', label=f'真实值 (步骤 {k_plot + 1})')  # True (Step ...)
#     plt.plot(y_pred_lstm_cat[:100, k_plot].numpy(), '-x',
#              label=f'LSTM 预测 (步骤 {k_plot + 1})')  # LSTM Pred (Step ...)
#     plt.plot(y_pred_idm_cat[:100, k_plot].numpy(), '-s', label=f'IDM 预测 (步骤 {k_plot + 1})')  # IDM Pred (Step ...)
#     plt.plot(y_pred_combined_cat[:100, k_plot].numpy(), '-.',
#              label=f'融合预测 (步骤 {k_plot + 1}, α={final_alpha_mean:.2f})')  # Combined Pred (Step ...)
#     plt.title(f'速度预测对比 (前100个样本, 步骤 {k_plot + 1})')  # Speed Prediction Comparison ...
#     plt.xlabel("样本索引")  # Sample Index
#     plt.ylabel("速度 (m/s)")  # Speed (m/s)
#     plt.legend()
#     plt.grid()
#     plot_filename = os.path.join(results_dir, f"{dataset_name}_speed_comparison_PLSTM_IDM.png")
#     plt.savefig(plot_filename)  # 保存图像
#     print(f"速度对比图已保存至 {plot_filename}")  # Speed comparison plot saved to ...
#     plt.close()  # 关闭图像，释放内存
#
#     # 返回用于存储的综合评估指标 (使用加权MSE计算的RMSE，以及整体MAE, MAPE)
#     return avg_combined_mse_summary, np.sqrt(avg_combined_mse_summary), mae_val_overall, mape_p_overall
#
#
# # === 修改后的 compute_position_and_spacing_and_save ===
# def compute_position_and_spacing_and_save(model,
#                                           test_loader,
#                                           raw_data_all,  # 完整原始数据 (ft单位)
#                                           label_data_all,  # 完整标签数据 (ft单位)
#                                           train_size,  # 训练集大小，用于定位测试数据在原始数据中的索引
#                                           pred_horizon,  # 当前数据集的预测步长 K
#                                           dt=DT,
#                                           output_file="predictions_multistep_extended.xlsx",
#                                           dataset_name=""):
#     """
#     计算未来K步的位置和间距，并保存结果。
#     使用英尺(ft)进行位移计算以匹配原始数据单位，最后转换为米(m)进行评估和保存。
#     :param model: 训练好的模型
#     :param test_loader: 测试数据加载器 (提供m/s单位的输入和标签)
#     :param raw_data_all: 完整的原始输入数据 (N_all, seq_len_raw, feat_raw)，单位ft, ft/s
#     :param label_data_all: 完整的原始标签数据 (N_all, K_lab, feat_lab)，单位ft, ft/s
#     :param train_size: 训练集样本数量
#     :param pred_horizon: 预测视界 K
#     :param dt: 时间步长 (s)
#     :param output_file: Excel输出文件名
#     :param dataset_name: 数据集名称，用作Excel中的sheet名
#     :return: rmse_p_last_step, mape_p_last_step (最后一步位置预测的评估指标)
#     """
#     model.eval()  # 设置模型为评估模式
#
#     test_start_idx_in_all_data = train_size  # 测试数据在完整数据集中的索引起始点
#
#     y_lstm_list_mps, y_idm_list_mps, y_true_speeds_list_mps = [], [], []
#     initial_ego_pos_ft_collected = []
#     initial_lead_pos_ft_collected = []
#     initial_ego_speed_ftps_collected = []
#     initial_lead_speed_ftps_collected = []
#     true_future_ego_pos_ft_collected = []
#     true_future_spacing_ft_collected = []
#
#     with torch.no_grad():
#         for i, (batch_x_mps, batch_y_multistep_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps) in enumerate(
#                 test_loader):
#             y_lstm_k_mps, y_idm_k_mps, alpha_tensor = model(batch_x_mps, batch_s_safe_initial_m,
#                                                             batch_v_lead_initial_mps)
#             y_lstm_list_mps.append(y_lstm_k_mps.cpu())
#             y_idm_list_mps.append(y_idm_k_mps.cpu())
#             y_true_speeds_list_mps.append(batch_y_multistep_mps.cpu())
#
#             batch_start_idx_in_loader = i * test_loader.batch_size
#             current_batch_indices_in_all_data = np.arange(
#                 test_start_idx_in_all_data + batch_start_idx_in_loader,
#                 test_start_idx_in_all_data + batch_start_idx_in_loader + batch_x_mps.size(0)
#             )
#
#             # 从 raw_data_all (ft单位) 提取初始状态
#             initial_ego_pos_ft_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 4].cpu())
#             initial_lead_pos_ft_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 7].cpu())
#             initial_ego_speed_ftps_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 0].cpu())
#             initial_lead_speed_ftps_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 5].cpu())
#
#             # 从 label_data_all (ft单位) 提取未来真实位置和间距
#             true_future_ego_pos_ft_collected.append(
#                 label_data_all[current_batch_indices_in_all_data, :pred_horizon, 3].cpu())
#             true_future_spacing_ft_collected.append(
#                 label_data_all[current_batch_indices_in_all_data, :pred_horizon, 1].cpu())
#
#     y_lstm_all_mps = torch.cat(y_lstm_list_mps, dim=0)
#     y_idm_all_mps = torch.cat(y_idm_list_mps, dim=0)
#     y_true_speeds_all_mps = torch.cat(y_true_speeds_list_mps, dim=0)  # 这是真实的目标速度，用于Excel输出
#
#     alpha_val = model.alpha_raw.sigmoid().item()
#     pred_speeds_fused_mps = alpha_val * y_lstm_all_mps + (1 - alpha_val) * y_idm_all_mps
#
#     initial_ego_pos_ft = torch.cat(initial_ego_pos_ft_collected, dim=0)
#     initial_lead_pos_ft = torch.cat(initial_lead_pos_ft_collected, dim=0)
#     initial_ego_speed_ftps = torch.cat(initial_ego_speed_ftps_collected, dim=0)
#     initial_lead_speed_ftps = torch.cat(initial_lead_speed_ftps_collected, dim=0)
#
#     true_future_ego_pos_ft = torch.cat(true_future_ego_pos_ft_collected, dim=0)
#     true_future_spacing_ft = torch.cat(true_future_spacing_ft_collected, dim=0)
#
#     pred_ego_pos_k_steps_ft = torch.zeros_like(pred_speeds_fused_mps)
#     pred_lead_pos_k_steps_ft = torch.zeros_like(pred_speeds_fused_mps)
#     pred_spacing_k_steps_ft = torch.zeros_like(pred_speeds_fused_mps)
#
#     pred_speeds_fused_ftps = pred_speeds_fused_mps / 0.3048
#
#     current_ego_pos_ft = initial_ego_pos_ft.clone()
#     current_lead_pos_ft = initial_lead_pos_ft.clone()
#
#     # 前车假设在未来K步内以其在t=0时刻的速度匀速行驶
#     lead_speed_constant_ftps = initial_lead_speed_ftps
#
#     for k in range(pred_horizon):
#         # 自车使用每一步的预测速度 V(t_0 + k*dt) 来计算 P(t_0 + (k+1)*dt)
#         # 当 k=0, speed_ego_this_step_ftps = initial_ego_speed_ftps (即 V(t_0))
#         # 当 k>0, speed_ego_this_step_ftps = pred_speeds_fused_ftps[:, k-1] (即 V(t_0 + k*dt))
#         speed_ego_this_step_ftps = initial_ego_speed_ftps if k == 0 else pred_speeds_fused_ftps[:, k - 1]
#
#         disp_ego_ft = speed_ego_this_step_ftps * dt
#         disp_lead_ft = lead_speed_constant_ftps * dt  # 前车匀速
#
#         current_ego_pos_ft += disp_ego_ft
#         current_lead_pos_ft += disp_lead_ft
#
#         pred_ego_pos_k_steps_ft[:, k] = current_ego_pos_ft
#         pred_lead_pos_k_steps_ft[:, k] = current_lead_pos_ft
#         pred_spacing_k_steps_ft[:, k] = current_lead_pos_ft - current_ego_pos_ft
#
#     pred_ego_pos_m = pred_ego_pos_k_steps_ft.numpy() * 0.3048
#     true_ego_pos_m = true_future_ego_pos_ft.numpy() * 0.3048
#     pred_spacing_m = pred_spacing_k_steps_ft.numpy() * 0.3048
#     true_spacing_m = true_future_spacing_ft.numpy() * 0.3048
#
#     print(f"\n--- 每一步的位置和间距误差评估 ---")
#     # Per-step position and spacing error evaluation
#     for k_s in range(pred_horizon):
#         # 位置误差
#         pos_err_sq_step = (pred_ego_pos_m[:, k_s] - true_ego_pos_m[:, k_s]) ** 2
#         rmse_Y_step = np.sqrt(np.mean(pos_err_sq_step))
#
#         valid_true_Y_step_mask = np.abs(true_ego_pos_m[:, k_s]) > 1e-6
#         if np.sum(valid_true_Y_step_mask) > 0:
#             mape_Y_step = np.mean(np.abs(
#                 (pred_ego_pos_m[valid_true_Y_step_mask, k_s] - true_ego_pos_m[valid_true_Y_step_mask, k_s]) /
#                 true_ego_pos_m[valid_true_Y_step_mask, k_s])) * 100
#         else:
#             mape_Y_step = float('nan')
#
#         # 间距误差
#         spacing_err_sq_step = (pred_spacing_m[:, k_s] - true_spacing_m[:, k_s]) ** 2
#         rmse_sp_step = np.sqrt(np.mean(spacing_err_sq_step))
#
#         valid_true_sp_step_mask = np.abs(true_spacing_m[:, k_s]) > 1e-6
#         if np.sum(valid_true_sp_step_mask) > 0:
#             mape_sp_step = np.mean(np.abs(
#                 (pred_spacing_m[valid_true_sp_step_mask, k_s] - true_spacing_m[valid_true_sp_step_mask, k_s]) /
#                 true_spacing_m[valid_true_sp_step_mask, k_s])) * 100
#         else:
#             mape_sp_step = float('nan')
#
#         print(f"  步骤 {k_s + 1}:")
#         print(
#             f"    位置误差 -- RMSE: {rmse_Y_step:.4f} m, MAPE: {mape_Y_step if not np.isnan(mape_Y_step) else 'N/A'}%")
#         print(
#             f"    间距误差 -- RMSE: {rmse_sp_step:.4f} m, MAPE: {mape_sp_step if not np.isnan(mape_sp_step) else 'N/A'}%")
#
#     # 计算最后一步的误差用于返回 (摘要指标)
#     k_eval = pred_horizon - 1
#     rmse_p_last_step = np.sqrt(np.mean((pred_ego_pos_m[:, k_eval] - true_ego_pos_m[:, k_eval]) ** 2))
#     valid_true_Y_last_mask = np.abs(true_ego_pos_m[:, k_eval]) > 1e-6
#     if np.sum(valid_true_Y_last_mask) > 0:
#         mape_p_last_step = np.mean(np.abs(
#             (pred_ego_pos_m[valid_true_Y_last_mask, k_eval] - true_ego_pos_m[valid_true_Y_last_mask, k_eval]) /
#             true_ego_pos_m[valid_true_Y_last_mask, k_eval])) * 100
#     else:
#         mape_p_last_step = float('nan')
#
#     print(f"\n--- 最后一步 (K={k_eval + 1}) 的位置和间距误差 (用于摘要) ---")
#     # Last step position and spacing error (for summary)
#     print(
#         f"  位置误差 -- RMSE: {rmse_p_last_step:.4f} m, MAPE: {mape_p_last_step if not np.isnan(mape_p_last_step) else 'N/A'}%")
#     # (间距的最后一步误差也可以打印，但函数只要求返回位置的)
#
#     # 保存所有K步的预测和真实值到Excel
#     df_data = {}
#     for k_idx in range(pred_horizon):
#         df_data[f"预测速度 (m/s) 步骤{k_idx + 1}"] = pred_speeds_fused_mps[:, k_idx].numpy()
#         df_data[f"真实速度 (m/s) 步骤{k_idx + 1}"] = y_true_speeds_all_mps[:, k_idx].numpy()
#         df_data[f"预测自车位置 Y (m) 步骤{k_idx + 1}"] = pred_ego_pos_m[:, k_idx]
#         df_data[f"真实自车位置 Y (m) 步骤{k_idx + 1}"] = true_ego_pos_m[:, k_idx]
#         df_data[f"预测间距 (m) 步骤{k_idx + 1}"] = pred_spacing_m[:, k_idx]
#         df_data[f"真实间距 (m) 步骤{k_idx + 1}"] = true_spacing_m[:, k_idx]
#
#     df_pos = pd.DataFrame(df_data)
#     sheet_name = dataset_name
#
#     try:
#         with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
#             df_pos.to_excel(writer, sheet_name=sheet_name, index=False)
#     except FileNotFoundError:
#         with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
#             df_pos.to_excel(writer, sheet_name=sheet_name, index=False)
#
#     print(f"{dataset_name} 的位置和间距预测结果已保存至 '{output_file}' 的 '{sheet_name}' 工作表。")
#     # Position and spacing predictions for {dataset_name} saved to ...
#
#     return rmse_p_last_step, mape_p_last_step
#
#
# # --- 存储和保存评估指标的辅助函数 ---
# all_datasets_metrics_summary = []  # 用于存储所有数据集的评估指标字典列表
#
#
# def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse_last_step,
#                           pos_mape_last_step):  # 参数名更新以反映其含义
#     """ 将单个数据集的评估指标存入列表 """
#     metrics = {
#         "数据集 (Dataset)": dataset_name,
#         "速度MSE (Speed_MSE_summary)": speed_mse,  # 综合指标
#         "速度RMSE (Speed_RMSE_summary)": speed_rmse,  # 综合指标
#         "速度MAE (Speed_MAE_overall)": speed_mae,  # 综合指标
#         "速度MAPE (%) (Speed_MAPE_overall_percent)": speed_mape,  # 综合指标
#         "末步位置RMSE (m) (Position_RMSE_last_step_m)": pos_rmse_last_step,
#         "末步位置MAPE (%) (Position_MAPE_last_step_percent)": pos_mape_last_step
#     }
#     all_datasets_metrics_summary.append(metrics)
#
#
# def save_all_metrics_to_csv(filepath="evaluation_summary_pID_LSTM_IDM.csv"):
#     """ 将所有数据集的评估指标汇总保存到CSV文件 """
#     if not all_datasets_metrics_summary:
#         print("没有评估指标可以保存。")  # No metrics to save.
#         return
#     df_metrics = pd.DataFrame(all_datasets_metrics_summary)
#     df_metrics.to_csv(filepath, index=False, encoding='utf-8-sig')  # utf-8-sig 确保中文在Excel中正确显示
#     print(f"所有数据集的评估指标汇总已保存至 {filepath}")  # All evaluation metrics saved to {filepath}
#
#
# # --- 主流程 ---
# if __name__ == "__main__":
#     torch.manual_seed(42)  # 设置随机种子以保证结果可复现
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
#     print(f"使用设备: {device}")  # Using device: ...
#
#     data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))  # 查找所有.mat数据文件
#     if not data_files:
#         print(f"在目录 {DATA_DIR} 中未找到 .mat 文件。程序将退出。")  # No .mat files found. Exiting.
#         exit()
#
#     print(f"找到以下数据集文件: {data_files}")  # Found the following dataset files:
#
#     # 定义位置预测结果的Excel文件完整路径
#     position_predictions_excel_path = os.path.join(RESULTS_DIR,
#                                                    "pred_positions_all_datasets_PLSTM_IDM_all_steps.xlsx")  # 更新文件名
#
#     # 遍历每个找到的数据文件
#     for data_file_path in data_files:
#         dataset_filename = os.path.basename(data_file_path)
#         dataset_name_clean = dataset_filename.replace(".mat", "")
#
#         print(f"\n==================== 开始处理数据集: {dataset_filename} ====================")
#         # Processing dataset: ...
#
#         data = sio.loadmat(data_file_path)
#         raw_all_ft = torch.tensor(data['train_data'], dtype=torch.float32)
#         lab_all_ft = torch.tensor(data['lable_data'], dtype=torch.float32)
#
#         seq_ft = raw_all_ft[:, :, [0, 1, 2, 3, 5]].clone()
#         y_multistep_ftps = lab_all_ft[:, :, 0].clone()
#
#         # --- 动态确定当前数据集的预测步长 K ---
#         current_pred_horizon = y_multistep_ftps.shape[1]
#         print(f"数据集 {dataset_filename} 的预测步长 K = {current_pred_horizon}")
#         # Prediction horizon K = ... for dataset ...
#
#         s_safe_initial_ft = seq_ft[:, -1, 1].clone()
#         v_lead_initial_ftps = seq_ft[:, -1, 4].clone()
#
#         seq_mps = seq_ft.clone()
#         seq_mps[:, :, [0, 2, 3, 4]] *= 0.3048
#         seq_mps[:, :, 1] *= 0.3048
#
#         y_multistep_mps = y_multistep_ftps * 0.3048
#         s_safe_initial_m = s_safe_initial_ft * 0.3048
#         v_lead_initial_mps = v_lead_initial_ftps * 0.3048
#
#         N_total = seq_mps.size(0)
#         N = int(N_total * 0.1)  # 使用全部数据 (原为1.0，可按需调整如0.1)
#         print(f"将使用 {N} / {N_total} 条数据进行训练和测试。")  # Will use ... / ... samples.
#
#         seq_mps_selected = seq_mps[:N]
#         y_multistep_mps_selected = y_multistep_mps[:N]
#         s_safe_initial_m_selected = s_safe_initial_m[:N]
#         v_lead_initial_mps_selected = v_lead_initial_mps[:N]
#
#         # 传递给位置计算的原始数据也应对应选择的N个样本
#         raw_all_ft_selected = raw_all_ft[:N]
#         lab_all_ft_selected = lab_all_ft[:N]
#
#         split_ratio = 0.8
#         train_size = int(N * split_ratio)
#
#         train_seq = seq_mps_selected[:train_size].to(device)
#         test_seq = seq_mps_selected[train_size:].to(device)
#         train_y_multistep = y_multistep_mps_selected[:train_size].to(device)
#         test_y_multistep = y_multistep_mps_selected[train_size:].to(device)
#         train_s_safe_initial = s_safe_initial_m_selected[:train_size].to(device)
#         test_s_safe_initial = s_safe_initial_m_selected[train_size:].to(device)
#         train_v_lead_initial = v_lead_initial_mps_selected[:train_size].to(device)
#         test_v_lead_initial = v_lead_initial_mps_selected[train_size:].to(device)
#
#         train_ds = torch.utils.data.TensorDataset(train_seq, train_y_multistep, train_s_safe_initial,
#                                                   train_v_lead_initial)
#         test_ds = torch.utils.data.TensorDataset(test_seq, test_y_multistep, test_s_safe_initial, test_v_lead_initial)
#         train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
#         test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
#
#         input_dim = train_seq.size(2)
#         hidden_dim = 128
#         # 实例化模型，传入当前数据集的预测步长 current_pred_horizon
#         model = HybridIDMModel(input_dim, hidden_dim, output_dim=current_pred_horizon, num_layers=1).to(device)
#         initialize_weights(model)
#         optimizer = optim.Adam(model.parameters(), lr=5e-4)
#
#         print(f"开始训练模型: {dataset_name_clean}...")  # Starting training for ...
#         model = train_model(model, train_loader, optimizer, pred_horizon=current_pred_horizon, num_epochs=50,
#                             alpha_decay_loss=0.05)
#
#         print(f"开始评估模型 (速度预测): {dataset_name_clean}...")  # Starting evaluation (speed prediction) for ...
#         speed_mse_summary, speed_rmse_summary, speed_mae_overall, speed_mape_overall = evaluate_model(
#             model, test_loader, pred_horizon=current_pred_horizon, alpha_decay_loss=0.05,
#             dataset_name=dataset_name_clean, results_dir=RESULTS_DIR
#         )
#
#         print(f"开始计算和评估位置/间距预测: {dataset_name_clean}...")  # Starting position/spacing computation for ...
#         pos_rmse_last_step, pos_mape_last_step = compute_position_and_spacing_and_save(
#             model,
#             test_loader,
#             raw_all_ft_selected,  # 使用选择后的原始数据
#             lab_all_ft_selected,  # 使用选择后的标签数据
#             train_size,
#             pred_horizon=current_pred_horizon,
#             dt=DT,
#             output_file=position_predictions_excel_path,
#             dataset_name=dataset_name_clean
#         )
#
#         # 存储当前数据集的评估指标
#         store_dataset_metrics(dataset_name_clean, speed_mse_summary, speed_rmse_summary, speed_mae_overall,
#                               speed_mape_overall, pos_rmse_last_step, pos_mape_last_step)
#         print(f"==================== 数据集 {dataset_filename} 处理完毕 ====================")
#         # Dataset {dataset_filename} processing finished.
#
#     summary_metrics_csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_all_datasets_PLSTM_IDM.csv")
#     save_all_metrics_to_csv(summary_metrics_csv_path)
#     print("\n所有数据集处理完毕。最终评估汇总已保存。")  # All datasets processed. Final evaluation summary saved.

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import glob  # 用于查找文件路径
import os  # 操作系统接口，用于路径操作和环境变量

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许多个OpenMP库存在，避免某些环境下的冲突

# --- 全局路径定义 ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # 数据集存放目录
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim_modified"  # 实验结果保存目录 (修改了目录名以区分)

# 确保结果目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- 全局常量 ---
DT = 0.1  # 时间步长 (s) - 从原代码推断


# --- 数据检查函数 ---
def check_data(data, name="data"):
    """ 检查数据中是否包含NaN或Inf值 """
    print(f"正在检查 {name} 是否包含 NaN 或 Inf 值...")
    print(f"包含 NaN: {torch.isnan(data).any().item()}")
    print(f"包含 Inf: {torch.isinf(data).any().item()}")


# --- 固定 IDM 参数预测函数 ---
def idm_fixed(v_n, s_safe, delta_v,
              v_desired=10.13701546, T=0.50290469, a_max= 0.10995557,
              b_safe=4.98369406, delta=5.35419582, s0=0.10337701,
              delta_t=0.1):
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
    device = v_n.device
    v_desired = torch.tensor(v_desired, device=device, dtype=v_n.dtype)
    T = torch.tensor(T, device=device, dtype=v_n.dtype)
    a_max = torch.tensor(a_max, device=device, dtype=v_n.dtype).clamp(min=1e-6)
    b_safe = torch.tensor(b_safe, device=device, dtype=v_n.dtype).clamp(min=1e-6)
    s0 = torch.tensor(s0, device=device, dtype=v_n.dtype)
    delta_param = torch.tensor(delta, device=device, dtype=v_n.dtype)
    delta_t_tensor = torch.tensor(delta_t, device=device, dtype=v_n.dtype)

    s_safe = s_safe.clamp(min=1e-6)

    s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe) + 1e-6)
    s_star = s_star.clamp(min=0.0)

    v_n_ratio = torch.zeros_like(v_n)
    mask_v_desired_nonzero = v_desired.abs() > 1e-6
    if mask_v_desired_nonzero.any():
        v_n_ratio[mask_v_desired_nonzero] = (v_n[mask_v_desired_nonzero] / v_desired[mask_v_desired_nonzero])

    acceleration_term = a_max * (
            1 - v_n_ratio ** delta_param - (s_star / s_safe) ** 2
    )
    v_follow = v_n + delta_t_tensor * acceleration_term
    return v_follow.clamp(min=0.0)


# --- 定义新融合模型 ---
# --- Definizione del modello ibrido (Modificata per alpha fisso) ---
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):  # output_dim è l'orizzonte di previsione K
        super(HybridIDMModel, self).__init__()
        self.pred_horizon = output_dim  # Salva l'orizzonte di previsione K
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.pred_horizon)  # L'LSTM produce K previsioni

        # Parametri IDM (valori fissi) - alpha non è più qui
        self.v_desired_idm = 12.64798288
        self.T_idm = 0.50284384
        self.a_max_idm = 0.10033688
        self.b_safe_idm = 4.98937183
        self.delta_idm = 1.0
        self.s0_idm = 0.13082412

    def forward(self, x, s_safe_initial, v_lead_initial):
        """
        Propagazione forward del modello.
        :param x: Sequenza di input, shape=(batch, seq_len, input_dim)
        :param s_safe_initial: Distanza di sicurezza attuale (valore osservato reale, per l'iterazione IDM multi-step), shape=(batch,)
        :param v_lead_initial: Velocità attuale del veicolo che precede (valore osservato reale, per l'iterazione IDM, si assume costante per K passi futuri), shape=(batch,)
        :return:
          y_lstm_multistep: Previsione di velocità a K passi dall'output diretto dell'LSTM, shape=(batch, K)
          y_idm_multistep: Previsione di velocità a K passi dall'IDM iterato, shape=(batch, K)
        """
        device = x.device

        # Previsione LSTM a K passi
        out, _ = self.lstm(x)
        y_lstm_multistep = self.fc(out[:, -1, :])

        # Previsione IDM iterata a K passi
        y_idm_multistep_list = []
        v_ego_current_idm = x[:, -1, 0].clone()
        s_current_idm = s_safe_initial.clone()
        v_lead_constant_idm = v_lead_initial.clone()

        for _ in range(self.pred_horizon):
            delta_v_idm = v_lead_constant_idm - v_ego_current_idm
            v_ego_next_pred_idm = idm_fixed(
                v_ego_current_idm, s_current_idm, delta_v_idm,
                v_desired=self.v_desired_idm, T=self.T_idm, a_max=self.a_max_idm,
                b_safe=self.b_safe_idm, delta=self.delta_idm, s0=self.s0_idm, delta_t=DT
            )
            y_idm_multistep_list.append(v_ego_next_pred_idm.unsqueeze(1))
            s_current_idm = (s_current_idm + (v_lead_constant_idm - v_ego_current_idm) * DT).clamp(min=1e-6)
            v_ego_current_idm = v_ego_next_pred_idm

        y_idm_multistep = torch.cat(y_idm_multistep_list, dim=1)
        # Alpha non è più un output del modello
        return y_lstm_multistep, y_idm_multistep



def initialize_weights(model):
    """ 初始化模型权重 """
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)


# --- 训练函数 (修改版) ---
# --- 训练函数 (修改版_v2) ---
# --- Funzione di addestramento (Modificata per alpha fisso = 0.7) ---
def train_model(model, train_loader, pred_horizon, num_epochs=30, alpha_decay_loss=0.1, lr_lstm=5e-4): # lr_alpha rimosso
    model.train()
    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(pred_horizon, dtype=torch.float32)).to(
        next(model.parameters()).device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * pred_horizon

    lstm_params = [param for name, param in model.named_parameters()] # alpha_raw non esiste più
    optimizer_lstm = optim.Adam(lstm_params, lr=lr_lstm)
    # optimizer_alpha rimosso

    alpha_fixed = 0.7 # Alpha impostato a un valore fisso

    print(f"--- Addestramento con Alpha Fisso ---")
    print(f"LSTM i parametri saranno ottimizzati in base a L_lstm = alpha * L_true + (1-alpha) * L_idm.")
    print(f"Alpha è fissato a: {alpha_fixed}")
    print(f"------------------------------------")

    for epoch in range(num_epochs):
        epoch_loss_lstm_objective = 0.0
        # epoch_loss_alpha_objective rimosso

        for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in train_loader:
            optimizer_lstm.zero_grad()
            # optimizer_alpha.zero_grad() rimosso

            # Propagazione forward - il modello ora restituisce 2 valori
            y_lstm_multistep, y_idm_multistep = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)
            # current_alpha_val rimosso

            # Calcolo delle componenti di perdita individuali
            loss_lstm_vs_true = ((y_lstm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            loss_lstm_vs_idm = ((y_lstm_multistep - y_idm_multistep.detach()).pow(2) * loss_weights.unsqueeze(0)).mean()

            # --- Aggiornamento dei parametri LSTM ---
            # Per aggiornare i parametri LSTM, usiamo la perdita combinata con alpha fisso.
            loss_for_lstm_params = alpha_fixed * loss_lstm_vs_true + \
                                   (1 - alpha_fixed) * loss_lstm_vs_idm

            loss_for_lstm_params.backward()
            optimizer_lstm.step() # Aggiorna i parametri LSTM

            epoch_loss_lstm_objective += loss_for_lstm_params.item()
            # epoch_loss_alpha_objective rimosso

        avg_lstm_loss = epoch_loss_lstm_objective / len(train_loader)
        # avg_alpha_loss rimosso
        print(f"Epoch {epoch + 1}/{num_epochs}  Perdita Obiettivo LSTM: {avg_lstm_loss:.6f}  (α={alpha_fixed})")
    return model


# --- 测试/评估函数 (修改版) ---
# --- Funzione di test/valutazione (Modificata per alpha fisso) ---
def evaluate_model(model, test_loader, pred_horizon, alpha_decay_loss=0.1, dataset_name="", results_dir=""):
    model.eval()
    all_pred_lstm, all_pred_idm, all_true = [], [], []
    # all_alpha_values rimosso

    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(pred_horizon, dtype=torch.float32)).to(
        next(model.parameters()).device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * pred_horizon

    total_mse_lstm_vs_true_weighted = 0
    total_mse_idm_vs_true_weighted = 0

    with torch.no_grad():
        for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in test_loader:
            # Il modello ora restituisce 2 valori
            y_lstm_multistep, y_idm_multistep = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)
            # alpha_tensor rimosso

            all_pred_lstm.append(y_lstm_multistep.cpu())
            all_pred_idm.append(y_idm_multistep.cpu())
            all_true.append(batch_y_multistep.cpu())
            # all_alpha_values.append rimosso

            loss_lstm_vs_true_batch_weighted = (
                        (y_lstm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_lstm_vs_true_weighted += loss_lstm_vs_true_batch_weighted.item() * batch_x.size(0)

            loss_idm_vs_true_batch_weighted = (
                        (y_idm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_idm_vs_true_weighted += loss_idm_vs_true_batch_weighted.item() * batch_x.size(0)

    num_samples = len(test_loader.dataset)
    avg_mse_lstm_vs_true_weighted = total_mse_lstm_vs_true_weighted / num_samples
    avg_mse_idm_vs_true_weighted = total_mse_idm_vs_true_weighted / num_samples

    fixed_alpha_for_metrics = 0.7  # Alpha fisso utilizzato
    # final_alpha_mean rimosso/modificato

    y_pred_lstm_cat = torch.cat(all_pred_lstm)
    y_pred_idm_cat = torch.cat(all_pred_idm)
    y_true_cat = torch.cat(all_true)

    y_final_prediction_cat = y_pred_lstm_cat

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

    print(
        f"\n--- Riepilogo risultati test (utilizzo finale previsione LSTM, metriche con media semplice su tutti i passi) ---")
    print(
        f"  Previsione LSTM vs Reale -- MSE: {mse_val_overall:.4f}, RMSE: {rmse_val_overall:.4f}, MAE: {mae_val_overall:.4f}, MAPE: {mape_p_overall if not np.isnan(mape_p_overall) else 'N/A'}%")
    print(f"  (Riferimento: Previsione IDM vs Reale MSE ponderato: {avg_mse_idm_vs_true_weighted:.4f})")
    print(
        f"  (Riferimento: Previsione LSTM vs Reale MSE ponderato (metrica addestramento): {avg_mse_lstm_vs_true_weighted:.4f})")
    print(f"  Alpha utilizzato (fisso)={fixed_alpha_for_metrics:.4f}")  # Aggiornato per riflettere alpha fisso

    print(f"\n--- Metriche dettagliate per ogni passo (utilizzo finale previsione velocità LSTM) ---")
    for k_step in range(pred_horizon):
        y_pred_lstm_step_k = y_pred_lstm_cat[:, k_step]
        y_pred_idm_step_k = y_pred_idm_cat[:, k_step]
        y_true_step_k = y_true_cat[:, k_step]

        mse_step_lstm = nn.MSELoss()(y_pred_lstm_step_k, y_true_step_k).item()
        rmse_step_lstm = np.sqrt(mse_step_lstm)
        mae_step_lstm = torch.mean(torch.abs(y_pred_lstm_step_k - y_true_step_k)).item()

        abs_error_step = torch.abs(y_pred_lstm_step_k - y_true_step_k)
        abs_true_step = torch.abs(y_true_step_k)
        valid_mape_mask_step = abs_true_step > 1e-6
        mape_step_lstm = float('nan')
        if torch.sum(valid_mape_mask_step) > 0:
            mape_step_lstm = torch.mean(
                abs_error_step[valid_mape_mask_step] / abs_true_step[valid_mape_mask_step]
            ).item() * 100

        mse_step_idm = nn.MSELoss()(y_pred_idm_step_k, y_true_step_k).item()

        print(f"  Passo {k_step + 1}:")
        print(
            f"    Previsione LSTM -- MSE: {mse_step_lstm:.4f}, RMSE: {rmse_step_lstm:.4f}, MAE: {mae_step_lstm:.4f}, MAPE: {mape_step_lstm if not np.isnan(mape_step_lstm) else 'N/A'}%")
        print(f"    IDM (Riferimento) -- MSE: {mse_step_idm:.4f}")

    k_plot = 0
    plt.figure(figsize=(12, 7))
    plt.plot(y_true_cat[:100, k_plot].numpy(), '--o', label=f'Valore Reale (Passo {k_plot + 1})')
    plt.plot(y_pred_lstm_cat[:100, k_plot].numpy(), '-x',
             label=f'Previsione LSTM (Passo {k_plot + 1}) (Utilizzo Finale)')
    plt.plot(y_pred_idm_cat[:100, k_plot].numpy(), '-s', label=f'Previsione IDM (Passo {k_plot + 1}) (Riferimento)')

    y_pred_combined_for_plot_cat = fixed_alpha_for_metrics * y_pred_lstm_cat + (
                1 - fixed_alpha_for_metrics) * y_pred_idm_cat
    plt.plot(y_pred_combined_for_plot_cat[:100, k_plot].numpy(), '-.',
             label=f'Fusione Ipotetica (Passo {k_plot + 1}, α={fixed_alpha_for_metrics:.2f}) (Riferimento Grafico)')

    plt.title(f'Confronto Previsioni Velocità (primi 100 campioni, Passo {k_plot + 1})')
    plt.xlabel("Indice Campione")
    plt.ylabel("Velocità (m/s)")
    plt.legend()
    plt.grid()
    plot_filename = os.path.join(results_dir, f"{dataset_name}_speed_comparison_LSTM_final_fixed_alpha.png")
    plt.savefig(plot_filename)
    print(f"Grafico confronto velocità salvato in {plot_filename}")
    plt.close()

    return mse_val_overall, rmse_val_overall, mae_val_overall, mape_p_overall


# === 修改后的 compute_position_and_spacing_and_save (使用LSTM直接输出) ===
def compute_position_and_spacing_and_save(model,
                                          test_loader,
                                          raw_data_all,
                                          label_data_all,
                                          train_size,
                                          pred_horizon,
                                          dt=0.1, # Default DT, ensure it matches global DT
                                          output_file="predictions_multistep_extended.xlsx",
                                          dataset_name=""):
    model.eval()

    test_start_idx_in_all_data = train_size

    y_lstm_list_mps, y_true_speeds_list_mps = [], []
    initial_ego_pos_ft_collected = []
    initial_lead_pos_ft_collected = []
    initial_ego_speed_ftps_collected = []
    initial_lead_speed_ftps_collected = []
    true_future_ego_pos_ft_collected = []
    true_future_spacing_ft_collected = []

    with torch.no_grad():
        for i, (batch_x_mps, batch_y_multistep_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps) in enumerate(
                test_loader):
            # Model now returns y_lstm_multistep, y_idm_multistep
            # We only need y_lstm_k_mps for the final speed prediction.
            y_lstm_k_mps, _ = model(batch_x_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps)

            y_lstm_list_mps.append(y_lstm_k_mps.cpu())
            y_true_speeds_list_mps.append(batch_y_multistep_mps.cpu())

            batch_start_idx_in_loader = i * test_loader.batch_size
            current_batch_indices_in_all_data = np.arange(
                test_start_idx_in_all_data + batch_start_idx_in_loader,
                test_start_idx_in_all_data + batch_start_idx_in_loader + batch_x_mps.size(0)
            )

            initial_ego_pos_ft_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 4].cpu())
            initial_lead_pos_ft_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 7].cpu())
            initial_ego_speed_ftps_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 0].cpu())
            initial_lead_speed_ftps_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 5].cpu())

            true_future_ego_pos_ft_collected.append(
                label_data_all[current_batch_indices_in_all_data, :pred_horizon, 3].cpu())
            true_future_spacing_ft_collected.append(
                label_data_all[current_batch_indices_in_all_data, :pred_horizon, 1].cpu())

    y_lstm_all_mps = torch.cat(y_lstm_list_mps, dim=0)
    y_true_speeds_all_mps = torch.cat(y_true_speeds_list_mps, dim=0)

    # The final predicted speed is directly the LSTM's output
    final_pred_speeds_mps = y_lstm_all_mps

    initial_ego_pos_ft = torch.cat(initial_ego_pos_ft_collected, dim=0)
    initial_lead_pos_ft = torch.cat(initial_lead_pos_ft_collected, dim=0)
    initial_ego_speed_ftps = torch.cat(initial_ego_speed_ftps_collected, dim=0)
    initial_lead_speed_ftps = torch.cat(initial_lead_speed_ftps_collected, dim=0)

    true_future_ego_pos_ft = torch.cat(true_future_ego_pos_ft_collected, dim=0)
    true_future_spacing_ft = torch.cat(true_future_spacing_ft_collected, dim=0)

    pred_ego_pos_k_steps_ft = torch.zeros_like(final_pred_speeds_mps)
    pred_lead_pos_k_steps_ft = torch.zeros_like(final_pred_speeds_mps)
    pred_spacing_k_steps_ft = torch.zeros_like(final_pred_speeds_mps)

    final_pred_speeds_ftps = final_pred_speeds_mps / 0.3048 # Convert to ft/s for calculation

    current_ego_pos_ft = initial_ego_pos_ft.clone()
    current_lead_pos_ft = initial_lead_pos_ft.clone()
    # Assume lead vehicle speed is constant over the prediction horizon for this calculation
    lead_speed_constant_ftps = initial_lead_speed_ftps # This is (batch_size, )

    # Make lead_speed_constant_ftps (batch_size,) compatible for broadcasting or explicit step use
    # The loop implies lead speed is constant for each sample across all k steps.

    for k in range(pred_horizon):
        # Speed of ego vehicle for the current step k
        # For k=0, use initial observed speed. For k > 0, use predicted speed from step k-1.
        if k == 0:
            # For the first step's displacement, use the *initial* ego speed.
            speed_ego_this_step_ftps = initial_ego_speed_ftps
        else:
            # For subsequent steps, use the LSTM's predicted speed for the *previous* interval's end.
            speed_ego_this_step_ftps = final_pred_speeds_ftps[:, k - 1]

        disp_ego_ft = speed_ego_this_step_ftps * dt
        # Lead vehicle's speed is assumed constant from its initial observation for all k steps.
        disp_lead_ft = lead_speed_constant_ftps * dt # lead_speed_constant_ftps is (batch_size,)

        current_ego_pos_ft += disp_ego_ft
        current_lead_pos_ft += disp_lead_ft

        pred_ego_pos_k_steps_ft[:, k] = current_ego_pos_ft
        pred_lead_pos_k_steps_ft[:, k] = current_lead_pos_ft
        pred_spacing_k_steps_ft[:, k] = current_lead_pos_ft - current_ego_pos_ft

    pred_ego_pos_m = pred_ego_pos_k_steps_ft.numpy() * 0.3048
    true_ego_pos_m = true_future_ego_pos_ft.numpy() * 0.3048
    pred_spacing_m = pred_spacing_k_steps_ft.numpy() * 0.3048
    true_spacing_m = true_future_spacing_ft.numpy() * 0.3048

    print(f"\n--- Position and Spacing Error Evaluation per step (based on LSTM direct speed prediction) ---")
    for k_s in range(pred_horizon):
        pos_err_sq_step = (pred_ego_pos_m[:, k_s] - true_ego_pos_m[:, k_s]) ** 2
        rmse_Y_step = np.sqrt(np.mean(pos_err_sq_step))
        valid_true_Y_step_mask = np.abs(true_ego_pos_m[:, k_s]) > 1e-6
        mape_Y_step = float('nan')
        if np.sum(valid_true_Y_step_mask) > 0:
            mape_Y_step = np.mean(np.abs(
                (pred_ego_pos_m[valid_true_Y_step_mask, k_s] - true_ego_pos_m[valid_true_Y_step_mask, k_s]) /
                true_ego_pos_m[valid_true_Y_step_mask, k_s])) * 100

        spacing_err_sq_step = (pred_spacing_m[:, k_s] - true_spacing_m[:, k_s]) ** 2
        rmse_sp_step = np.sqrt(np.mean(spacing_err_sq_step))
        valid_true_sp_step_mask = np.abs(true_spacing_m[:, k_s]) > 1e-6
        mape_sp_step = float('nan')
        if np.sum(valid_true_sp_step_mask) > 0:
            mape_sp_step = np.mean(np.abs(
                (pred_spacing_m[valid_true_sp_step_mask, k_s] - true_spacing_m[valid_true_sp_step_mask, k_s]) /
                true_spacing_m[valid_true_sp_step_mask, k_s])) * 100

        print(f"  Step {k_s + 1}:")
        print(
            f"    Position Error -- RMSE: {rmse_Y_step:.4f} m, MAPE: {mape_Y_step if not np.isnan(mape_Y_step) else 'N/A'}%")
        print(
            f"    Spacing Error -- RMSE: {rmse_sp_step:.4f} m, MAPE: {mape_sp_step if not np.isnan(mape_sp_step) else 'N/A'}%")

    # k_eval = pred_horizon - 1 # Evaluate at the last prediction step
    # rmse_p_last_step = np.sqrt(np.mean((pred_ego_pos_m[:, k_eval] - true_ego_pos_m[:, k_eval]) ** 2))
    # valid_true_Y_last_mask = np.abs(true_ego_pos_m[:, k_eval]) > 1e-6
    # mape_p_last_step = float('nan')
    # if np.sum(valid_true_Y_last_mask) > 0:
    #     mape_p_last_step = np.mean(np.abs(
    #         (pred_ego_pos_m[valid_true_Y_last_mask, k_eval] - true_ego_pos_m[valid_true_Y_last_mask, k_eval]) /
    #         true_ego_pos_m[valid_true_Y_last_mask, k_eval])) * 100
    # Calculate RMSE and MAPE over the entire prediction horizon
    rmse_p_overall = np.sqrt(np.mean((pred_ego_pos_m - true_ego_pos_m) ** 2))  # RMSE over all prediction steps
    valid_true_Y_mask = np.abs(true_ego_pos_m) > 1e-6  # Mask for valid true values

    mape_p_overall = float('nan')
    if np.sum(valid_true_Y_mask) > 0:
        mape_p_overall = np.mean(np.abs(
            (pred_ego_pos_m[valid_true_Y_mask] - true_ego_pos_m[valid_true_Y_mask]) /
            true_ego_pos_m[valid_true_Y_mask])) * 100  # MAPE over all prediction steps

    print(f"\n--- Final Step (K={rmse_p_overall}) Position Error (based on LSTM direct prediction, for summary) ---")
    print(
        f"  Position Error -- RMSE: {rmse_p_overall:.4f} m, MAPE: {mape_p_overall if not np.isnan(mape_p_overall) else 'N/A'}%")

    df_data = {}
    for k_idx in range(pred_horizon):
        df_data[f"Predicted Speed LSTM (m/s) Step {k_idx + 1}"] = final_pred_speeds_mps[:, k_idx].numpy()
        df_data[f"True Speed (m/s) Step {k_idx + 1}"] = y_true_speeds_all_mps[:, k_idx].numpy()
        df_data[f"Predicted Ego Position Y (m) Step {k_idx + 1}"] = pred_ego_pos_m[:, k_idx]
        df_data[f"True Ego Position Y (m) Step {k_idx + 1}"] = true_ego_pos_m[:, k_idx]
        df_data[f"Predicted Spacing (m) Step {k_idx + 1}"] = pred_spacing_m[:, k_idx]
        df_data[f"True Spacing (m) Step {k_idx + 1}"] = true_spacing_m[:, k_idx]

    df_pos = pd.DataFrame(df_data)
    # Ensure results directory exists if not handled by main script
    # os.makedirs(os.path.dirname(output_file), exist_ok=True) # If output_file includes a path

    # Append to existing excel file or create new, replacing sheet if it exists
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            df_pos.to_excel(writer, sheet_name=dataset_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df_pos.to_excel(writer, sheet_name=dataset_name, index=False)

    print(f"Position and spacing predictions for {dataset_name} (based on LSTM) saved to '{output_file}' sheet '{dataset_name}'.")
    return rmse_p_overall, mape_p_overall


# --- 存储和保存评估指标的辅助函数 ---
all_datasets_metrics_summary = []


def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse_last_step,
                          pos_mape_last_step):
    metrics = {
        "数据集 (Dataset)": dataset_name,
        "速度MSE_LSTM (Speed_MSE_LSTM_summary)": speed_mse,  # 基于LSTM的加权MSE
        "速度RMSE_LSTM (Speed_RMSE_LSTM_summary)": speed_rmse,  # 基于LSTM的加权RMSE
        "速度MAE_LSTM (Speed_MAE_LSTM_overall)": speed_mae,  # 基于LSTM的整体MAE
        "速度MAPE_LSTM (%) (Speed_MAPE_LSTM_overall_percent)": speed_mape,  # 基于LSTM的整体MAPE
        "末步位置RMSE_LSTM (m) (Position_RMSE_LSTM_last_step_m)": pos_rmse_last_step,
        "末步位置MAPE_LSTM (%) (Position_MAPE_LSTM_last_step_percent)": pos_mape_last_step
    }
    all_datasets_metrics_summary.append(metrics)


def save_all_metrics_to_csv(filepath="evaluation_summary_LSTM_final.csv"):  # 文件名修改
    if not all_datasets_metrics_summary:
        print("没有评估指标可以保存。")
        return
    df_metrics = pd.DataFrame(all_datasets_metrics_summary)
    df_metrics.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"所有数据集的评估指标汇总已保存至 {filepath}")


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
                                                   "pred_positions_all_datasets_LSTM_final_all_steps1128.xlsx")  # 文件名修改

    # 定义学习率
    LR_LSTM_PARAMS = 5e-4  # 原始学习率
    LR_ALPHA_PARAM = 1e-3  # Alpha的学习率，可能需要调整

    for data_file_path in data_files:
        dataset_filename = os.path.basename(data_file_path)
        dataset_name_clean = dataset_filename.replace(".mat", "")
        print(f"\n==================== 开始处理数据集: {dataset_filename} ====================")

        data = sio.loadmat(data_file_path)
        raw_all_ft = torch.tensor(data['train_data'], dtype=torch.float32)
        lab_all_ft = torch.tensor(data['lable_data'], dtype=torch.float32)

        seq_ft = raw_all_ft[:, :, [0, 1, 2, 3, 5]].clone()
        y_multistep_ftps = lab_all_ft[:, :, 0].clone()
        current_pred_horizon = y_multistep_ftps.shape[1]
        print(f"数据集 {dataset_filename} 的预测步长 K = {current_pred_horizon}")

        s_safe_initial_ft = seq_ft[:, -1, 1].clone()
        v_lead_initial_ftps = seq_ft[:, -1, 4].clone()

        seq_mps = seq_ft.clone()
        seq_mps[:, :, [0, 2, 3, 4]] *= 0.3048
        seq_mps[:, :, 1] *= 0.3048
        y_multistep_mps = y_multistep_ftps * 0.3048
        s_safe_initial_m = s_safe_initial_ft * 0.3048
        v_lead_initial_mps = v_lead_initial_ftps * 0.3048

        N_total = seq_mps.size(0)
        N = int(N_total * 0.2)  # 使用100%数据，之前是0.1
        print(f"将使用 {N} / {N_total} 条数据进行训练和测试。")

        seq_mps_selected = seq_mps[:N]
        y_multistep_mps_selected = y_multistep_mps[:N]
        s_safe_initial_m_selected = s_safe_initial_m[:N]
        v_lead_initial_mps_selected = v_lead_initial_mps[:N]
        raw_all_ft_selected = raw_all_ft[:N]
        lab_all_ft_selected = lab_all_ft[:N]

        split_ratio = 0.8
        train_size = int(N * split_ratio)

        train_seq = seq_mps_selected[:train_size].to(device)
        test_seq = seq_mps_selected[train_size:].to(device)
        train_y_multistep = y_multistep_mps_selected[:train_size].to(device)
        test_y_multistep = y_multistep_mps_selected[train_size:].to(device)
        train_s_safe_initial = s_safe_initial_m_selected[:train_size].to(device)
        test_s_safe_initial = s_safe_initial_m_selected[train_size:].to(device)
        train_v_lead_initial = v_lead_initial_mps_selected[:train_size].to(device)
        test_v_lead_initial = v_lead_initial_mps_selected[train_size:].to(device)

        train_ds = torch.utils.data.TensorDataset(train_seq, train_y_multistep, train_s_safe_initial,
                                                  train_v_lead_initial)
        test_ds = torch.utils.data.TensorDataset(test_seq, test_y_multistep, test_s_safe_initial, test_v_lead_initial)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

        input_dim = train_seq.size(2)
        hidden_dim = 128
        model = HybridIDMModel(input_dim, hidden_dim, output_dim=current_pred_horizon, num_layers=1).to(device)
        initialize_weights(model)
        # Optimizers are now created inside train_model

        print(f"开始训练模型: {dataset_name_clean}...")
        model = train_model(model, train_loader, pred_horizon=current_pred_horizon, num_epochs=50,  # 之前是50
                            alpha_decay_loss=0.05, lr_lstm=LR_LSTM_PARAMS)

        print(f"开始评估模型 (速度预测, 最终使用LSTM输出): {dataset_name_clean}...")
        speed_mse_summary, speed_rmse_summary, speed_mae_overall, speed_mape_overall = evaluate_model(
            model, test_loader, pred_horizon=current_pred_horizon, alpha_decay_loss=0.05,
            dataset_name=dataset_name_clean, results_dir=RESULTS_DIR
        )

        print(f"开始计算和评估位置/间距预测 (基于LSTM直接输出): {dataset_name_clean}...")
        pos_rmse_last_step, pos_mape_last_step = compute_position_and_spacing_and_save(
            model, test_loader, raw_all_ft_selected, lab_all_ft_selected, train_size,
            pred_horizon=current_pred_horizon, dt=DT,
            output_file=position_predictions_excel_path, dataset_name=dataset_name_clean
        )

        store_dataset_metrics(dataset_name_clean, speed_mse_summary, speed_rmse_summary, speed_mae_overall,
                              speed_mape_overall, pos_rmse_last_step, pos_mape_last_step)
        print(f"==================== 数据集 {dataset_filename} 处理完毕 ====================")

    summary_metrics_csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_all_datasets_LSTM_final1128.csv")  # 文件名修改
    save_all_metrics_to_csv(summary_metrics_csv_path)
    print("\n所有数据集处理完毕。最终评估汇总已保存。")

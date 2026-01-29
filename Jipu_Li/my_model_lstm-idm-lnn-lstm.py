#
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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# ###########################
# #  数据检查与初始化函数
# ###########################
# def check_data(data, name="data"):
#     print(f"Checking {name} for NaN or Inf values...")
#     print(f"Has NaN: {torch.isnan(data).any().item()}")
#     print(f"Has Inf: {torch.isinf(data).any().item()}")
#
# def initialize_weights(model):
#     for name, param in model.named_parameters():
#         if "weight" in name:
#             nn.init.xavier_uniform_(param)
#         elif "bias" in name:
#             nn.init.constant_(param, 0)
#
# ###########################
# # 1. LSTM-IDM模型定义
# ###########################
# class HybridIDMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers=2):
#         """
#         :param input_dim: 输入特征数（例如选取5个特征：自车速度、前车距离、速度差、加速度、前车速度）
#         :param hidden_dim: LSTM隐藏层单元数
#         :param num_layers: LSTM层数
#         """
#         super(HybridIDMModel, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, 6)  # IDM参数：[v_desired, T, a_max, b_safe, delta, s0]
#         self.softplus = nn.Softplus()
#         self.delta_t = 0.1  # 时间步长
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         params = self.fc(out[:, -1, :])
#         params = self.softplus(params)  # 确保参数为正
#         return params
#
#     def predict_speed(self, x, s_safe):
#         # x: (batch, seq_len, input_dim)
#         params = self.forward(x)
#         # 取当前时刻自车速度与速度差
#         v_n = x[:, -1, 0]
#         delta_v = x[:, -1, 2]
#         v_desired, T, a_max, b_safe, delta, s0 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]
#         # 计算安全距离与预测跟驰速度
#         s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe) + 1e-6)
#         s_star = torch.clamp(s_star, min=0)
#         v_follow = v_n + self.delta_t * a_max * (1 - (v_n / v_desired) ** delta - (s_star / s_safe) ** 2)
#         predicted_speed = torch.clamp(v_follow, min=0)
#         # 调整为二维张量，形状：(batch, 1)
#         return predicted_speed.unsqueeze(1), params
#
# ###########################
# # 2. 液态神经网络（LNN）模型定义
# ###########################
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
# class LiquidNeuralNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers=1, num_steps=50, output_dim=1):
#         super(LiquidNeuralNetwork, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.num_steps = num_steps
#
#         # 构建多个液态单元
#         self.liquid_cells = nn.ModuleList([
#             LiquidCell(input_dim if i == 0 else hidden_dim, hidden_dim)
#             for i in range(num_layers)
#         ])
#
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
#         T = min(seq_len, self.num_steps)
#         # 初始化每层隐藏状态
#         h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
#         for t in range(T):
#             input_t = x[:, t, :]
#             for i, cell in enumerate(self.liquid_cells):
#                 if i == 0:
#                     h[i] = cell(input_t, h[i])
#                 else:
#                     h[i] = cell(h[i-1], h[i])
#         out = self.fc(h[-1])
#         return out
#
#     def predict_speed(self, x):
#         return self.forward(x)
#
# ###########################
# # 3. 融合模块（基于LSTM）的定义
# ###########################
# class FusionModule(nn.Module):
#     def __init__(self, input_dim=2, hidden_dim=32, num_layers=1):
#         """
#         融合模块输入为 shape: (batch, seq_len, 2)，其中2个特征分别为原始数据的第一列与最后一列，
#         序列长度为5（历史5步到当前步）。
#         输出经全连接层及sigmoid映射得到门控值 lambda，形状为 (batch, 1)。
#         """
#         super(FusionModule, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         # 取最后一时刻的隐藏状态
#         h_last = out[:, -1, :]
#         lambda_val = self.sigmoid(self.fc(h_last))
#         return lambda_val
#
# ###########################
# # 模型训练函数定义
# ###########################
# def train_idm_model(model, train_loader, optimizer, criterion, num_epochs=30):
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch_data, batch_speed, batch_s_safe in train_loader:
#             optimizer.zero_grad()
#             predicted_speed, _ = model.predict_speed(batch_data.to(device), batch_s_safe.to(device))
#             loss = criterion(predicted_speed, batch_speed.to(device).unsqueeze(1))
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"[LSTM-IDM] Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
#     return model
#
# def train_lnn_model(model, train_loader, optimizer, criterion, num_epochs=30):
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch_data, batch_speed in train_loader:
#             optimizer.zero_grad()
#             predicted_speed = model.predict_speed(batch_data.to(device))
#             loss = criterion(predicted_speed, batch_speed.to(device))
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"[LNN] Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
#     return model
#
# def train_fusion_module(fusion_module, idm_model, lnn_model, fusion_loader, optimizer, criterion, num_epochs=20):
#     # 固定 IDM 和 LNN 模型参数
#     idm_model.eval()
#     lnn_model.eval()
#     fusion_module.train()
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch in fusion_loader:
#             # 每个 batch 包含:
#             # fusion_input: 融合模块输入，形状 (batch, 5, 2)
#             # idm_input: LSTM-IDM输入
#             # lnn_input: LNN输入
#             # ground_truth: 真实车速（标量）
#             # s_safe: IDM所需安全距离参数
#             fusion_input, idm_input, lnn_input, ground_truth, s_safe = batch
#             fusion_input = fusion_input.to(device)
#             idm_input = idm_input.to(device)
#             lnn_input = lnn_input.to(device)
#             ground_truth = ground_truth.to(device).unsqueeze(1)  # 调整为 (batch, 1)
#             s_safe = s_safe.to(device)
#
#             with torch.no_grad():
#                 # 获得两个模型的预测输出，保证输出形状为 (batch, 1)
#                 y_ph, _ = idm_model.predict_speed(idm_input, s_safe)
#                 y_da = lnn_model.predict_speed(lnn_input)
#
#             # 融合模块输出门控值 lambda (shape: (batch, 1))
#             lambda_val = fusion_module(fusion_input)
#             # 融合预测 (逐元素相乘，保证形状一致)
#             fused_output = lambda_val * y_da + (1 - lambda_val) * y_ph
#
#             loss = criterion(fused_output, ground_truth)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"[Fusion] Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(fusion_loader):.4f}")
#     return fusion_module
#
# ###########################
# # 评估函数（适用于单模型或融合模型）
# ###########################
# def evaluate_model(model, test_loader, model_type="IDM"):
#     model.eval()
#     mse_loss = nn.MSELoss()
#     total_mse = 0
#     all_predicted = []
#     all_true = []
#     with torch.no_grad():
#         for batch in test_loader:
#             if model_type == "IDM":
#                 batch_data, batch_speed, batch_s_safe = batch
#                 predicted_speed, _ = model.predict_speed(batch_data.to(device), batch_s_safe.to(device))
#                 gt = batch_speed.to(device).unsqueeze(1)
#                 # predicted_speed, params = model.predict_speed(batch_data, batch_s_safe)
#                 # gt = batch_speed.to(device).unsqueeze(1)
#             elif model_type == "LNN":
#                 batch_data, batch_speed = batch
#                 predicted_speed = model.predict_speed(batch_data.to(device))
#                 gt = batch_speed.to(device)
#             elif model_type == "Fusion":
#                 # 对于融合模型，batch 中包含：fusion_input, idm_input, lnn_input, ground_truth, s_safe
#                 fusion_input, idm_input, lnn_input, gt, s_safe = batch
#                 fusion_input = fusion_input.to(device)
#                 idm_input = idm_input.to(device)
#                 lnn_input = lnn_input.to(device)
#                 gt = gt.to(device).unsqueeze(1)
#                 s_safe = s_safe.to(device)
#                 lambda_val = model(fusion_input)
#                 with torch.no_grad():
#                     y_ph, _ = idm_model.predict_speed(idm_input, s_safe)
#                     y_da = lnn_model.predict_speed(lnn_input)
#                 predicted_speed = lambda_val * y_da + (1 - lambda_val) * y_ph
#             loss = mse_loss(predicted_speed, gt)
#             total_mse += loss.item()
#             all_predicted.append(predicted_speed.cpu())
#             all_true.append(gt.cpu())
#     mse = total_mse / len(test_loader)
#     rmse = torch.sqrt(torch.tensor(mse))
#     mae = torch.mean(torch.abs(torch.cat(all_predicted) - torch.cat(all_true))).item()
#     print(f"Evaluation Metrics:\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
#     # 绘制前100个样本对比图
#     plt.figure(figsize=(10, 6))
#     plt.plot(torch.cat(all_true)[:300].numpy(), label='True Speed', linestyle='--', marker='o')
#     plt.plot(torch.cat(all_predicted)[:300].numpy(), label='Predicted Speed', linestyle='-', marker='x')
#     plt.xlabel('Sample Index')
#     plt.ylabel('Speed (m/s)')
#     plt.legend()
#     plt.grid()
#     # plt.show()
#
# ###########################
# # 主函数
# import os
# import torch
# import numpy as np
# import pandas as pd
#
# def compute_position_and_spacing_and_save(
#         fusion_module,
#         idm_model,
#         lnn_model,
#         fusion_input,    # Tensor: (N_test, seq_len_fusion, 2)
#         idm_input,       # Tensor: (N_test, seq_len_idm, 5)
#         lnn_input,       # Tensor: (N_test, seq_len_lnn, 5)
#         s_safe,          # Tensor: (N_test,)
#         raw_data,        # Tensor: full raw_data from mat
#         label_data,      # Tensor: full label_data from mat
#         train_size,      # int, 划分训练/测试时的训练集大小
#         dt=0.1,
#         output_file="fusion_predictions.xlsx"
#     ):
#     fusion_module.eval()
#     idm_model.eval()
#     lnn_model.eval()
#
#     with torch.no_grad():
#         # 1. IDM 预测
#         y_ph, _ = idm_model.predict_speed(idm_input.to(fusion_input.device), s_safe.to(fusion_input.device))
#         # 2. LNN 预测
#         y_da = lnn_model.predict_speed(lnn_input.to(fusion_input.device))
#         # 3. Fusion 预测
#         lambda_val = fusion_module(fusion_input.to(fusion_input.device))
#         pred_speed = (lambda_val * y_da + (1 - lambda_val) * y_ph).squeeze().cpu().numpy()
#
#     N_test = fusion_input.shape[0]
#     idx = np.arange(train_size, train_size + N_test)
#
#     # 当前时刻的 Y（ft） 和 当前速度（m/s）
#     current_Y_ft   = raw_data[idx, -1, 4].numpy()           # 第五列为 Y(ft)
#     current_speed_m = fusion_input[:, -1, 0].numpy()        # 融合输入第一列即车速(m/s)
#
#     # 真值 Y(ft) 和 间距(m)
#     true_Y_ft      = label_data[idx, -1, 3].numpy()         # 第四列为真实 Y(ft)
#     true_spacing_m = label_data[idx, -1, 1].numpy() * 0.3048  # 第二列为间距(ft)，先转 m
#
#     # 运动学积分计算位移（m→ft）
#     # 0.5 * a_avg * dt^2 + v0 * dt,  a_avg = (pred - v0) / dt
#     disp_m    = current_speed_m * dt + 0.5 * ((pred_speed - current_speed_m) / dt) * dt**2
#     disp_ft   = disp_m / 0.3048
#     pred_Y_ft = current_Y_ft + disp_ft
#
#     # 转回 m
#     pred_Y_m      = pred_Y_ft * 0.3048
#     true_Y_m      = true_Y_ft * 0.3048
#     pred_spacing_m = true_Y_m - pred_Y_m + true_spacing_m
#
#     # 计算指标
#     rmse_Y  = np.sqrt(np.mean((pred_Y_m - true_Y_m)**2))
#     mape_Y  = np.mean(np.abs((pred_Y_m - true_Y_m)/true_Y_m)) * 100
#     rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m)**2))
#     mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m)/true_spacing_m)) * 100
#
#     print(f"[Fusion] Position Error -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
#     print(f"[Fusion] Spacing  Error -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")
#
#     # 保存到 Excel
#     df = pd.DataFrame({
#         "Pred Speed (m/s)":      pred_speed,
#         "Predicted Y (m)":       pred_Y_m,
#         "True Y (m)":            true_Y_m,
#         "Predicted Spacing (m)": pred_spacing_m,
#         "True Spacing (m)":      true_spacing_m,
#     })
#     sheet_name = "Fusion"
#
#     with pd.ExcelWriter(output_file,
#                         engine="openpyxl",
#                         mode="a" if os.path.exists(output_file) else "w") as writer:
#         df.to_excel(writer, sheet_name=sheet_name, index=False)
#         print(f"Fusion results saved to '{output_file}' sheet '{sheet_name}'.")
#
#
# ###########################
# if __name__ == "__main__":
#     torch.manual_seed(42)
#
#     ##############################################
#     # 数据加载与预处理（统一使用同一数据文件）
#     ##############################################
#     # 假设数据保存在 'data_fine_0.1.mat' 中
#     data = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')
#     # 原始数据，假设变量名为 'train_data'
#     raw_data = torch.tensor(data['train_data'], dtype=torch.float32)  # shape: (样本数, 时间步长, features)
#     # 标签数据，假设 'lable_data' 中第1列为车速
#     lable_data = torch.tensor(data['lable_data'], dtype=torch.float32)
#     # 对于IDM模型，使用最后50个时刻，选取5个特征：第一列、第二列、第三列、第四列、最后一列
#     data_idm = raw_data[:, -50:, [0, 1, 2, 3, -1]].clone()
#     # 对于LNN模型，同样取最后50个时刻5个特征
#     data_lnn = raw_data[:, -50:, [0, 1, 2, 3, -1]].clone()
#     # 对于融合模块：输入为原始数据中“第一列”和“最后一列”，选取历史5个步长（包含当前步）
#     fusion_data = raw_data[:, -5:, :].clone()  # 从中抽取需要的列
#     fusion_data = fusion_data[:, :, [0, -1]]  # shape: (samples, 5, 2) #存在错误
#
#     # 单位转换（例如ft->m, ft/s->m/s），假设所有需要转换的列均乘以0.3048
#     for tensor in [data_idm, data_lnn, fusion_data]:
#         tensor *= 0.3048
#     # 标签：取最后时刻的车速（代码中采用 lable_data[:, -1, 0]）
#     ground_truth = lable_data[:, -1, 0].clone() * 0.3048
#
#     # 对于IDM模型，需要额外的 s_safe 参数：取 data_idm 中第2列（前车距离）的最后时刻
#     s_safe = data_idm[:, -1, 1].clone()
#
#     # 选取部分数据（例如前10%）用于快速实验
#     total_samples = data_idm.shape[0]
#     sample_size = int(total_samples * 0.1)
#     data_idm = data_idm[:sample_size]
#     data_lnn = data_lnn[:sample_size]
#     fusion_data = fusion_data[:sample_size]
#     ground_truth = ground_truth[:sample_size]
#     s_safe = s_safe[:sample_size]
#
#     # 检查数据
#     check_data(data_idm, "data_idm")
#     check_data(ground_truth, "ground_truth")
#     check_data(s_safe, "s_safe")
#     check_data(fusion_data, "fusion_data")
#
#     # 划分训练集和测试集（80%训练，20%测试）
#     dataset_size = data_idm.shape[0]
#     train_size = int(dataset_size * 0.8)
#
#     # IDM数据集（输入, ground truth, s_safe）
#     train_idm_data = data_idm[:train_size]
#     test_idm_data = data_idm[train_size:]
#     train_idm_gt = ground_truth[:train_size]
#     test_idm_gt = ground_truth[train_size:]
#     train_s_safe = s_safe[:train_size]
#     test_s_safe = s_safe[train_size:]
#
#     # LNN数据集（输入, ground truth）
#     train_lnn_data = data_lnn[:train_size]
#     test_lnn_data = data_lnn[train_size:]
#     train_lnn_gt = ground_truth[:train_size].unsqueeze(1)  # 保持shape一致
#     test_lnn_gt = ground_truth[train_size:].unsqueeze(1)
#
#     # 融合数据集：需同时提供融合模块输入、以及对应的IDM和LNN输入数据与ground truth
#     train_fusion_input = fusion_data[:train_size]
#     test_fusion_input = fusion_data[train_size:]
#     # 对于idm和lnn输入，直接使用各自的原始输入数据
#     train_fusion_idm = train_idm_data
#     test_fusion_idm = test_idm_data
#     train_fusion_lnn = train_lnn_data
#     test_fusion_lnn = test_lnn_data
#     train_fusion_gt = ground_truth[:train_size]
#     test_fusion_gt = ground_truth[train_size:]
#     train_fusion_s_safe = train_s_safe
#     test_fusion_s_safe = test_s_safe
#
#     #构造前车距离、自车位置的计算函数输入
#     test_fusion_input = fusion_data[train_size:]  # (N_test, 5, 2), 已乘 0.3048
#     test_idm_input = data_idm[train_size:]  # (N_test, 50, 5)
#     test_lnn_input = data_lnn[train_size:]  # (N_test, 50, 5)
#     test_s_safe = s_safe[train_size:]  # (N_test,)
#
#     # 创建数据加载器
#     batch_size = 32
#     idm_train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(train_idm_data, train_idm_gt, train_s_safe),
#         batch_size=batch_size, shuffle=True
#     )
#     idm_test_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(test_idm_data, test_idm_gt, test_s_safe),
#         batch_size=batch_size, shuffle=False
#     )
#     lnn_train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(train_lnn_data, train_lnn_gt),
#         batch_size=batch_size, shuffle=True
#     )
#     lnn_test_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(test_lnn_data, test_lnn_gt),
#         batch_size=batch_size, shuffle=False
#     )
#     # 融合模块加载器：每个 batch 返回 (fusion_input, idm_input, lnn_input, ground_truth, s_safe)
#     fusion_train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(train_fusion_input, train_fusion_idm, train_fusion_lnn, train_fusion_gt, train_fusion_s_safe),
#         batch_size=batch_size, shuffle=True
#     )
#     fusion_test_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(test_fusion_input, test_fusion_idm, test_fusion_lnn, test_fusion_gt, test_fusion_s_safe),
#         batch_size=batch_size, shuffle=False
#     )
#
#     ##############################################
#     # 训练 LSTM-IDM 模型
#     ##############################################
#     input_dim_idm = train_idm_data.shape[2]  # 5
#     hidden_dim_idm = 128
#     num_layers_idm = 1
#     idm_model = HybridIDMModel(input_dim_idm, hidden_dim_idm, num_layers=num_layers_idm).to(device)
#     initialize_weights(idm_model)
#     criterion_idm = nn.MSELoss()
#     optimizer_idm = optim.Adam(idm_model.parameters(), lr=0.0005)
#     num_epochs_idm = 100
#     print("\n--- Training LSTM-IDM Model ---")
#     idm_model = train_idm_model(idm_model, idm_train_loader, optimizer_idm, criterion_idm, num_epochs=num_epochs_idm)
#     print("\n--- Evaluating LSTM-IDM Model on Test Set ---")
#     evaluate_model(idm_model, idm_test_loader, model_type="IDM")
#
#     ##############################################
#     # 训练 LNN 模型
#     ##############################################
#     input_dim_lnn = train_lnn_data.shape[2]  # 同样为5
#     hidden_dim_lnn = 128
#     num_layers_lnn = 1
#     num_steps = train_lnn_data.shape[1]  # 50
#     lnn_model = LiquidNeuralNetwork(input_dim_lnn, hidden_dim_lnn, num_layers=num_layers_lnn, num_steps=num_steps, output_dim=1).to(device)
#     initialize_weights(lnn_model)
#     criterion_lnn = nn.MSELoss()
#     optimizer_lnn = optim.Adam(lnn_model.parameters(), lr=0.0005)
#     num_epochs_lnn = 150
#     print("\n--- Training Liquid Neural Network (LNN) Model ---")
#     lnn_model = train_lnn_model(lnn_model, lnn_train_loader, optimizer_lnn, criterion_lnn, num_epochs=num_epochs_lnn)
#     print("\n--- Evaluating LNN Model on Test Set ---")
#     evaluate_model(lnn_model, lnn_test_loader, model_type="LNN")
#
#     ##############################################
#     # 训练 融合模块
#     ##############################################
#     # 在训练融合模块时，idm_model和 lnn_model 参数固定，不更新
#     fusion_module = FusionModule(input_dim=2, hidden_dim=32, num_layers=1).to(device)
#     initialize_weights(fusion_module)
#     criterion_fusion = nn.MSELoss()
#     optimizer_fusion = optim.Adam(fusion_module.parameters(), lr=0.001)
#     num_epochs_fusion = 20
#     print("\n--- Training Fusion Module ---")
#     fusion_module = train_fusion_module(fusion_module, idm_model, lnn_model, fusion_train_loader, optimizer_fusion, criterion_fusion, num_epochs=num_epochs_fusion)
#     print("\n--- Evaluating Fusion Model on Test Set ---")
#     # 评估时调用融合评估函数，此处需要全局 idm_model 和 lnn_model
#     # 注意：评估函数内部调用了 idm_model 和 lnn_model，因此需要保证这两个模型在全局可见
#     evaluate_model(fusion_module, fusion_test_loader, model_type="Fusion")
#     # 直接调用
#     compute_position_and_spacing_and_save(
#         fusion_module=fusion_module,
#         idm_model=idm_model,
#         lnn_model=lnn_model,
#         fusion_input=test_fusion_input,
#         idm_input=test_idm_input,
#         lnn_input=test_lnn_input,
#         s_safe=test_s_safe,
#         raw_data=raw_data,
#         label_data=lable_data,
#         train_size=train_size,
#         dt=0.1,
#         output_file="mymodel_1.xlsx"
#     )



import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################
#  数据检查与初始化函数
###########################
def check_data(data, name="data"):
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")

def initialize_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)

###########################
# 1. LSTM-IDM模型定义
###########################
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        """
        :param input_dim: 输入特征数（例如选取5个特征：自车速度、前车距离、速度差、加速度、前车速度）
        :param hidden_dim: LSTM隐藏层单元数
        :param num_layers: LSTM层数
        """
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
        # x: (batch, seq_len, input_dim)
        params = self.forward(x)
        # 取当前时刻自车速度与速度差
        v_n = x[:, -1, 0]
        delta_v = x[:, -1, 2]
        v_desired, T, a_max, b_safe, delta, s0 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]
        # 计算安全距离与预测跟驰速度
        s_star = s0 + v_n * T + (v_n * -delta_v) / (2 * torch.sqrt(a_max * b_safe) + 1e-6)  #由于把速度差计算反了，是自车速度减前车速度。原始数据中前-跟车
        s_star = torch.clamp(s_star, min=0)
        v_follow = v_n + self.delta_t * a_max * (1 - (v_n / v_desired) ** delta - (s_star / s_safe) ** 2)
        predicted_speed = torch.clamp(v_follow, min=0)
        # 调整为二维张量，形状：(batch, 1)
        return predicted_speed.unsqueeze(1), params

###########################
# 2. 液态神经网络（LNN）模型定义
###########################
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

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_steps=50, output_dim=1):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_steps

        # 构建多个液态单元
        self.liquid_cells = nn.ModuleList([
            LiquidCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        T = min(seq_len, self.num_steps)
        # 初始化每层隐藏状态
        h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        for t in range(T):
            input_t = x[:, t, :]
            for i, cell in enumerate(self.liquid_cells):
                if i == 0:
                    h[i] = cell(input_t, h[i])
                else:
                    h[i] = cell(h[i-1], h[i])
        out = self.fc(h[-1])
        return out

    def predict_speed(self, x):
        return self.forward(x)

###########################
# 3. 融合模块（基于LSTM）的定义
###########################
class FusionModule(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, num_layers=1):
        """
        融合模块输入为 shape: (batch, seq_len, 2)，其中2个特征分别为原始数据的第一列与最后一列，
        序列长度为5（历史5步到当前步）。
        输出经全连接层及sigmoid映射得到门控值 lambda，形状为 (batch, 1)。
        """
        super(FusionModule, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        # 取最后一时刻的隐藏状态
        h_last = out[:, -1, :]
        lambda_val = self.sigmoid(self.fc(h_last))
        return lambda_val

###########################
# 模型训练函数定义
###########################
def train_idm_model(model, train_loader, optimizer, criterion, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed, batch_s_safe in train_loader:
            optimizer.zero_grad()
            # Move data to the correct device
            predicted_speed, _ = model.predict_speed(batch_data.to(device), batch_s_safe.to(device))
            loss = criterion(predicted_speed, batch_speed.to(device).unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[LSTM-IDM] Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
    return model

def train_lnn_model(model, train_loader, optimizer, criterion, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed in train_loader:
            optimizer.zero_grad()
            # Move data to the correct device
            predicted_speed = model.predict_speed(batch_data.to(device))
            loss = criterion(predicted_speed, batch_speed.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[LNN] Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
    return model

def train_fusion_module(fusion_module, idm_model, lnn_model, fusion_loader, optimizer, criterion, num_epochs=20):
    # Fixed IDM and LNN model parameters
    idm_model.eval()
    lnn_model.eval()
    fusion_module.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in fusion_loader:
            # Each batch contains:
            # fusion_input: input to fusion module, shape (batch, 5, 2)
            # idm_input: LSTM-IDM input
            # lnn_input: LNN input
            # ground_truth: true speed (scalar)
            # s_safe: safety distance parameter for IDM
            fusion_input, idm_input, lnn_input, ground_truth, s_safe = batch
            # Move all batch data to the correct device
            fusion_input = fusion_input.to(device)
            idm_input = idm_input.to(device)
            lnn_input = lnn_input.to(device)
            ground_truth = ground_truth.to(device).unsqueeze(1)  # Adjust to (batch, 1)
            s_safe = s_safe.to(device)

            with torch.no_grad():
                # Get predictions from both models, ensuring output shape is (batch, 1)
                y_ph, _ = idm_model.predict_speed(idm_input, s_safe)
                y_da = lnn_model.predict_speed(lnn_input)

            # Fusion module outputs gating value lambda (shape: (batch, 1))
            lambda_val = fusion_module(fusion_input)
            # Fused prediction (element-wise multiplication, ensure shape consistency)
            fused_output = lambda_val * y_da + (1 - lambda_val) * y_ph

            loss = criterion(fused_output, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Fusion] Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(fusion_loader):.4f}")
    return fusion_module

###########################
# 评估函数（适用于单模型或融合模型）
###########################
def evaluate_model(model, test_loader, model_type="IDM"):
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    all_predicted = []
    all_true = []
    with torch.no_grad():
        for batch in test_loader:
            if model_type == "IDM":
                batch_data, batch_speed, batch_s_safe = batch
                predicted_speed, _ = model.predict_speed(batch_data.to(device), batch_s_safe.to(device))
                gt = batch_speed.to(device).unsqueeze(1)
            elif model_type == "LNN":
                batch_data, batch_speed = batch
                predicted_speed =model.predict_speed(batch_data.to(device))
                gt = batch_speed.to(device)
            elif model_type == "Fusion":
                # For fusion model, batch contains: fusion_input, idm_input, lnn_input, ground_truth, s_safe
                fusion_input, idm_input, lnn_input, gt, s_safe = batch
                # Move all batch data to the correct device
                fusion_input = fusion_input.to(device)
                idm_input = idm_input.to(device)
                lnn_input = lnn_input.to(device)
                gt = gt.to(device).unsqueeze(1)
                s_safe = s_safe.to(device)
                lambda_val = model(fusion_input)
                with torch.no_grad():
                    # Ensure idm_model and lnn_model are accessible in this scope and on the correct device
                    y_ph, _ = idm_model.predict_speed(idm_input, s_safe)
                    y_da = lnn_model.predict_speed(lnn_input)
                predicted_speed = lambda_val * y_da + (1 - lambda_val) * y_ph
            loss = mse_loss(predicted_speed, gt)
            total_mse += loss.item()
            all_predicted.append(predicted_speed.cpu())
            all_true.append(gt.cpu())
    mse = total_mse / len(test_loader)
    rmse = torch.sqrt(torch.tensor(mse))
    mae = torch.mean(torch.abs(torch.cat(all_predicted) - torch.cat(all_true))).item()
    print(f"Evaluation Metrics:\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    # Plot first 100 samples comparison
    plt.figure(figsize=(10, 6))
    plt.plot(torch.cat(all_true)[:300].numpy(), label='True Speed', linestyle='--', marker='o')
    plt.plot(torch.cat(all_predicted)[:300].numpy(), label='Predicted Speed', linestyle='-', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    # plt.show()

###########################
# 主函数
import os
import torch
import numpy as np
import pandas as pd

def compute_position_and_spacing_and_save(
        fusion_module,
        idm_model,
        lnn_model,
        fusion_input,    # Tensor: (N_test, seq_len_fusion, 2)
        idm_input,       # Tensor: (N_test, seq_len_idm, 5)
        lnn_input,       # Tensor: (N_test, seq_len_lnn, 5)
        s_safe,          # Tensor: (N_test,)
        raw_data,        # Tensor: full raw_data from mat
        label_data,      # Tensor: full label_data from mat
        train_size,      # int, 划分训练/测试时的训练集大小
        dt=0.1,
        output_file="fusion_predictions.xlsx"
    ):
    fusion_module.eval()
    idm_model.eval()
    lnn_model.eval()

    with torch.no_grad():
        # Move inputs to the correct device before passing to models
        fusion_input_device = fusion_input.to(device)
        idm_input_device = idm_input.to(device)
        lnn_input_device = lnn_input.to(device)
        s_safe_device = s_safe.to(device)

        # 1. IDM Prediction
        y_ph, _ = idm_model.predict_speed(idm_input_device, s_safe_device)
        # 2. LNN Prediction
        y_da = lnn_model.predict_speed(lnn_input_device)
        # 3. Fusion Prediction
        lambda_val = fusion_module(fusion_input_device)
        pred_speed = (lambda_val * y_da + (1 - lambda_val) * y_ph).squeeze().cpu().numpy()

    N_test = fusion_input.shape[0]
    idx = np.arange(train_size, train_size + N_test)

    # Current Y (ft) and current speed (m/s)
    current_Y_ft   = raw_data[idx, -1, 4].numpy()           # Fifth column is Y(ft)
    current_speed_m = fusion_input[:, -1, 0].numpy()        # First column of fusion input is speed(m/s)

    # True Y (ft) and spacing (m)
    true_Y_ft      = label_data[idx, -1, 3].numpy()         # Fourth column is true Y(ft)
    true_spacing_m = label_data[idx, -1, 1].numpy() * 0.3048  # Second column is spacing(ft), convert to m

    # Kinematic integration to calculate displacement (m->ft)
    # 0.5 * a_avg * dt^2 + v0 * dt,  a_avg = (pred - v0) / dt
    disp_m    = current_speed_m * dt + 0.5 * ((pred_speed - current_speed_m) / dt) * dt**2
    disp_ft   = disp_m / 0.3048
    pred_Y_ft = current_Y_ft + disp_ft

    # Convert back to m
    pred_Y_m      = pred_Y_ft * 0.3048
    true_Y_m      = true_Y_ft * 0.3048
    pred_spacing_m = true_Y_m - pred_Y_m + true_spacing_m

    # Calculate metrics
    rmse_Y  = np.sqrt(np.mean((pred_Y_m - true_Y_m)**2))
    mape_Y  = np.mean(np.abs((pred_Y_m - true_Y_m)/true_Y_m)) * 100
    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m)**2))
    mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m)/true_spacing_m)) * 100

    print(f"[Fusion] Position Error -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
    print(f"[Fusion] Spacing  Error -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")

    # Save to Excel
    df = pd.DataFrame({
        "Pred Speed (m/s)":      pred_speed,
        "Predicted Y (m)":       pred_Y_m,
        "True Y (m)":            true_Y_m,
        "Predicted Spacing (m)": pred_spacing_m,
        "True Spacing (m)":      true_spacing_m,
    })
    sheet_name = "Fusion"

    with pd.ExcelWriter(output_file,
                        engine="openpyxl",
                        mode="a" if os.path.exists(output_file) else "w") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Fusion results saved to '{output_file}' sheet '{sheet_name}'.")


###########################
if __name__ == "__main__":
    torch.manual_seed(42)

    ##############################################
    # 数据加载与预处理（统一使用同一数据文件）
    ##############################################
    # Assume data is saved in 'data_fine_0.1.mat'
    data = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')
    # Raw data, assume variable name is 'train_data'
    raw_data = torch.tensor(data['train_data'], dtype=torch.float32)  # shape: (samples, time_steps, features)
    # Label data, assume 'lable_data' column 1 is speed
    lable_data = torch.tensor(data['lable_data'], dtype=torch.float32)
    # For IDM model, use last 50 time steps, select 5 features: first, second, third, fourth, last column
    data_idm = raw_data[:, -50:, [0, 1, 2, 3, -1]].clone()
    # For LNN model, also take last 50 time steps, 5 features
    data_lnn = raw_data[:, -50:, [0, 1, 2, 3, -1]].clone()
    # For fusion module: input is "first column" and "last column" from raw data, select last 5 time steps (including current step)
    fusion_data = raw_data[:, -5:, :].clone()  # Extract necessary columns from here
    fusion_data = fusion_data[:, :, [0, -1]]  # shape: (samples, 5, 2) # Exists error, fixed below

    # Unit conversion (e.g., ft->m, ft/s->m/s), assume all columns needing conversion are multiplied by 0.3048
    for tensor in [data_idm, data_lnn, fusion_data]:
        tensor *= 0.3048
    # Label: take speed at the last time step (lable_data[:, -1, 0] in the code)
    ground_truth = lable_data[:, -1, 0].clone() * 0.3048

    # For IDM model, need additional s_safe parameter: take the last time step of the 2nd column (front car distance) from data_idm
    s_safe = data_idm[:, -1, 1].clone()

    # Select a portion of data (e.g., first 10%) for quick experiments
    total_samples = data_idm.shape[0]
    sample_size = int(total_samples * 0.1)
    data_idm = data_idm[:sample_size]
    data_lnn = data_lnn[:sample_size]
    fusion_data = fusion_data[:sample_size]
    ground_truth = ground_truth[:sample_size]
    s_safe = s_safe[:sample_size]

    # Check data
    check_data(data_idm, "data_idm")
    check_data(ground_truth, "ground_truth")
    check_data(s_safe, "s_safe")
    check_data(fusion_data, "fusion_data")

    # Split into training and testing sets (80% train, 20% test)
    dataset_size = data_idm.shape[0]
    train_size = int(dataset_size * 0.8)

    # IDM dataset (input, ground truth, s_safe)
    train_idm_data = data_idm[:train_size]
    test_idm_data = data_idm[train_size:]
    train_idm_gt = ground_truth[:train_size]
    test_idm_gt = ground_truth[train_size:]
    train_s_safe = s_safe[:train_size]
    test_s_safe = s_safe[train_size:]

    # LNN dataset (input, ground truth)
    train_lnn_data = data_lnn[:train_size]
    test_lnn_data = data_lnn[train_size:]
    train_lnn_gt = ground_truth[:train_size].unsqueeze(1)  # Keep shape consistent
    test_lnn_gt = ground_truth[train_size:].unsqueeze(1)

    # Fusion dataset: need to provide fusion module input, and corresponding IDM and LNN input data with ground truth
    train_fusion_input = fusion_data[:train_size]
    test_fusion_input = fusion_data[train_size:]
    # For idm and lnn input, directly use their original input data
    train_fusion_idm = train_idm_data
    test_fusion_idm = test_idm_data
    train_fusion_lnn = train_lnn_data
    test_fusion_lnn = test_lnn_data
    train_fusion_gt = ground_truth[:train_size]
    test_fusion_gt = ground_truth[train_size:]
    train_fusion_s_safe = train_s_safe
    test_fusion_s_safe = test_s_safe

    # Construct input for calculating front car distance and self-car position
    test_fusion_input = fusion_data[train_size:]  # (N_test, 5, 2), already multiplied by 0.3048
    test_idm_input = data_idm[train_size:]  # (N_test, 50, 5)
    test_lnn_input = data_lnn[train_size:]  # (N_test, 50, 5)
    test_s_safe = s_safe[train_size:]  # (N_test,)

    # Create data loaders
    batch_size = 32
    idm_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_idm_data, train_idm_gt, train_s_safe),
        batch_size=batch_size, shuffle=True
    )
    idm_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_idm_data, test_idm_gt, test_s_safe),
        batch_size=batch_size, shuffle=False
    )
    lnn_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_lnn_data, train_lnn_gt),
        batch_size=batch_size, shuffle=True
    )
    lnn_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_lnn_data, test_lnn_gt),
        batch_size=batch_size, shuffle=False
    )
    # Fusion module loader: each batch returns (fusion_input, idm_input, lnn_input, ground_truth, s_safe)
    fusion_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_fusion_input, train_fusion_idm, train_fusion_lnn, train_fusion_gt, train_fusion_s_safe),
        batch_size=batch_size, shuffle=True
    )
    fusion_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_fusion_input, test_fusion_idm, test_fusion_lnn, test_fusion_gt, test_fusion_s_safe),
        batch_size=batch_size, shuffle=False
    )

    ##############################################
    # Train LSTM-IDM Model
    ##############################################
    input_dim_idm = train_idm_data.shape[2]  # 5
    hidden_dim_idm = 128
    num_layers_idm = 1
    idm_model = HybridIDMModel(input_dim_idm, hidden_dim_idm, num_layers=num_layers_idm).to(device)
    initialize_weights(idm_model)
    criterion_idm = nn.MSELoss()
    optimizer_idm = optim.Adam(idm_model.parameters(), lr=0.0005)
    num_epochs_idm = 100
    print("\n--- Training LSTM-IDM Model ---")
    idm_model = train_idm_model(idm_model, idm_train_loader, optimizer_idm, criterion_idm, num_epochs=num_epochs_idm)
    print("\n--- Evaluating LSTM-IDM Model on Test Set ---")
    evaluate_model(idm_model, idm_test_loader, model_type="IDM")

    ##############################################
    # Train LNN Model
    ##############################################
    input_dim_lnn = train_lnn_data.shape[2]# Also 5
    hidden_dim_lnn = 128
    num_layers_lnn = 1
    num_steps = train_lnn_data.shape[1]  # 50
    lnn_model = LiquidNeuralNetwork(input_dim_lnn, hidden_dim_lnn, num_layers=num_layers_lnn, num_steps=num_steps, output_dim=1).to(device)
    initialize_weights(lnn_model)
    criterion_lnn = nn.MSELoss()
    optimizer_lnn = optim.Adam(lnn_model.parameters(), lr=0.0005)
    num_epochs_lnn = 150
    print("\n--- Training Liquid Neural Network (LNN) Model ---")
    lnn_model = train_lnn_model(lnn_model, lnn_train_loader, optimizer_lnn, criterion_lnn, num_epochs=num_epochs_lnn)
    print("\n--- Evaluating LNN Model on Test Set ---")
    evaluate_model(lnn_model, lnn_test_loader, model_type="LNN")

    ##############################################
    # Train Fusion Module
    ##############################################
    # When training the fusion module, idm_model and lnn_model parameters are fixed and not updated
    fusion_module = FusionModule(input_dim=2, hidden_dim=32, num_layers=1).to(device)
    initialize_weights(fusion_module)
    criterion_fusion = nn.MSELoss()
    optimizer_fusion = optim.Adam(fusion_module.parameters(), lr=0.001)
    num_epochs_fusion = 30
    print("\n--- Training Fusion Module ---")
    fusion_module = train_fusion_module(fusion_module, idm_model, lnn_model, fusion_train_loader, optimizer_fusion, criterion_fusion, num_epochs=num_epochs_fusion)
    print("\n--- Evaluating Fusion Model on Test Set ---")
    # Call fusion evaluation function, idm_model and lnn_model are needed globally here
    # Note: The evaluation function internally calls idm_model and lnn_model, so these two models need to be globally visible
    evaluate_model(fusion_module, fusion_test_loader, model_type="Fusion")
    # Direct call
    compute_position_and_spacing_and_save(
        fusion_module=fusion_module,
        idm_model=idm_model,
        lnn_model=lnn_model,
        fusion_input=test_fusion_input,
        idm_input=test_idm_input,
        lnn_input=test_lnn_input,
        s_safe=test_s_safe,
        raw_data=raw_data,
        label_data=lable_data,
        train_size=train_size,
        dt=0.1,
        output_file="mymodel_1.xlsx"
    )


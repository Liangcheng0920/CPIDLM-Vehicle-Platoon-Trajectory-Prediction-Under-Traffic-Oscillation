#参与优化
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import scipy.io as sio
# import pandas as pd
# import numpy as np
# import math  # 导入 math 模块
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 设置环境变量，防止因重复加载库而报错
#
#
# # --- 数据检查函数 ---
# def check_data(data, name="data"):
#     """
#     检查数据中是否包含 NaN 或 Inf 值。
#     Args:
#         data (torch.Tensor): 需要检查的数据。
#         name (str): 数据名称，用于打印信息。
#     """
#     print(f"正在检查 {name} 是否包含 NaN 或 Inf 值...")
#     print(f"包含 NaN: {torch.isnan(data).any().item()}")
#     print(f"包含 Inf: {torch.isinf(data).any().item()}")
#
#
# # --- 固定 IDM 参数预测函数 ---
# def idm_fixed(v_n, s_safe, delta_v,
#               v_desired=12.64798288, T=0.50284384, a_max= 0.10033688,
#               b_safe=4.98937183, delta=1.0, s0=0.13082412,
#               delta_t=0.1):
#     """
#     基于固定参数的智能驾驶模型 (IDM) 计算下一时刻的速度。
#     Args:
#         v_n (torch.Tensor): 当前时刻车辆速度。
#         s_safe (torch.Tensor): 当前时刻与前车的安全距离 (实际间距)。
#         delta_v (torch.Tensor): 当前时刻与前车的速度差 (v_n - v_leader)。
#         v_desired (float):期望速度。
#         T (float): 安全时间 headway。
#         a_max (float): 最大加速度。
#         b_safe (float):舒适减速度。
#         delta (float): 加速度指数。
#         s0 (float): 静止时最小间距。
#         delta_t (float): 时间步长 (此处IDM模型内部使用，与外部的dt可能不同，但通常假设一致)。
#     Returns:
#         torch.Tensor: IDM预测的下一时刻速度。
#     """
#     device = v_n.device  # 获取张量所在的设备
#     # 把常数转成 0-d tensor 并确保它们有最小值以避免除零等问题
#     a_max_t = torch.tensor(a_max, device=device).clamp(min=1e-6)
#     b_safe_t = torch.tensor(b_safe, device=device).clamp(min=1e-6)
#     s_safe_t = s_safe.clamp(min=1e-6)  # 确保安全距离大于0
#     v_desired_t = torch.tensor(v_desired, device=device)
#     s0_t = torch.tensor(s0, device=device)
#     T_t = torch.tensor(T, device=device)
#     delta_t_val = torch.tensor(delta_t, device=device)
#     delta_exp = torch.tensor(delta, device=device)
#
#     # 计算期望间距 s_star
#     # s* = s0 + v*T + (v * Δv) / (2 * sqrt(a_max * b_safe))
#     s_star = s0_t + v_n * T_t + (v_n * delta_v) / (2 * torch.sqrt(a_max_t * b_safe_t))
#     s_star = s_star.clamp(min=0.0)  # 期望间距不能为负
#
#     # IDM 加速度公式: a_idm = a_max * (1 - (v/v_desired)^delta - (s_star/s_actual)^2)
#     # 此处 s_safe 被用作 s_actual
#     acceleration_idm = a_max_t * (
#             1 - (v_n / v_desired_t).pow(delta_exp) - (s_star / s_safe_t).pow(2)
#     )
#
#     # 更新速度: v_next = v_n + delta_t * a_idm
#     v_follow = v_n + delta_t_val * acceleration_idm
#     return v_follow.clamp(min=0.0)  # 速度不能为负
#
#
# # --- 位置编码 (Positional Encoding) ---
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         """
#         位置编码层初始化。
#         Args:
#             d_model (int): 模型的特征维度 (embedding dim)。
#             dropout (float): Dropout 比率。
#             max_len (int): 支持的最大序列长度。
#         """
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # # (1, max_len, d_model)
#         self.register_buffer('pe', pe)  # 注册为buffer，不参与梯度更新
#
#     def forward(self, x):
#         """
#         前向传播。
#         Args:
#             x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。
#                                Transformer的输入特征维度已调整为d_model。
#         Returns:
#             torch.Tensor: 添加了位置编码的输入张量。
#         """
#         # x 的形状: (batch_size, seq_len, d_model)
#         # self.pe[:, :x.size(1), :] 的形状: (1, seq_len, d_model)
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)
#
#
# # --- 定义新的基于Transformer的融合模型 ---
# class HybridIDMTransformerModel(nn.Module):
#     def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, input_seq_len=50):
#         """
#         基于Transformer的混合IDM模型初始化。
#         Args:
#             input_dim (int): 输入序列中每个时间步的特征数量。
#             d_model (int): Transformer模型的内部特征维度 (embedding dimension)。
#             nhead (int): 多头注意力机制中的头数。
#             num_encoder_layers (int): Transformer编码器的层数。
#             dim_feedforward (int): Transformer编码器中前馈网络层的维度。
#             dropout (float): Dropout比率。
#             input_seq_len (int): 输入序列的长度，用于位置编码。
#         """
#         super(HybridIDMTransformerModel, self).__init__()
#         self.d_model = d_model
#
#         # 输入嵌入层：将原始 input_dim 维特征映射到 d_model 维
#         self.input_embedder = nn.Linear(input_dim, d_model)
#         # 位置编码器
#         self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=input_seq_len)
#
#         # 定义Transformer编码器层
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True  # 输入输出格式为 (batch, seq, feature)
#         )
#         # Transformer编码器
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#
#         # 全连接层：将Transformer的输出映射到单个速度预测值
#         self.fc = nn.Linear(d_model, 1)  # 输出一个预测速度
#
#         # 可学习的融合权重 alpha_raw，通过 sigmoid 映射到 [0,1]
#         self.alpha_raw = nn.Parameter(torch.tensor(0.0))  # 初始化为0，使得sigmoid(0)=0.5
#
#     def forward(self, x, s_safe):
#         """
#         模型前向传播。
#         Args:
#             x (torch.Tensor):  输入序列，形状为 (batch_size, seq_len, input_dim)。
#             s_safe (torch.Tensor): 当前安全距离 (实际观测到的与前车的间距)，形状为 (batch_size,)。
#         Returns:
#             Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#                 - y_transformer (torch.Tensor): Transformer直接输出的预测速度，形状 (batch_size,)。
#                 - y_idm (torch.Tensor): 用固定参数IDM计算的预测速度，形状 (batch_size,)。
#                 - alpha (torch.Tensor): 当前学习到的融合权重标量。
#         """
#         # Transformer 部分
#         # 1. 输入嵌入并调整维度
#         x_embedded = self.input_embedder(x) * math.sqrt(self.d_model)  # (batch, seq_len, d_model)
#         # 2. 添加位置编码
#         x_pos_encoded = self.pos_encoder(x_embedded)  # (batch, seq_len, d_model)
#         # 3. Transformer编码器处理
#         transformer_out = self.transformer_encoder(x_pos_encoded)  # (batch, seq_len, d_model)
#         # 4. 取序列最后一个时间步的输出进行预测
#         y_transformer = self.fc(transformer_out[:, -1, :]).squeeze(1)  # (batch,)
#
#         # IDM 部分
#         # 计算固定参数IDM的预测速度
#         # 需要从输入x中提取IDM所需的参数：
#         # v_n: 当前车辆速度，假设是输入特征的第0列 (x[:, -1, 0])
#         # delta_v: 与前车的速度差，假设是输入特征的第2列 (x[:, -1, 2])
#         # s_safe: 作为参数传入，是当前的实际观测间距
#         v_n = x[:, -1, 0]  # 当前时刻自身速度
#         delta_v = x[:, -1, 2]  # 当前时刻与前车速度差 (v_self - v_leader)
#         # 注意：IDM公式中的 delta_v 通常是 v_self - v_leader。
#         # 如果输入x中的delta_v是 v_leader - v_self，则需要取反。
#         # 假设输入x的第2列已经是 v_self - v_leader
#
#         y_idm = idm_fixed(v_n, s_safe, delta_v)  # 调用IDM函数
#
#         # 融合权重
#         alpha = torch.sigmoid(self.alpha_raw)  # 将alpha_raw映射到(0,1)区间，作为标量权重
#
#         return y_transformer, y_idm, alpha
#
#
# def initialize_weights(model):
#     """
#     初始化模型权重。
#     Args:
#         model (nn.Module): 需要初始化的模型。
#     """
#     for name, param in model.named_parameters():
#         if param.dim() > 1:  # 通常权重矩阵是二维或更高维度 (例如 nn.Linear, nn.Transformer的内部权重)
#             if "weight" in name:
#                 nn.init.xavier_uniform_(param)  # 使用Xavier均匀分布初始化
#         elif "bias" in name:  # 偏置项
#             nn.init.constant_(param, 0)  # 初始化为0
#         # alpha_raw 是标量参数，可以保持默认初始化或特定设置，这里不特别处理
#
#
# # --- 训练函数 ---
# def train_model(model, train_loader, optimizer, num_epochs=30):
#     """
#     训练模型。
#     Args:
#         model (nn.Module): 待训练的模型 (HybridIDMTransformerModel)。
#         train_loader (torch.utils.data.DataLoader): 训练数据加载器。
#         optimizer (torch.optim.Optimizer): 优化器。
#         num_epochs (int): 训练轮数。
#     Returns:
#         nn.Module: 训练好的模型。
#     """
#     model.train()  # 设置模型为训练模式
#     for epoch in range(num_epochs):
#         epoch_loss = 0.0
#         for batch_x, batch_y, batch_s_safe in train_loader:  # 遍历每个批次数据
#             optimizer.zero_grad()  # 清空梯度
#
#             # 模型前向传播，获取Transformer预测、IDM预测和融合权重alpha
#             y_model_pred, y_idm_pred, alpha = model(batch_x, batch_s_safe)
#
#             # 计算加权MSE损失函数
#             # Loss = alpha * MSE(model_pred, true_y) + (1-alpha) * MSE(model_pred, idm_pred)
#             # 第一个项使模型预测接近真实值，第二个项作为正则化，使模型预测在不确定时接近IDM的物理行为
#             loss_term_data = (y_model_pred - batch_y).pow(2).mean()  # MSE(model_pred, true_y)
#             loss_term_idm = (y_model_pred - y_idm_pred).pow(2).mean()  # MSE(model_pred, idm_pred)
#
#             loss = alpha * loss_term_data + (1 - alpha) * loss_term_idm
#
#             loss.backward()  # 反向传播计算梯度
#             optimizer.step()  # 更新模型参数
#
#             epoch_loss += loss.item()  # 累积批次损失
#
#         # 打印当前轮的平均损失和学习到的alpha值
#         print(f"轮次 {epoch + 1}/{num_epochs}  损失: {epoch_loss / len(train_loader):.6f}  α={alpha.item():.4f}")
#     return model
#
#
# # --- 测试/评估函数 ---
# def evaluate_model(model, test_loader):
#     """
#     评估模型。
#     Args:
#         model (nn.Module): 待评估的模型。
#         test_loader (torch.utils.data.DataLoader): 测试数据加载器。
#     """
#     model.eval()  # 设置模型为评估模式
#     all_model_preds, all_true_ys, all_idm_preds = [], [], []  # 存储预测值和真实值
#     final_alpha = 0.0
#
#     with torch.no_grad():  # 测试时不需要计算梯度
#         for batch_x, batch_y, batch_s_safe in test_loader:
#             y_model_pred, y_idm_pred, alpha = model(batch_x, batch_s_safe)
#
#             # 收集模型直接输出的预测（融合前的）和真实值
#             all_model_preds.append(y_model_pred.cpu())
#             all_true_ys.append(batch_y.cpu())
#             all_idm_preds.append(y_idm_pred.cpu())  # 也收集IDM的预测，或许可以用于分析
#             final_alpha = alpha.item()  # 获取最后一批的alpha值（所有批次alpha应该一样）
#
#     y_model_pred_cat = torch.cat(all_model_preds)  # 拼接所有批次的模型预测
#     y_true_cat = torch.cat(all_true_ys)  # 拼接所有批次的真实值
#     y_idm_pred_cat = torch.cat(all_idm_preds)  # 拼接所有批次的IDM预测
#
#     # 计算融合后的最终预测 (根据学习到的alpha)
#     # 注意: 这里的alpha是训练后得到的最终值，它是一个标量
#     # 最终预测 = alpha * Transformer预测 + (1-alpha) * IDM预测
#     # 然而，论文或通常做法可能是用 y_model_pred 直接作为模型的输出，而损失函数引导它学习
#     # 如果损失函数是 L = alpha * MSE(y_nn, y_true) + (1-alpha)*MSE(y_nn, y_idm)
#     # 那么 y_nn (即 y_model_pred_cat) 是模型学习到的主要输出。
#     # 评估时，我们就用这个 y_model_pred_cat。alpha的作用是在训练中平衡两个目标。
#
#     mse = nn.MSELoss()(y_model_pred_cat, y_true_cat).item()
#     rmse = torch.sqrt(torch.tensor(mse)).item()
#     mae = torch.mean(torch.abs(y_model_pred_cat - y_true_cat)).item()
#
#     print(f"\n测试结果 -- MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, α={final_alpha:.4f}")
#
#     # 绘图比较前100个点的真实值和模型预测值
#     plt.figure(figsize=(10, 6))
#     plt.plot(y_true_cat[:100].numpy(), '--o', label='真实值 (True)')
#     plt.plot(y_model_pred_cat[:100].numpy(), '-x', label='模型预测值 (Model Pred)')
#     # 可以选择性绘制IDM的预测作为参考
#     # plt.plot(y_idm_pred_cat[:100].numpy(), ':s', label='IDM 预测值 (IDM Pred)')
#     plt.title("模型预测与真实值对比 (前100个测试点)")
#     plt.xlabel("样本点索引")
#     plt.ylabel("速度 (m/s)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# # === 修改后的 compute_position_and_spacing_and_save ===
# def compute_position_and_spacing_and_save(model,
#                                           test_input_data,  # 测试集的输入序列 (N_test, seq_len, feat_dim)
#                                           test_true_speed,  # 测试集的真实下一时刻速度 (N_test,)
#                                           original_raw_data_all,  # 完整的原始数据 (N_all, full_seq_len, raw_feat_dim)
#                                           original_label_data_all,  # 完整的标签数据 (N_all, label_seq_len, label_feat_dim)
#                                           test_set_start_index,  # 测试集在 original_raw_data_all 中的起始索引
#                                           test_s_safe_values,  # 测试集对应的安全距离s_safe (N_test,)
#                                           dt=0.1,  # 时间步长 (秒)
#                                           output_file="predictions_extended.xlsx",
#                                           sheet_name_prefix="PID-Transformer-IDM"):  # 表格名称前缀
#     """
#     计算预测的位置、间距，并与真实值比较，然后保存到Excel。
#     Args:
#         model: 已训练好的HybridIDMTransformerModel。
#         test_input_data: 测试集的输入特征数据。
#         test_true_speed: 测试集的真实目标速度。
#         original_raw_data_all: 完整的原始特征数据 (单位通常是ft, ft/s)。
#         original_label_data_all: 完整的原始标签数据 (单位通常是ft, ft/s)。
#         test_set_start_index: 测试数据在完整原始数据中的起始行号。
#         test_s_safe_values: 测试集中每个样本对应的安全间距s_safe (m)。
#         dt: 时间步长 (s)。
#         output_file: 输出Excel文件名。
#         sheet_name_prefix: Excel中工作表名的前缀。
#     """
#     model.eval()  # 设置模型为评估模式
#     with torch.no_grad():
#         # 1) 调用模型，获取 Transformer 的直接预测 (y_model_direct_pred)、IDM 的预测 (y_idm_calc_pred) 和学习到的 alpha
#         y_model_direct_pred, y_idm_calc_pred, learned_alpha = model(test_input_data, test_s_safe_values)
#
#         # 2) 确定最终用于评估和位置计算的预测速度
#         # 根据训练时的损失函数设计，y_model_direct_pred 是模型被优化的目标之一。
#         # alpha 的作用是平衡数据驱动项和物理模型引导项。
#         # 因此，评估时我们主要关心 y_model_direct_pred 的性能。
#         # 如果需要“融合后”的预测，则： pred_speed_final = learned_alpha * y_model_direct_pred + (1 - learned_alpha) * y_idm_calc_pred
#         # 这里我们使用 y_model_direct_pred 作为模型的输出进行后续计算。
#         pred_speed_model_output_m_s = y_model_direct_pred.cpu().numpy()
#
#     true_speed_m_s = test_true_speed.cpu().numpy()  # 真实的目标速度 (m/s)
#
#     N_test_samples = test_input_data.size(0)  # 测试样本数量
#     # 获取测试样本在原始数据(original_raw_data_all)中的实际索引范围
#     indices_in_original_data = np.arange(test_set_start_index, test_set_start_index + N_test_samples)
#
#     # 3) 从 original_raw_data_all 中提取当前时刻的Y坐标 (假设是第5列，索引4, 单位ft)
#     #    和当前时刻的速度 (用于运动学公式的v_prev)
#     #    raw_data 的最后一帧 ([:, -1, :]) 对应 test_input_data 输入序列的最后一帧
#     current_Y_ft_original = original_raw_data_all[indices_in_original_data, -1, 4].cpu().numpy()  # 当前Y坐标 (ft)
#     # 当前速度 (m/s) - 从转换后的 test_input_data 中获取更准确，因为它已经是m/s
#     current_speed_m_s_input = test_input_data[:, -1, 0].cpu().numpy()  # 当前速度 (m/s)
#
#     # 从 original_label_data_all 中提取真实的下一时刻Y坐标 (假设是第4列，索引3, 单位ft)
#     # 和真实的下一时刻间距 (假设是第2列，索引1, 单位ft，需要转为m)
#     true_next_Y_ft_original = original_label_data_all[indices_in_original_data, -1, 3].cpu().numpy()  # 真实下一时刻Y坐标 (ft)
#     true_next_spacing_ft_original = original_label_data_all[
#         indices_in_original_data, -1, 1].cpu().numpy()  # 真实下一时刻间距 (ft)
#     true_next_spacing_m = true_next_spacing_ft_original * 0.3048  # 转换为米
#
#     # 4) 计算位移: disp = v_prev*dt + 0.5*a*dt^2
#     #    v_prev_m_s 是当前时刻的速度 (current_speed_m_s_input)
#     #    v_next_pred_m_s 是模型预测的下一时刻速度 (pred_speed_model_output_m_s)
#     acceleration_pred_m_s2 = (pred_speed_model_output_m_s - current_speed_m_s_input) / dt  # 预测的平均加速度 (m/s^2)
#     displacement_pred_m = current_speed_m_s_input * dt + 0.5 * acceleration_pred_m_s2 * dt ** 2  # 预测的位移 (m)
#
#     # 预测的下一时刻Y坐标 (m)
#     current_Y_m = current_Y_ft_original * 0.3048  # 当前Y坐标转换为米
#     pred_next_Y_m = current_Y_m + displacement_pred_m
#
#     # 5) 真实的下一时刻Y坐标 (m)
#     true_next_Y_m = true_next_Y_ft_original * 0.3048
#
#     # 计算预测的下一时刻与前车间距 (m)
#     # 假设：真实下一时刻前车Y坐标 (Y_leader_next) 可以从 true_next_Y_m 和 true_next_spacing_m 推算
#     # Y_leader_next_m = true_next_Y_m + true_next_spacing_m (如果Y是车辆后端，间距是车头到前车尾)
#     # 或者 Y_leader_next_m = true_next_Y_m - true_next_spacing_m (如果Y是车辆前端...)
#     # 这里需要明确坐标定义和间距定义。
#     # 假设 Y 是车辆后端全局坐标，Spacing 是本车车头到前车车尾的距离。
#     # 假设车长 L_veh。本车车头位置 Y_head_current = current_Y_m + L_veh。
#     # 前车车尾位置 Y_leader_tail_current = Y_head_current + s_safe (s_safe 是当前观测间距)
#     # 如果我们只关心相对位置变化，且前车速度未知，则无法精确预测下一时刻绝对间距。
#     # 题目中的 `compute_position_and_spacing_and_save` 原函数逻辑:
#     # `pred_spacing_m = (true_Y_ft - pred_Y_ft) * 0.3048 + true_spacing_m`
#     # 这似乎是基于一个假设：真实的前车位置已知，或者说前车的位移已知。
#     # (true_Y_ft - pred_Y_ft) * 0.3048 是本车真实位移与预测位移的差值。
#     # 这个差值被加到真实的下一时刻间距上。
#     # 这意味着如果本车预测走的更远 (pred_Y > true_Y), 那么 (true_Y - pred_Y) 为负，会使得预测间距减小。
#     # 反之，如果本车预测走的更近 (pred_Y < true_Y), (true_Y - pred_Y) 为正，预测间距增大。
#     # 这实际上是在调整真实间距，以反映本车预测位置的偏差。
#
#     # 沿用原函数的间距计算逻辑，注意单位统一为米
#     # (true_next_Y_m - pred_next_Y_m) 是真实位移与预测位移之差造成的位置误差
#     # 这个误差被加到真实的下一时刻间距上
#     pred_next_spacing_m = (true_next_Y_m - pred_next_Y_m) + true_next_spacing_m
#
#     # 6) 打印误差
#     # 过滤掉真实值为0的情况以避免MAPE计算错误
#     valid_Y_mask = true_next_Y_m != 0
#     if np.sum(valid_Y_mask) > 0:
#         rmse_Y = np.sqrt(np.mean((pred_next_Y_m[valid_Y_mask] - true_next_Y_m[valid_Y_mask]) ** 2))
#         mape_Y = np.mean(
#             np.abs((pred_next_Y_m[valid_Y_mask] - true_next_Y_m[valid_Y_mask]) / true_next_Y_m[valid_Y_mask])) * 100
#     else:
#         rmse_Y = np.sqrt(np.mean((pred_next_Y_m - true_next_Y_m) ** 2))  # Fallback if all true_Y are zero
#         mape_Y = float('inf')
#         print("警告: 所有真实Y坐标为0，MAPE(Y)计算可能不准确。")
#
#     valid_spacing_mask = true_next_spacing_m != 0
#     if np.sum(valid_spacing_mask) > 0:
#         rmse_sp = np.sqrt(
#             np.mean((pred_next_spacing_m[valid_spacing_mask] - true_next_spacing_m[valid_spacing_mask]) ** 2))
#         mape_sp = np.mean(np.abs(
#             (pred_next_spacing_m[valid_spacing_mask] - true_next_spacing_m[valid_spacing_mask]) / true_next_spacing_m[
#                 valid_spacing_mask])) * 100
#     else:
#         rmse_sp = np.sqrt(np.mean((pred_next_spacing_m - true_next_spacing_m) ** 2))  # Fallback
#         mape_sp = float('inf')
#         print("警告: 所有真实间距为0，MAPE(Spacing)计算可能不准确。")
#
#     print(f"位置预测误差 (Y) -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
#     print(f"间距预测误差 (Spacing) -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")
#
#     # 7) 保存到 Excel
#     df = pd.DataFrame({
#         "Pred_Speed_Model (m/s)": pred_speed_model_output_m_s,  # 模型直接预测的速度
#         "True_Speed (m/s)": true_speed_m_s,  # 真实的下一时刻速度
#         "Pred_IDM_Speed (m/s)": y_idm_calc_pred.cpu().numpy(),  # IDM计算的速度 (供参考)
#         "Learned_Alpha": learned_alpha.item(),  # 学习到的Alpha值 (重复N_test_samples次)
#         "Predicted_Next_Y (m)": pred_next_Y_m,  # 预测的下一时刻Y坐标
#         "True_Next_Y (m)": true_next_Y_m,  # 真实的下一时刻Y坐标
#         "Predicted_Next_Spacing (m)": pred_next_spacing_m,  # 预测的下一时刻间距
#         "True_Next_Spacing (m)": true_next_spacing_m,  # 真实的下一时刻间距
#         "Current_Observed_Spacing_s_safe (m)": test_s_safe_values.cpu().numpy()  # 输入的s_safe (供参考)
#     })
#     # 如果文件已存在，则以追加模式添加新的sheet，否则创建新文件
#     # 为了确保sheet name唯一，可以加上时间戳或计数
#     final_sheet_name = f"{sheet_name_prefix}_{pd.Timestamp.now().strftime('%H%M%S')}"
#
#     if os.path.exists(output_file):
#         mode = "a"
#         if_sheet_exists = 'replace'  # 或者 'new' 来创建新名字的sheet
#     else:
#         mode = "w"
#         if_sheet_exists = None  # 无需此参数
#
#     with pd.ExcelWriter(output_file, engine="openpyxl", mode=mode) as writer:
#         df.to_excel(writer, sheet_name=final_sheet_name, index=False)
#         if mode == "a" and if_sheet_exists == 'replace':
#             print(f"结果已追加/替换到 '{output_file}' 的工作表 '{final_sheet_name}'.")
#         else:
#             print(f"结果已保存到 '{output_file}' 的新工作表 '{final_sheet_name}'.")
#
#
# # --- 主流程 ---
# if __name__ == "__main__":
#     torch.manual_seed(42)  # 设置随机种子保证结果可复现
#
#     # --- 1. 加载 MAT 数据 ---
#     # !!! 请确保以下路径是正确的 !!!
#     data_path = 'E:\\pythonProject1\\data_fine_0.1.mat'
#     try:
#         data = sio.loadmat(data_path)
#     except FileNotFoundError:
#         print(f"错误: 数据文件 '{data_path}' 未找到。请检查路径。")
#         exit()
#
#     original_raw_data = torch.tensor(data['train_data'], dtype=torch.float32)  # 原始特征数据
#     original_label_data = torch.tensor(data['lable_data'], dtype=torch.float32)  # 原始标签数据
#
#     # --- 2. 数据预处理与特征选择 ---
#     # 根据原代码，输入特征 seq[:, :, [0,1,2,3,4]] 对应:
#     # [0]: v_n (当前车速)
#     # [1]: s_safe (当前安全距离/实际间距)
#     # [2]: Δv (与前车速度差 v_n - v_leader)
#     # [3]: 加速度 (本车当前加速度)
#     # [4]: 前车速度 (v_leader)
#     # 输入序列长度为最后50步
#     input_seq_len = 50
#     selected_features_indices = [0, 1, 2, 3, -1]  # 选择的特征列索引
#
#     # 准备输入序列 (input_x)
#     input_x_ft_fps = original_raw_data[:, -input_seq_len:, selected_features_indices].clone()
#
#     # 准备目标速度 (target_y)，即下一时刻的本车速度
#     # 标签数据 lab[:, -1, 0] 表示最后一个时间步的第0列特征，这里假设是下一时刻的速度
#     target_y_ft_fps = original_label_data[:, -1, 0].clone().squeeze()  # (N_samples,)
#
#     # 提取当前的安全距离/实际间距 s_safe (用于IDM计算和模型输入)
#     # 从输入序列的最后一个时间步提取 s_safe，对应特征索引1
#     current_s_safe_ft = input_x_ft_fps[:, -1, 1].clone()  # (N_samples,)
#
#     # 单位转换: ft -> m, ft/s -> m/s
#     conversion_factor = 0.3048
#     input_x_m_mps = input_x_ft_fps.clone()
#     # 速度特征 (索引0, 2, 4) 和加速度特征 (索引3) 需要转换
#     # 距离/间距特征 (索引1) 也需要转换
#     # input_x_m_mps[:, :, [0,2,3,4]] *= conversion_factor # 错误：加速度单位是ft/s^2 -> m/s^2
#     input_x_m_mps[:, :, 0] *= conversion_factor  # v_n (ft/s -> m/s)
#     input_x_m_mps[:, :, 1] *= conversion_factor  # s_safe (ft -> m)
#     input_x_m_mps[:, :, 2] *= conversion_factor  # delta_v (ft/s -> m/s)
#     input_x_m_mps[:, :, 3] *= conversion_factor  # acceleration (ft/s^2 -> m/s^2)
#     input_x_m_mps[:, :, 4] *= conversion_factor  # v_leader (ft/s -> m/s)
#
#     target_y_m_mps = target_y_ft_fps * conversion_factor
#     current_s_safe_m = current_s_safe_ft * conversion_factor
#
#     # --- 3. 数据抽样 (例如，使用前10%的数据以加速示例) ---
#     sampling_fraction = 0.05  # 使用10%的数据
#     # sampling_fraction = 1.0 # 使用全部数据
#
#     num_total_samples = input_x_m_mps.size(0)
#     num_samples_to_use = int(num_total_samples * sampling_fraction)
#
#     sampled_input_x = input_x_m_mps[:num_samples_to_use]
#     sampled_target_y = target_y_m_mps[:num_samples_to_use]
#     sampled_s_safe = current_s_safe_m[:num_samples_to_use]
#
#     print(f"原始总样本数: {num_total_samples}")
#     print(f"抽样后用于实验的样本数: {num_samples_to_use}")
#     check_data(sampled_input_x, "抽样后的输入X")
#     check_data(sampled_target_y, "抽样后的目标Y")
#     check_data(sampled_s_safe, "抽样后的s_safe")
#
#     # --- 4. 划分训练集和测试集 (例如80/20划分) ---
#     split_ratio = 0.8
#     # split_index 是在 *抽样后* 数据集中的分割点
#     split_index = int(num_samples_to_use * split_ratio)
#
#     # 这个 train_size_in_sampled 是指在抽样后的数据中，训练集的大小。
#     # 它也对应于测试集在 *原始完整数据集* 中的起始索引（因为我们是从头部开始抽样的）。
#     train_set_size_after_sampling = split_index
#
#     train_seq_tensor = sampled_input_x[:split_index]
#     test_seq_tensor = sampled_input_x[split_index:]
#
#     train_y_tensor = sampled_target_y[:split_index]
#     test_y_tensor = sampled_target_y[split_index:]
#
#     train_s_safe_tensor = sampled_s_safe[:split_index]
#     test_s_safe_tensor = sampled_s_safe[split_index:]
#
#     print(f"训练集大小: {train_seq_tensor.size(0)}")
#     print(f"测试集大小: {test_seq_tensor.size(0)}")
#
#     # 创建 TensorDataset 和 DataLoader
#     batch_size = 32
#     train_dataset = torch.utils.data.TensorDataset(train_seq_tensor, train_y_tensor, train_s_safe_tensor)
#     test_dataset = torch.utils.data.TensorDataset(test_seq_tensor, test_y_tensor, test_s_safe_tensor)
#
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     # --- 5. 模型配置、初始化与优化器设定 ---
#     input_feature_dim = sampled_input_x.size(2)  # 输入特征的数量
#     # Transformer 超参数
#     d_model = 128  # Transformer 内部特征维度 (原hidden_dim)
#     nhead = 4  # 多头注意力头数 (d_model 需能被 nhead 整除)
#     num_encoder_layers = 2  # Transformer编码器层数 (原num_layers可借鉴)
#     dim_feedforward = 256  # 前馈网络隐藏层维度
#     transformer_dropout = 0.1  # Dropout率
#
#     # 实例化基于Transformer的混合模型
#     hybrid_model = HybridIDMTransformerModel(
#         input_dim=input_feature_dim,
#         d_model=d_model,
#         nhead=nhead,
#         num_encoder_layers=num_encoder_layers,
#         dim_feedforward=dim_feedforward,
#         dropout=transformer_dropout,
#         input_seq_len=input_seq_len  # 序列长度
#     )
#
#     initialize_weights(hybrid_model)  # 初始化模型权重
#     optimizer = optim.Adam(hybrid_model.parameters(), lr=5e-4)  # Adam优化器
#
#     # --- 6. 模型训练 ---
#     num_epochs_train = 100  # 训练轮数 (可调整)
#     print("\n开始训练基于Transformer的混合IDM模型...")
#     trained_hybrid_model = train_model(hybrid_model, train_loader, optimizer, num_epochs=num_epochs_train)
#
#     # --- 7. 模型评估 ---
#     print("\n开始评估模型...")
#     evaluate_model(trained_hybrid_model, test_loader)
#
#     # --- 8. 计算位置/间距并保存结果 ---
#     print("\n开始计算位置/间距并保存结果...")
#     # test_set_start_index_in_original_data 是测试集在 *未抽样* 的原始数据中的索引起始点
#     # 因为我们是按顺序抽样和划分的，这个索引起始点就是 train_set_size_after_sampling
#     test_set_start_index_in_original_data = train_set_size_after_sampling
#
#     compute_position_and_spacing_and_save(
#         model=trained_hybrid_model,
#         test_input_data=test_seq_tensor,  # 测试集的输入序列 (m, m/s)
#         test_true_speed=test_y_tensor,  # 测试集的真实下一时刻速度 (m/s)
#         original_raw_data_all=original_raw_data,  # 完整的原始特征数据 (ft, ft/s)
#         original_label_data_all=original_label_data,  # 完整的原始标签数据 (ft, ft/s)
#         test_set_start_index=test_set_start_index_in_original_data,  # 测试集在完整数据中的起始索引
#         test_s_safe_values=test_s_safe_tensor,  # 测试集对应的s_safe值 (m)
#         dt=0.1,  # 时间步长 (s)
#         output_file="predictions_extended.xlsx",  # 输出文件名
#         sheet_name_prefix="HybridTransformerIDM"  # Excel表名
#     )
#     print("\n所有流程执行完毕。")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import glob  # 用于查找文件路径
import os  # 操作系统接口，用于路径操作和环境变量
import math # Transformer Positional Encoding 需要

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许多个OpenMP库存在，避免某些环境下的冲突

# --- 全局路径定义 ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # 数据集存放目录 (请根据您的实际路径修改)
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim_modified_transformer"  # 实验结果保存目录 (修改了目录名以区分)

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
    # device = v_n.device # 获取输入张量所在的设备 (这行是正确的)
    # 从输入张量v_n推断设备，确保所有中间张量都在同一设备上
    current_device = v_n.device
    # 将IDM参数转换为与输入数据相同设备和类型的张量
    v_desired = torch.tensor(v_desired, device=current_device, dtype=v_n.dtype)
    T = torch.tensor(T, device=current_device, dtype=v_n.dtype)
    a_max = torch.tensor(a_max, device=current_device, dtype=v_n.dtype).clamp(min=1e-6) # 避免除以零
    b_safe = torch.tensor(b_safe, device=current_device, dtype=v_n.dtype).clamp(min=1e-6) # 避免除以零
    s0 = torch.tensor(s0, device=current_device, dtype=v_n.dtype)
    delta_param = torch.tensor(delta, device=current_device, dtype=v_n.dtype) # 'delta'是IDM中的指数参数
    delta_t_tensor = torch.tensor(delta_t, device=current_device, dtype=v_n.dtype)

    s_safe = s_safe.clamp(min=1e-6) # 确保安全间距为正，避免计算错误

    # 计算期望间距 s*
    # 为了数值稳定性，在分母中添加一个很小的数 (1e-6)
    s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe) + 1e-6)
    s_star = s_star.clamp(min=0.0) # 期望间距不能为负

    # 计算速度项 (v_n / v_desired)^delta
    v_n_ratio = torch.zeros_like(v_n) # 初始化为零
    mask_v_desired_nonzero = v_desired.abs() > 1e-6 # 创建一个掩码，标记期望速度不为零的位置
    if mask_v_desired_nonzero.any(): # 仅在期望速度不为零时计算比率
        v_n_ratio[mask_v_desired_nonzero] = (v_n[mask_v_desired_nonzero] / v_desired[mask_v_desired_nonzero])

    # IDM加速度公式
    acceleration_term = a_max * (
            1 - v_n_ratio ** delta_param - (s_star / s_safe) ** 2
    )
    # 根据加速度更新速度
    v_follow = v_n + delta_t_tensor * acceleration_term
    return v_follow.clamp(min=0.0) # 速度不能为负


# --- Transformer 位置编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50): # d_model是模型的维度 (embedding_dim), max_len是序列最大长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # 创建一个 (max_len, d_model) 的零张量用于存储位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # 创建一个 (max_len, 1) 的张量表示位置
        # 计算div_term，用于缩放不同维度的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # 计算偶数索引的正弦编码
        pe[:, 1::2] = torch.cos(position * div_term) # 计算奇数索引的余弦编码
        pe = pe.unsqueeze(0) # 增加一个维度以匹配批处理格式 (1, max_len, d_model) -> (1, seq_len, d_model)
        self.register_buffer('pe', pe) # 将pe注册为buffer，这样它不会被视为模型参数，但会随模型移动（例如.to(device)）

    def forward(self, x):
        """
        x: 输入张量，形状为 (batch_size, seq_len, d_model)
        """
        # x 的 seq_len 可能小于 max_len，所以我们只取需要的部分
        # 将位置编码加到输入张量 x 上。x 的第二维是序列长度。
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- 定义新的基于 Transformer 的融合模型 ---
class HybridIDMTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, # 基本参数
                 nhead=4, transformer_num_layers=2, dim_feedforward=512, dropout_transformer=0.1): # Transformer特定参数
        super(HybridIDMTransformerModel, self).__init__()
        self.pred_horizon = output_dim  # 预测步长 K
        self.model_dim = hidden_dim # Transformer的内部维度 (d_model)

        # 输入线性层：将原始input_dim映射到Transformer的model_dim
        self.input_fc = nn.Linear(input_dim, self.model_dim)
        # 位置编码器
        self.pos_encoder = PositionalEncoding(self.model_dim, dropout_transformer)
        # Transformer编码器层定义
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nhead, # 多头注意力头数
            dim_feedforward=dim_feedforward, # 前馈网络层维度
            dropout=dropout_transformer,
            batch_first=True # 输入和输出张量的形状为 (batch, seq, feature)
        )
        # 组合多个Transformer编码器层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=transformer_num_layers)
        # 输出线性层：将Transformer最后一个时间步的输出映射到K个预测值
        self.fc = nn.Linear(self.model_dim, self.pred_horizon)

        # IDM参数 (固定值)
        self.v_desired_idm = 12.64798288
        self.T_idm = 0.50284384
        self.a_max_idm = 0.10033688
        self.b_safe_idm = 4.98937183
        self.delta_idm = 1.0
        self.s0_idm = 0.13082412

    def forward(self, x, s_safe_initial, v_lead_initial):
        """
        模型的前向传播。
        :param x: 输入序列, shape=(batch, seq_len, input_dim)
        :param s_safe_initial: 当前安全距离 (真实观测值，用于IDM多步迭代), shape=(batch,)
        :param v_lead_initial: 当前前车速度 (真实观测值，用于IDM迭代，假设在K步内保持不变), shape=(batch,)
        :return:
          y_nn_multistep: 基于Transformer的K步速度预测, shape=(batch, K)
          y_idm_multistep: 基于IDM迭代的K步速度预测, shape=(batch, K)
        """
        # device = x.device # x 应该已经位于正确的设备上

        # Transformer K步预测
        # x shape: (batch, seq_len, input_dim)
        x_transformed = self.input_fc(x)  # -> (batch, seq_len, model_dim)
        x_transformed = self.pos_encoder(x_transformed)  # -> (batch, seq_len, model_dim)
        transformer_out = self.transformer_encoder(x_transformed)  # -> (batch, seq_len, model_dim)

        # 我们取Transformer最后一个时间步的输出来进行预测
        # transformer_out[:, -1, :] shape is (batch, model_dim)
        y_nn_multistep = self.fc(transformer_out[:, -1, :]) # -> (batch, pred_horizon)
        # 为了与后续代码兼容，我们仍称之为y_lstm_multistep，尽管它现在来自Transformer
        y_lstm_multistep = y_nn_multistep


        # IDM迭代K步预测 (这部分逻辑与原LSTM模型中的IDM部分相同)
        y_idm_multistep_list = []
        # x, s_safe_initial, v_lead_initial 应该已经位于目标设备上
        v_ego_current_idm = x[:, -1, 0].clone() # 自车当前速度从输入x的最后一个时间步获取 (m/s)
        s_current_idm = s_safe_initial.clone() # 当前间距 (m)
        v_lead_constant_idm = v_lead_initial.clone() # 前车速度，假设在预测K步内不变 (m/s)

        for _ in range(self.pred_horizon):
            delta_v_idm = v_lead_constant_idm - v_ego_current_idm # 速度差
            # 使用idm_fixed函数进行单步IDM预测
            v_ego_next_pred_idm = idm_fixed(
                v_ego_current_idm, s_current_idm, delta_v_idm,
                v_desired=self.v_desired_idm, T=self.T_idm, a_max=self.a_max_idm,
                b_safe=self.b_safe_idm, delta=self.delta_idm, s0=self.s0_idm, delta_t=DT
            )
            y_idm_multistep_list.append(v_ego_next_pred_idm.unsqueeze(1)) # 收集预测结果

            # 更新IDM迭代的下一状态
            # 更新间距：s_new = s_old + (v_lead - v_ego_old) * dt
            # 这里使用 v_ego_current_idm (即当前时间步开始时的自车速度) 来计算位移差
            s_current_idm = (s_current_idm + (v_lead_constant_idm - v_ego_current_idm) * DT).clamp(min=1e-6)
            v_ego_current_idm = v_ego_next_pred_idm # 更新自车速度为预测的下一时刻速度

        y_idm_multistep = torch.cat(y_idm_multistep_list, dim=1) # 将K步预测拼接起来
        # Alpha不再是模型的输出
        return y_lstm_multistep, y_idm_multistep


def initialize_weights(model):
    """ 初始化模型权重 """
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1: # 通常是线性层和卷积层的权重
            nn.init.xavier_uniform_(param) # 使用Xavier均匀初始化
        elif "bias" in name: # 偏置项
            nn.init.constant_(param, 0) # 初始化为0


# --- 训练函数 (修改版，alpha固定，添加device参数) ---
def train_model(model, train_loader, pred_horizon, device, num_epochs=30, alpha_decay_loss=0.1, lr_nn=5e-4): # lr_lstm -> lr_nn
    model.train() # 设置模型为训练模式
    # 计算损失权重，对预测序列中较早的步骤给予更高权重
    # loss_weights 会根据模型参数的设备自动调整
    loss_weights_device = next(model.parameters()).device
    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(pred_horizon, dtype=torch.float32)).to(loss_weights_device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * pred_horizon # 归一化并调整尺度

    # 优化器只针对神经网络部分的参数（现在是Transformer）
    nn_params = [param for name, param in model.named_parameters()]
    optimizer_nn = optim.Adam(nn_params, lr=lr_nn) # 使用Adam优化器

    alpha_fixed = 0.7 # Alpha固定为0.7

    print(f"--- 使用固定 Alpha 开始训练 (设备: {device}) ---")
    print(f"神经网络参数将根据 L_nn = alpha * L_true + (1-alpha) * L_idm 进行优化。") # nn 指代神经网络部分
    print(f"Alpha 固定为: {alpha_fixed}")
    print(f"------------------------------------")

    for epoch in range(num_epochs):
        epoch_loss_nn_objective = 0.0 # 当前epoch的神经网络部分的目标损失

        for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in train_loader:
            # 显式将批处理数据移动到目标设备
            batch_x = batch_x.to(device)
            batch_y_multistep = batch_y_multistep.to(device)
            batch_s_safe_initial = batch_s_safe_initial.to(device)
            batch_v_lead_initial = batch_v_lead_initial.to(device)

            optimizer_nn.zero_grad() # 清空梯度

            # 前向传播 - 模型返回两个输出：神经网络预测和IDM预测
            y_nn_multistep, y_idm_multistep = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)

            # 计算各项损失分量
            # 损失1: 神经网络预测与真实值之间的差异
            loss_nn_vs_true = ((y_nn_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            # 损失2: 神经网络预测与IDM预测之间的差异 (IDM预测在此处不参与梯度计算 .detach())
            loss_nn_vs_idm = ((y_nn_multistep - y_idm_multistep.detach()).pow(2) * loss_weights.unsqueeze(0)).mean()

            # --- 更新神经网络参数 ---
            # 使用固定的alpha组合损失来更新神经网络参数
            loss_for_nn_params = alpha_fixed * loss_nn_vs_true + \
                                   (1 - alpha_fixed) * loss_nn_vs_idm

            loss_for_nn_params.backward() # 反向传播计算梯度
            optimizer_nn.step() # 更新神经网络参数

            epoch_loss_nn_objective += loss_for_nn_params.item() # 累积损失

        avg_nn_loss = epoch_loss_nn_objective / len(train_loader) # 计算平均损失
        print(f"Epoch {epoch + 1}/{num_epochs}  神经网络目标损失: {avg_nn_loss:.6f}  (α={alpha_fixed})")
    return model


# --- 测试/评估函数 (修改版，alpha固定，添加device参数) ---
def evaluate_model(model, test_loader, pred_horizon, device, alpha_decay_loss=0.1, dataset_name="", results_dir=""):
    model.eval() # 设置模型为评估模式
    all_pred_nn, all_pred_idm, all_true = [], [], [] # 存储所有预测和真实值, nn 指代神经网络

    # 损失权重，与训练时相同
    loss_weights_device = next(model.parameters()).device
    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(pred_horizon, dtype=torch.float32)).to(loss_weights_device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * pred_horizon

    total_mse_nn_vs_true_weighted = 0 # 神经网络预测与真实值之间的加权MSE总和
    total_mse_idm_vs_true_weighted = 0 # IDM预测与真实值之间的加权MSE总和

    with torch.no_grad(): # 评估时不需要计算梯度
        for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in test_loader:
            # 显式将批处理数据移动到目标设备
            batch_x = batch_x.to(device)
            batch_y_multistep = batch_y_multistep.to(device)
            batch_s_safe_initial = batch_s_safe_initial.to(device)
            batch_v_lead_initial = batch_v_lead_initial.to(device)

            # 模型返回两个输出
            y_nn_multistep, y_idm_multistep = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)

            all_pred_nn.append(y_nn_multistep.cpu()) # 收集神经网络预测 (转到CPU存储)
            all_pred_idm.append(y_idm_multistep.cpu()) # 收集IDM预测 (转到CPU存储)
            all_true.append(batch_y_multistep.cpu()) # 收集真实值 (转到CPU存储)

            # 计算当前批次的加权MSE (神经网络 vs 真实)
            loss_nn_vs_true_batch_weighted = (
                        (y_nn_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_nn_vs_true_weighted += loss_nn_vs_true_batch_weighted.item() * batch_x.size(0)

            # 计算当前批次的加权MSE (IDM vs 真实)
            loss_idm_vs_true_batch_weighted = (
                        (y_idm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_idm_vs_true_weighted += loss_idm_vs_true_batch_weighted.item() * batch_x.size(0)

    num_samples = len(test_loader.dataset) # 测试样本总数
    avg_mse_nn_vs_true_weighted = total_mse_nn_vs_true_weighted / num_samples # 平均加权MSE (神经网络 vs 真实)
    avg_mse_idm_vs_true_weighted = total_mse_idm_vs_true_weighted / num_samples # 平均加权MSE (IDM vs 真实)

    fixed_alpha_for_metrics = 0.7  # 使用的固定 Alpha 值

    y_pred_nn_cat = torch.cat(all_pred_nn) # 拼接所有神经网络预测
    y_pred_idm_cat = torch.cat(all_pred_idm) # 拼接所有IDM预测
    y_true_cat = torch.cat(all_true) # 拼接所有真实值

    # 最终的预测结果直接使用神经网络的输出
    y_final_prediction_cat = y_pred_nn_cat

    # 计算总体评估指标 (所有预测步骤的简单平均)
    mse_val_overall = torch.mean((y_final_prediction_cat - y_true_cat).pow(2)).item()
    rmse_val_overall = np.sqrt(mse_val_overall)
    mae_val_overall = torch.mean(torch.abs(y_final_prediction_cat - y_true_cat)).item()

    # 计算 MAPE (%)
    abs_error_overall = torch.abs(y_final_prediction_cat - y_true_cat)
    abs_true_overall = torch.abs(y_true_cat)
    valid_mape_mask_overall = abs_true_overall > 1e-6 # 避免除以零或极小值
    mape_p_overall = float('nan')
    if torch.sum(valid_mape_mask_overall) > 0:
        mape_p_overall = torch.mean(
            abs_error_overall[valid_mape_mask_overall] / abs_true_overall[valid_mape_mask_overall]
        ).item() * 100

    print(
        f"\n--- 测试结果总结 (最终使用神经网络预测，指标为所有步骤的简单平均) ---")
    print(
        f"  神经网络预测 vs 真实 -- MSE: {mse_val_overall:.4f}, RMSE: {rmse_val_overall:.4f}, MAE: {mae_val_overall:.4f}, MAPE: {mape_p_overall if not np.isnan(mape_p_overall) else 'N/A'}%")
    print(f"  (参考: IDM预测 vs 真实 加权MSE: {avg_mse_idm_vs_true_weighted:.4f})")
    print(
        f"  (参考: 神经网络预测 vs 真实 加权MSE (训练指标): {avg_mse_nn_vs_true_weighted:.4f})")
    print(f"  Alpha 使用值 (固定)={fixed_alpha_for_metrics:.4f}")

    print(f"\n--- 各预测步骤的详细指标 (最终使用神经网络速度预测) ---")
    for k_step in range(pred_horizon):
        y_pred_nn_step_k = y_pred_nn_cat[:, k_step] # 第k步的神经网络预测
        y_pred_idm_step_k = y_pred_idm_cat[:, k_step] # 第k步的IDM预测
        y_true_step_k = y_true_cat[:, k_step] # 第k步的真实值

        # 神经网络预测的指标
        mse_step_nn = nn.MSELoss()(y_pred_nn_step_k, y_true_step_k).item()
        rmse_step_nn = np.sqrt(mse_step_nn)
        mae_step_nn = torch.mean(torch.abs(y_pred_nn_step_k - y_true_step_k)).item()

        abs_error_step = torch.abs(y_pred_nn_step_k - y_true_step_k)
        abs_true_step = torch.abs(y_true_step_k)
        valid_mape_mask_step = abs_true_step > 1e-6
        mape_step_nn = float('nan')
        if torch.sum(valid_mape_mask_step) > 0:
            mape_step_nn = torch.mean(
                abs_error_step[valid_mape_mask_step] / abs_true_step[valid_mape_mask_step]
            ).item() * 100

        # IDM预测的MSE (作为参考)
        mse_step_idm = nn.MSELoss()(y_pred_idm_step_k, y_true_step_k).item()

        print(f"  步骤 {k_step + 1}:")
        print(
            f"    神经网络预测 -- MSE: {mse_step_nn:.4f}, RMSE: {rmse_step_nn:.4f}, MAE: {mae_step_nn:.4f}, MAPE: {mape_step_nn if not np.isnan(mape_step_nn) else 'N/A'}%")
        print(f"    IDM (参考) -- MSE: {mse_step_idm:.4f}")

    # 绘制第一个预测步骤 (k_plot=0) 的部分样本预测对比图
    k_plot = 0
    plt.figure(figsize=(12, 7))
    plt.plot(y_true_cat[:100, k_plot].numpy(), '--o', label=f'真实值 (步骤 {k_plot + 1})')
    plt.plot(y_pred_nn_cat[:100, k_plot].numpy(), '-x',
             label=f'神经网络预测 (步骤 {k_plot + 1}) (最终使用)')
    plt.plot(y_pred_idm_cat[:100, k_plot].numpy(), '-s', label=f'IDM预测 (步骤 {k_plot + 1}) (参考)')

    # 绘制一个假设的融合结果作为图形参考
    y_pred_combined_for_plot_cat = fixed_alpha_for_metrics * y_pred_nn_cat + (
                1 - fixed_alpha_for_metrics) * y_pred_idm_cat
    plt.plot(y_pred_combined_for_plot_cat[:100, k_plot].numpy(), '-.',
             label=f'假设融合 (步骤 {k_plot + 1}, α={fixed_alpha_for_metrics:.2f}) (图形参考)')

    plt.title(f'速度预测对比 (前100样本, 步骤 {k_plot + 1}) ({dataset_name})')
    plt.xlabel("样本索引")
    plt.ylabel("速度 (m/s)")
    plt.legend()
    plt.grid()
    plot_filename = os.path.join(results_dir, f"{dataset_name}_speed_comparison_PITANSFORMER_IDM_final_fixed_alpha.png") #文件名中NN代表神经网络
    plt.savefig(plot_filename)
    print(f"速度对比图已保存至 {plot_filename}")
    plt.close()

    # 返回基于加权MSE计算的RMSE（与训练目标一致）以及整体MAE和MAPE
    return mse_val_overall, rmse_val_overall, mae_val_overall, mape_p_overall


# === 修改后的 compute_position_and_spacing_and_save (使用神经网络直接输出，添加device参数) ===
def compute_position_and_spacing_and_save(model,
                                          test_loader,
                                          raw_data_all, # 原始数据集的完整部分 (用于获取初始状态)
                                          label_data_all, # 标签数据集的完整部分 (用于获取真实未来位置/间距)
                                          train_size, # 训练集大小，用于定位测试集在原始数据中的起始点
                                          pred_horizon, # 预测步长K
                                          device, # 添加device参数
                                          dt=0.1, # 时间步长
                                          output_file="predictions_multistep_extended.xlsx", # Excel输出文件名
                                          dataset_name=""): # 当前数据集名称，用于Excel的sheet名
    model.eval() # 设置模型为评估模式

    test_start_idx_in_all_data = train_size # 测试数据在完整原始数据中的起始索引

    # 用于存储各个批次的预测速度和真实速度 (单位: m/s)
    y_nn_list_mps, y_true_speeds_list_mps = [], []
    # 用于存储从原始数据中提取的初始状态 (单位: ft 或 ft/s, 根据原始数据格式)
    initial_ego_pos_ft_collected = []
    initial_lead_pos_ft_collected = []
    initial_ego_speed_ftps_collected = []
    initial_lead_speed_ftps_collected = []
    # 用于存储从标签数据中提取的真实未来位置和间距 (单位: ft)
    true_future_ego_pos_ft_collected = []
    true_future_spacing_ft_collected = []

    with torch.no_grad(): # 评估时不需要梯度
        for i, (batch_x_mps, batch_y_multistep_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps) in enumerate(
                test_loader):
            # 显式将模型输入数据移动到目标设备
            batch_x_mps = batch_x_mps.to(device)
            batch_s_safe_initial_m = batch_s_safe_initial_m.to(device)
            batch_v_lead_initial_mps = batch_v_lead_initial_mps.to(device)
            # batch_y_multistep_mps 主要用于CPU端收集，无需移动到device给模型

            # 模型返回神经网络预测和IDM预测，我们只需要神经网络的预测 y_nn_k_mps
            y_nn_k_mps, _ = model(batch_x_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps)

            y_nn_list_mps.append(y_nn_k_mps.cpu()) # 收集神经网络预测的速度 (转到CPU)
            y_true_speeds_list_mps.append(batch_y_multistep_mps.cpu()) # 收集真实速度 (已经在CPU或转到CPU)

            # 计算当前批次在完整数据集中的索引范围
            batch_start_idx_in_loader = i * test_loader.batch_size
            current_batch_indices_in_all_data = np.arange(
                test_start_idx_in_all_data + batch_start_idx_in_loader,
                test_start_idx_in_all_data + batch_start_idx_in_loader + batch_x_mps.size(0)
            )

            # 从 raw_data_all (英尺单位) 中提取初始状态
            # raw_data_all[:, -1, col_idx] 表示取每个样本序列的最后一个时间步的特定特征
            # 假设列索引: 4=自车位置Y, 7=前车位置Y, 0=自车速度, 5=前车速度
            initial_ego_pos_ft_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 4].cpu())
            initial_lead_pos_ft_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 7].cpu())
            initial_ego_speed_ftps_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 0].cpu()) # 英尺/秒
            initial_lead_speed_ftps_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 5].cpu()) # 英尺/秒

            # 从 lab_data_all (英尺单位) 中提取真实的未来位置和间距
            # label_data_all[:, :pred_horizon, col_idx] 表示取未来K个时间步的特定标签
            # 假设列索引: 3=自车未来位置Y, 1=未来间距
            true_future_ego_pos_ft_collected.append(
                label_data_all[current_batch_indices_in_all_data, :pred_horizon, 3].cpu())
            true_future_spacing_ft_collected.append(
                label_data_all[current_batch_indices_in_all_data, :pred_horizon, 1].cpu())

    # 拼接所有批次的数据
    y_nn_all_mps = torch.cat(y_nn_list_mps, dim=0) # (num_test_samples, pred_horizon)
    y_true_speeds_all_mps = torch.cat(y_true_speeds_list_mps, dim=0) # (num_test_samples, pred_horizon)

    # 最终的预测速度直接使用神经网络的输出 (m/s)
    final_pred_speeds_mps = y_nn_all_mps

    # 拼接初始状态数据 (单位: ft 或 ft/s)
    initial_ego_pos_ft = torch.cat(initial_ego_pos_ft_collected, dim=0) # (num_test_samples,)
    initial_lead_pos_ft = torch.cat(initial_lead_pos_ft_collected, dim=0) # (num_test_samples,)
    initial_ego_speed_ftps = torch.cat(initial_ego_speed_ftps_collected, dim=0) # (num_test_samples,)
    initial_lead_speed_ftps = torch.cat(initial_lead_speed_ftps_collected, dim=0) # (num_test_samples,)

    # 拼接真实未来位置和间距数据 (单位: ft)
    true_future_ego_pos_ft = torch.cat(true_future_ego_pos_ft_collected, dim=0) # (num_test_samples, pred_horizon)
    true_future_spacing_ft = torch.cat(true_future_spacing_ft_collected, dim=0) # (num_test_samples, pred_horizon)

    # 初始化用于存储K步预测位置和间距的张量 (单位: ft)
    pred_ego_pos_k_steps_ft = torch.zeros_like(final_pred_speeds_mps) # (num_test_samples, pred_horizon)
    pred_lead_pos_k_steps_ft = torch.zeros_like(final_pred_speeds_mps)
    pred_spacing_k_steps_ft = torch.zeros_like(final_pred_speeds_mps)

    # 将预测速度从 m/s 转换为 ft/s 用于位置计算
    final_pred_speeds_ftps = final_pred_speeds_mps / 0.3048

    # 当前迭代的自车和前车位置 (ft)，从初始观测值开始
    current_ego_pos_ft = initial_ego_pos_ft.clone()
    current_lead_pos_ft = initial_lead_pos_ft.clone()
    # 假设前车在预测的K步内保持其初始观测速度不变 (ft/s)
    lead_speed_constant_ftps = initial_lead_speed_ftps # shape: (num_test_samples,)

    # 迭代计算未来K步的位置和间距
    for k in range(pred_horizon):
        # 自车在该时间步k开始时的速度 (ft/s)
        if k == 0:
            # 对于第一步的位移，使用 *初始观测* 的自车速度
            speed_ego_this_step_ftps = initial_ego_speed_ftps
        else:
            # 对于后续步骤，使用神经网络在前一个时间步 (k-1) 预测的速度
            # final_pred_speeds_ftps[:, k-1] 是指在第 k-1 个预测区间末端的速度，即第k个区间的初速度
            speed_ego_this_step_ftps = final_pred_speeds_ftps[:, k - 1]

        # 计算该时间步内的位移 (ft)
        disp_ego_ft = speed_ego_this_step_ftps * dt
        disp_lead_ft = lead_speed_constant_ftps * dt # 前车以恒定速度移动

        # 更新位置 (ft)
        current_ego_pos_ft += disp_ego_ft
        current_lead_pos_ft += disp_lead_ft

        # 存储第k步结束时的预测位置和间距
        pred_ego_pos_k_steps_ft[:, k] = current_ego_pos_ft
        pred_lead_pos_k_steps_ft[:, k] = current_lead_pos_ft
        pred_spacing_k_steps_ft[:, k] = current_lead_pos_ft - current_ego_pos_ft

    # 将预测和真实的位置/间距从英尺转换为米，用于评估和保存
    pred_ego_pos_m = pred_ego_pos_k_steps_ft.numpy() * 0.3048
    true_ego_pos_m = true_future_ego_pos_ft.numpy() * 0.3048
    pred_spacing_m = pred_spacing_k_steps_ft.numpy() * 0.3048
    true_spacing_m = true_future_spacing_ft.numpy() * 0.3048

    print(f"\n--- 基于神经网络直接速度预测的逐位置和间距误差评估 ---")
    for k_s in range(pred_horizon): # 遍历每个预测步骤
        # 位置误差
        pos_err_sq_step = (pred_ego_pos_m[:, k_s] - true_ego_pos_m[:, k_s]) ** 2
        rmse_Y_step = np.sqrt(np.mean(pos_err_sq_step))
        valid_true_Y_step_mask = np.abs(true_ego_pos_m[:, k_s]) > 1e-6 # 避免除以零
        mape_Y_step = float('nan')
        if np.sum(valid_true_Y_step_mask) > 0:
            mape_Y_step = np.mean(np.abs(
                (pred_ego_pos_m[valid_true_Y_step_mask, k_s] - true_ego_pos_m[valid_true_Y_step_mask, k_s]) /
                true_ego_pos_m[valid_true_Y_step_mask, k_s])) * 100

        # 间距误差
        spacing_err_sq_step = (pred_spacing_m[:, k_s] - true_spacing_m[:, k_s]) ** 2
        rmse_sp_step = np.sqrt(np.mean(spacing_err_sq_step))
        valid_true_sp_step_mask = np.abs(true_spacing_m[:, k_s]) > 1e-6
        mape_sp_step = float('nan')
        if np.sum(valid_true_sp_step_mask) > 0:
            mape_sp_step = np.mean(np.abs(
                (pred_spacing_m[valid_true_sp_step_mask, k_s] - true_spacing_m[valid_true_sp_step_mask, k_s]) /
                true_spacing_m[valid_true_sp_step_mask, k_s])) * 100

        print(f"  步骤 {k_s + 1}:")
        print(
            f"    位置误差 -- RMSE: {rmse_Y_step:.4f} m, MAPE: {mape_Y_step if not np.isnan(mape_Y_step) else 'N/A'}%")
        print(
            f"    间距误差 -- RMSE: {rmse_sp_step:.4f} m, MAPE: {mape_sp_step if not np.isnan(mape_sp_step) else 'N/A'}%")

    # 评估最后一个预测步骤 (K) 的位置误差，用于总结
    rmse_p_overall = np.sqrt(np.mean((pred_ego_pos_m - true_ego_pos_m) ** 2))  # RMSE over all prediction steps
    valid_true_Y_mask = np.abs(true_ego_pos_m) > 1e-6  # Mask for valid true values

    mape_p_overall = float('nan')
    if np.sum(valid_true_Y_mask) > 0:
        mape_p_overall = np.mean(np.abs(
            (pred_ego_pos_m[valid_true_Y_mask] - true_ego_pos_m[valid_true_Y_mask]) /
            true_ego_pos_m[valid_true_Y_mask])) * 100  # MAPE over all prediction steps

    print(f"\n--- 最后一步 (K={rmse_p_overall}) 位置误差 (基于神经网络直接预测, 用于总结) ---")
    print(
        f"  位置误差 -- RMSE: {rmse_p_overall:.4f} m, MAPE: {mape_p_overall if not np.isnan(mape_p_overall) else 'N/A'}%")

    # 准备数据写入Excel
    df_data = {}
    for k_idx in range(pred_horizon):
        df_data[f"神经网络预测速度 (m/s) 步骤 {k_idx + 1}"] = final_pred_speeds_mps[:, k_idx].numpy()
        df_data[f"真实速度 (m/s) 步骤 {k_idx + 1}"] = y_true_speeds_all_mps[:, k_idx].numpy()
        df_data[f"预测自车位置 Y (m) 步骤 {k_idx + 1}"] = pred_ego_pos_m[:, k_idx]
        df_data[f"真实自车位置 Y (m) 步骤 {k_idx + 1}"] = true_ego_pos_m[:, k_idx]
        df_data[f"预测间距 (m) 步骤 {k_idx + 1}"] = pred_spacing_m[:, k_idx]
        df_data[f"真实间距 (m) 步骤 {k_idx + 1}"] = true_spacing_m[:, k_idx]

    df_pos = pd.DataFrame(df_data)

    # 将结果写入Excel文件，如果文件已存在则替换同名sheet，否则创建新文件
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            df_pos.to_excel(writer, sheet_name=dataset_name, index=False)
    except FileNotFoundError: # 如果文件不存在，则以写入模式创建
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df_pos.to_excel(writer, sheet_name=dataset_name, index=False)

    print(f"{dataset_name} 的位置和间距预测 (基于神经网络) 已保存到 '{output_file}' 的 sheet '{dataset_name}'.")
    return rmse_p_overall, mape_p_overall


# --- 存储和保存评估指标的辅助函数 ---
all_datasets_metrics_summary = [] # 用于存储所有数据集的评估指标


def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse_last_step,
                          pos_mape_last_step):
    """存储单个数据集的评估指标"""
    metrics = {
        "数据集 (Dataset)": dataset_name,
        "速度MSE_NN (Speed_MSE_NN_summary)": speed_mse,  # 基于神经网络的加权MSE (训练目标)
        "速度RMSE_NN (Speed_RMSE_NN_summary)": speed_rmse, # 基于神经网络的加权RMSE (训练目标)
        "速度MAE_NN (Speed_MAE_NN_overall)": speed_mae,  # 基于神经网络的整体MAE (所有步骤简单平均)
        "速度MAPE_NN (%) (Speed_MAPE_NN_overall_percent)": speed_mape,  # 基于神经网络的整体MAPE (%)
        "末步位置RMSE_NN (m) (Position_RMSE_NN_last_step_m)": pos_rmse_last_step, # 最后一步的位置RMSE
        "末步位置MAPE_NN (%) (Position_MAPE_NN_last_step_percent)": pos_mape_last_step # 最后一步的位置MAPE (%)
    }
    all_datasets_metrics_summary.append(metrics)


def save_all_metrics_to_csv(filepath="evaluation_summary_PItransform_idm_final.csv"):  # 文件名修改以反映是NN
    """将所有数据集的评估指标汇总保存到CSV文件"""
    if not all_datasets_metrics_summary:
        print("没有评估指标可以保存。")
        return
    df_metrics = pd.DataFrame(all_datasets_metrics_summary)
    # 使用 utf-8-sig 编码以确保中文在Excel中正确显示
    df_metrics.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"所有数据集的评估指标汇总已保存至 {filepath}")


# --- 主流程 ---
if __name__ == "__main__":
    torch.manual_seed(42) # 设置随机种子以保证结果可复现
    # 自动选择可用设备 (GPU优先)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat")) # 查找所有.mat数据文件
    if not data_files:
        print(f"在目录 {DATA_DIR} 中未找到 .mat 文件。程序将退出。")
        exit()
    print(f"找到以下数据集文件: {data_files}")

    # 定义保存所有数据集位置预测的Excel文件名
    position_predictions_excel_path = os.path.join(RESULTS_DIR,
                                                   "pred_positions_all_datasets_pitransformer_idm_final_all_steps1128.xlsx")

    # 定义学习率
    LR_NN_PARAMS = 5e-4  # 神经网络部分的学习率

    for data_file_path in data_files: # 遍历每个数据集文件
        dataset_filename = os.path.basename(data_file_path) # 获取文件名
        dataset_name_clean = dataset_filename.replace(".mat", "") # 去掉.mat后缀作为数据集名称
        print(f"\n==================== 开始处理数据集: {dataset_filename} ====================")

        # 加载数据
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
        N = int(N_total * 0.2)
        print(f"将使用 {N} / {N_total} 条数据进行训练和测试。")

        seq_mps_selected = seq_mps[:N]
        y_multistep_mps_selected = y_multistep_mps[:N]
        s_safe_initial_m_selected = s_safe_initial_m[:N]
        v_lead_initial_mps_selected = v_lead_initial_mps[:N]
        raw_all_ft_selected = raw_all_ft[:N]
        lab_all_ft_selected = lab_all_ft[:N]

        split_ratio = 0.8
        train_size = int(N * split_ratio)

        # 将数据移动到目标设备 (GPU或CPU)
        train_seq = seq_mps_selected[:train_size].to(device)
        test_seq = seq_mps_selected[train_size:].to(device)
        train_y_multistep = y_multistep_mps_selected[:train_size].to(device)
        test_y_multistep = y_multistep_mps_selected[train_size:].to(device)
        train_s_safe_initial = s_safe_initial_m_selected[:train_size].to(device)
        test_s_safe_initial = s_safe_initial_m_selected[train_size:].to(device)
        train_v_lead_initial = v_lead_initial_mps_selected[:train_size].to(device)
        test_v_lead_initial = v_lead_initial_mps_selected[train_size:].to(device)

        batch_size = 32
        train_ds = torch.utils.data.TensorDataset(train_seq, train_y_multistep, train_s_safe_initial,
                                                  train_v_lead_initial)
        test_ds = torch.utils.data.TensorDataset(test_seq, test_y_multistep, test_s_safe_initial, test_v_lead_initial)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        input_dim = train_seq.size(2)
        hidden_dim = 128
        n_head = 4
        transformer_layers = 2
        feedforward_dim = 512
        transformer_dropout = 0.1

        model = HybridIDMTransformerModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=current_pred_horizon,
            nhead=n_head,
            transformer_num_layers=transformer_layers,
            dim_feedforward=feedforward_dim,
            dropout_transformer=transformer_dropout
        ).to(device) # 将模型移动到目标设备

        initialize_weights(model)

        print(f"开始训练模型: {dataset_name_clean}...")
        # 传递device参数给训练函数
        model = train_model(model, train_loader, pred_horizon=current_pred_horizon, device=device, num_epochs=50,
                            alpha_decay_loss=0.05, lr_nn=LR_NN_PARAMS)

        print(f"开始评估模型 (速度预测, 最终使用神经网络输出): {dataset_name_clean}...")
        # 传递device参数给评估函数
        speed_mse_summary, speed_rmse_summary, speed_mae_overall, speed_mape_overall = evaluate_model(
            model, test_loader, pred_horizon=current_pred_horizon, device=device, alpha_decay_loss=0.05,
            dataset_name=dataset_name_clean, results_dir=RESULTS_DIR
        )

        print(f"开始计算和评估位置/间距预测 (基于神经网络直接输出): {dataset_name_clean}...")
        # 传递device参数给位置计算函数
        pos_rmse_last_step, pos_mape_last_step = compute_position_and_spacing_and_save(
            model, test_loader, raw_all_ft_selected, lab_all_ft_selected, train_size,
            pred_horizon=current_pred_horizon, device=device, dt=DT,
            output_file=position_predictions_excel_path, dataset_name=dataset_name_clean
        )

        store_dataset_metrics(dataset_name_clean, speed_mse_summary, speed_rmse_summary, speed_mae_overall,
                              speed_mape_overall, pos_rmse_last_step, pos_mape_last_step)
        print(f"==================== 数据集 {dataset_filename} 处理完毕 ====================")

    summary_metrics_csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_all_datasets_NN_final1128.csv")
    save_all_metrics_to_csv(summary_metrics_csv_path)
    print("\n所有数据集处理完毕。最终评估汇总已保存。")

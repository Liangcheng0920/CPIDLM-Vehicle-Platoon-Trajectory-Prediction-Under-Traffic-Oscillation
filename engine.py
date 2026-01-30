import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.utils import device  # 导入全局设备


# =========================
# 辅助函数：IDM训练的前车轨迹预计算
# =========================
def precompute_leader_trajectories_for_idm_training(
        leader_model, raw_data_slice, pred_steps_K, dt, device, hist_len=50
):
    leader_model.eval()
    num_samples = raw_data_slice.shape[0]

    if num_samples == 0:
        _IDM_INPUT_DIM_placeholder = 5
        empty = torch.empty(0, pred_steps_K, dtype=torch.float32, device=device)
        empty_in = torch.empty(0, hist_len, _IDM_INPUT_DIM_placeholder, dtype=torch.float32, device=device)
        empty_scal = torch.empty(0, dtype=torch.float32, device=device)
        return empty_in, empty_scal, empty_scal, empty, empty, empty_scal

    initial_idm_input_seqs = raw_data_slice[:, -hist_len:, [0, 1, 2, 3, 5]].clone() * 0.3048
    initial_follower_poses = raw_data_slice[:, -1, 4].clone() * 0.3048
    initial_leader_poses_val = raw_data_slice[:, -1, -1].clone() * 0.3048
    initial_s_safes = initial_idm_input_seqs[:, -1, 1].clone()
    batch_d1 = initial_leader_poses_val - initial_follower_poses - initial_s_safes

    leader_hist_for_lnn = raw_data_slice[:, -hist_len:, [5, 6]].clone() * 0.3048

    pred_leader_speeds_K_list = []
    pred_leader_pos_K_list = []
    current_dt = dt if dt > 1e-6 else 1e-6

    with torch.no_grad():
        all_pred_l_speeds_k_steps_tensor = leader_model.predict_speed(leader_hist_for_lnn.to(device)).cpu()

        for i in range(num_samples):
            pred_l_speeds_k_steps_tensor_i = all_pred_l_speeds_k_steps_tensor[i]
            pred_leader_speeds_K_list.append(pred_l_speeds_k_steps_tensor_i)

            current_l_pos = initial_leader_poses_val[i].item()
            prev_l_v = leader_hist_for_lnn[i, -1, 0].item()
            l_pos_k_steps = []

            for k_idx in range(pred_steps_K):
                vp = pred_l_speeds_k_steps_tensor_i[k_idx].item()
                a_leader = (vp - prev_l_v) / current_dt
                displacement_leader = prev_l_v * current_dt + 0.5 * a_leader * current_dt * current_dt
                next_l_pos = current_l_pos + displacement_leader
                l_pos_k_steps.append(next_l_pos)
                prev_l_v = vp
                current_l_pos = next_l_pos
            pred_leader_pos_K_list.append(torch.tensor(l_pos_k_steps, dtype=torch.float32))

    pred_leader_speeds_K = torch.stack(pred_leader_speeds_K_list) if num_samples > 0 else torch.empty(0, pred_steps_K)
    pred_leader_pos_K = torch.stack(pred_leader_pos_K_list) if num_samples > 0 else torch.empty(0, pred_steps_K)

    return initial_idm_input_seqs.to(device), initial_follower_poses.to(device), initial_s_safes.to(device), \
        pred_leader_speeds_K.to(device), pred_leader_pos_K.to(device), batch_d1.to(device)


# =========================
# 辅助函数：IDM 多步预测 (用于融合训练)
# =========================
def predict_multi_step_idm_for_fusion_training(
        idm_model, leader_model, initial_idm_input_seq_batch, raw_data_slice_batch,
        pred_steps_K, dt, hist_len, device_compute
):
    idm_model.eval()
    leader_model.eval()

    (_, initial_f_pos_batch, initial_s_safe_batch,
     pred_l_speeds_K_batch, pred_l_pos_K_batch, d1_offset_batch) = \
        precompute_leader_trajectories_for_idm_training(
            leader_model, raw_data_slice_batch.to(device_compute),
            pred_steps_K, dt, device_compute, hist_len
        )

    initial_idm_input_seq_batch = initial_idm_input_seq_batch.to(device_compute)

    batch_current_idm_input_torch = initial_idm_input_seq_batch.clone()
    batch_current_follower_speed_pred = batch_current_idm_input_torch[:, -1, 0].clone()
    batch_current_follower_pos = initial_f_pos_batch.clone()
    # batch_current_s_actual_for_idm = initial_s_safe_batch.clone() # 逻辑中已被 initial_s_safe_batch 替代

    all_predicted_follower_speeds_batch_list = []
    current_dt_val = dt if dt > 1e-6 else 1e-6

    with torch.no_grad():
        for k_step in range(pred_steps_K):
            # 获取当前实际间距 (batch_current_idm_input_torch 的最后一帧是当前状态)
            # 注意：在原始代码中这里使用的是 batch_current_s_actual_for_idm 变量进行维护
            # 为了保持一致性，我们在这里恢复那个变量的维护逻辑
            if k_step == 0:
                current_s_actual = initial_s_safe_batch.clone()

            if torch.isnan(batch_current_idm_input_torch).any() or torch.isnan(current_s_actual).any():
                v_follower_pred = batch_current_follower_speed_pred.clone()
                if torch.isnan(v_follower_pred).any(): v_follower_pred[:] = 0.0
            else:
                v_follower_pred_unsqueeze, _ = idm_model.predict_speed(
                    batch_current_idm_input_torch, current_s_actual
                )
                v_follower_pred = v_follower_pred_unsqueeze.squeeze(1)

            if torch.isnan(v_follower_pred).any() or torch.isinf(v_follower_pred).any():
                nan_inf_mask = torch.isnan(v_follower_pred) | torch.isinf(v_follower_pred)
                v_follower_pred[nan_inf_mask] = batch_current_follower_speed_pred[nan_inf_mask]
                if torch.isnan(v_follower_pred).any(): v_follower_pred[torch.isnan(v_follower_pred)] = 0.0

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

                if not torch.isnan(new_feature_slice).any():
                    batch_current_idm_input_torch = torch.cat(
                        [batch_current_idm_input_torch[:, 1:, :], new_feature_slice.unsqueeze(1)], dim=1
                    )

                batch_current_follower_speed_pred = v_follower_pred.clone()
                batch_current_follower_pos = pos_follower_next.clone()
                current_s_actual = spacing_adjusted_next.clone()  # 更新 s_actual

    return torch.cat(all_predicted_follower_speeds_batch_list, dim=1)


# =========================
# 通用训练函数
# =========================
def train_generic_model(model, loader, optimizer, criterion, epochs=30, model_name="Generic Model", clip_value=1.0,
                        is_fusion_model=False, idm_model_frozen=None, lnn_ego_model_frozen=None,
                        leader_model_frozen=None, FUSION_HIST_LEN=None,
                        PRED_STEPS_K_fusion=None, DT_fusion=None, HIST_LEN_idm_ego_fusion=None, device_fusion=None):
    model.train()
    if is_fusion_model:
        if idm_model_frozen: idm_model_frozen.eval()
        if lnn_ego_model_frozen: lnn_ego_model_frozen.eval()
        if leader_model_frozen: leader_model_frozen.eval()

    for ep in range(epochs):
        tot_loss = 0
        num_batches_processed = 0

        if is_fusion_model:
            for batch_idx, (raw_batch, label_batch) in enumerate(loader):
                raw_batch, label_batch = raw_batch.to(device_fusion), label_batch.to(device_fusion)
                optimizer.zero_grad()

                idm_input_hist_batch = raw_batch[:, -HIST_LEN_idm_ego_fusion:, [0, 1, 2, 3, 5]].clone() * 0.3048
                with torch.no_grad():
                    y_lstm_idm_pred_batch = predict_multi_step_idm_for_fusion_training(
                        idm_model_frozen, leader_model_frozen,
                        idm_input_hist_batch, raw_batch,
                        PRED_STEPS_K_fusion, DT_fusion, HIST_LEN_idm_ego_fusion, device_fusion
                    )

                lnn_ego_input_batch = raw_batch[:, -HIST_LEN_idm_ego_fusion:, [0, 1, 2, 3, 5]].clone() * 0.3048
                with torch.no_grad():
                    y_lnn_ego_pred_batch = lnn_ego_model_frozen.predict_speed(lnn_ego_input_batch)

                ego_speed_hist_fusion = raw_batch[:, -FUSION_HIST_LEN:, 0].unsqueeze(-1).clone() * 0.3048
                leader_speed_hist_fusion = raw_batch[:, -FUSION_HIST_LEN:, 5].unsqueeze(-1).clone() * 0.3048

                alpha_batch = model(ego_speed_hist_fusion, leader_speed_hist_fusion)

                y_fusion_batch = alpha_batch * y_lstm_idm_pred_batch.detach() + \
                                 (1 - alpha_batch) * y_lnn_ego_pred_batch.detach()

                true_follower_speeds_K_batch = label_batch[:, :PRED_STEPS_K_fusion, 0].clone() * 0.3048
                loss = criterion(y_fusion_batch, true_follower_speeds_K_batch)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                tot_loss += loss.item()
                num_batches_processed += 1
        else:
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model.predict_speed(x_batch) if hasattr(model, 'predict_speed') else model(x_batch)
                loss = criterion(pred, y_batch)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                tot_loss += loss.item()
                num_batches_processed += 1

        avg_loss = tot_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
        print(f"[{model_name}] Epoch {ep + 1}/{epochs}, 平均损失: {avg_loss:.4f}")
        if np.isnan(avg_loss) and ep > 0:
            print(f"警告: {model_name} 平均损失为 NaN，训练提前停止。")
            break
    return model


# =========================
# IDM 多步训练函数
# =========================
def train_idm_model_multistep(
        model, train_loader, optimizer,
        num_epochs=30, pred_steps_K=5, dt=0.1, alpha_decay=0.0,
        teacher_forcing_initial_ratio=1.0, min_teacher_forcing_ratio=0.0,
        teacher_forcing_decay_epochs_ratio=0.75, clip_value=1.0
):
    model.train()
    criterion_mse_elementwise = nn.MSELoss(reduction='none')
    loss_weights = torch.exp(-alpha_decay * torch.arange(pred_steps_K, device=device).float())
    decay_epochs = int(num_epochs * teacher_forcing_decay_epochs_ratio)
    current_dt = dt if dt > 1e-6 else 1e-6

    for epoch in range(num_epochs):
        total_loss_epoch = 0
        num_valid_batches = 0

        current_teacher_forcing_ratio = teacher_forcing_initial_ratio - \
                                        (teacher_forcing_initial_ratio - min_teacher_forcing_ratio) * \
                                        (float(epoch) / decay_epochs if decay_epochs > 0 else 0)
        current_teacher_forcing_ratio = max(min_teacher_forcing_ratio, current_teacher_forcing_ratio)

        print(f"[LSTM-IDM 多步训练] Epoch [{epoch + 1}/{num_epochs}], TF Ratio: {current_teacher_forcing_ratio:.2f}")

        for batch_idx, (batch_initial_idm_input_seq, batch_true_follower_speeds_K,
                        batch_initial_follower_pos, batch_initial_s_safe,
                        batch_pred_leader_speeds_K, batch_pred_leader_pos_K, batch_d1_offset,
                        batch_true_follower_all_features_K, batch_true_follower_pos_K) in enumerate(train_loader):

            batch_true_follower_speeds_K = batch_true_follower_speeds_K.to(device)
            # ... (其他数据也确保在device上，如果dataloader没做的话) ...

            optimizer.zero_grad()
            batch_current_idm_input_torch = batch_initial_idm_input_seq.clone()
            batch_current_follower_speed_pred = batch_current_idm_input_torch[:, -1, 0].clone()
            batch_current_follower_pos = batch_initial_follower_pos.clone()
            batch_current_s_actual_for_idm = batch_initial_s_safe.clone()
            all_predicted_batch_list = []
            skip_batch = False

            for k_step in range(pred_steps_K):
                if torch.isnan(batch_current_idm_input_torch).any(): skip_batch = True; break

                v_pred_unsqueeze, _ = model.predict_speed(batch_current_idm_input_torch, batch_current_s_actual_for_idm)
                v_pred = v_pred_unsqueeze.squeeze(1)

                if torch.isnan(v_pred).any(): skip_batch = True; break
                all_predicted_batch_list.append(v_pred.unsqueeze(1))

                if k_step < pred_steps_K - 1:
                    use_ground_truth = torch.rand(1).item() < current_teacher_forcing_ratio
                    v_leader_next = batch_pred_leader_speeds_K[:, k_step]
                    pos_leader_next = batch_pred_leader_pos_K[:, k_step]

                    if use_ground_truth:
                        # Teacher Forcing
                        v_f_next_true = batch_true_follower_all_features_K[:, k_step, 0]
                        s_next_true = batch_true_follower_all_features_K[:, k_step, 1]
                        dv_next_true = batch_true_follower_all_features_K[:, k_step, 2]
                        a_f_next_true = batch_true_follower_all_features_K[:, k_step, 3]
                        pos_f_next_true = batch_true_follower_pos_K[:, k_step]

                        new_feature_slice = torch.stack(
                            [v_f_next_true, s_next_true, dv_next_true, a_f_next_true, v_leader_next], dim=1)
                        batch_current_follower_speed_pred = v_f_next_true.clone()
                        batch_current_follower_pos = pos_f_next_true.clone()
                        batch_current_s_actual_for_idm = s_next_true.clone()
                    else:
                        # Student Forcing
                        a_f_next = (v_pred - batch_current_follower_speed_pred) / current_dt
                        a_f_next = torch.clamp(a_f_next, -10.0, 10.0)
                        pos_f_next = batch_current_follower_pos + batch_current_follower_speed_pred * current_dt + 0.5 * a_f_next * current_dt ** 2
                        s_next = torch.clamp(pos_leader_next - pos_f_next - batch_d1_offset, min=0.1)
                        dv_next = v_leader_next - v_pred

                        new_feature_slice = torch.stack([v_pred, s_next, dv_next, a_f_next, v_leader_next], dim=1)
                        batch_current_follower_speed_pred = v_pred.clone()
                        batch_current_follower_pos = pos_f_next.clone()
                        batch_current_s_actual_for_idm = s_next.clone()

                    batch_current_idm_input_torch = torch.cat(
                        [batch_current_idm_input_torch[:, 1:, :], new_feature_slice.unsqueeze(1)], dim=1)

            if skip_batch: optimizer.zero_grad(); continue

            batch_preds = torch.cat(all_predicted_batch_list, dim=1)
            loss = (criterion_mse_elementwise(batch_preds, batch_true_follower_speeds_K) * loss_weights.unsqueeze(
                0)).sum(dim=1).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                total_loss_epoch += loss.item()
                num_valid_batches += 1

        avg_loss = total_loss_epoch / num_valid_batches if num_valid_batches > 0 else float('nan')
        print(f"  平均损失: {avg_loss:.4f}")
        if np.isnan(avg_loss) and epoch > 0: break
    return model


# =========================
# 通用评估函数
# =========================
def evaluate_generic_model(model, test_loader, pred_steps=5, model_name="Generic Model", device_eval=None):
    model.eval()
    all_predicted, all_true = [], []

    if not test_loader: return

    with torch.no_grad():
        for batch_data, batch_target_speed in test_loader:
            batch_data, batch_target_speed = batch_data.to(device_eval), batch_target_speed.to(device_eval)
            predicted_speed = model.predict_speed(batch_data) if hasattr(model, 'predict_speed') else model(batch_data)
            all_predicted.append(predicted_speed.cpu())
            all_true.append(batch_target_speed.cpu())

    if not all_predicted: return
    all_predicted_cat = torch.cat(all_predicted, dim=0).numpy()
    all_true_cat = torch.cat(all_true, dim=0).numpy()

    mse_val = np.mean((all_predicted_cat - all_true_cat) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted_cat - all_true_cat))
    rmse_per_step = np.sqrt(np.mean((all_predicted_cat - all_true_cat) ** 2, axis=0))

    print(f"\n{model_name} 评估 (RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f})")
    for i in range(min(pred_steps, len(rmse_per_step))):
        print(f"  Step {i + 1}: RMSE {rmse_per_step[i]:.4f}")

    # 绘图逻辑
    plot_results(all_true_cat, all_predicted_cat, pred_steps, model_name)


def plot_results(true_data, pred_data, pred_steps, title_suffix):
    num_plot_samples = min(30, true_data.shape[0])
    step_interval = max(1, true_data.shape[0] // num_plot_samples if num_plot_samples > 0 else 1)
    true_cat, pred_cat = [], []

    for i in range(0, true_data.shape[0], step_interval):
        if len(true_cat) / pred_steps >= num_plot_samples: break
        true_cat.extend(true_data[i])
        pred_cat.extend(pred_data[i])

    if true_cat:
        plt.figure(figsize=(10, 5))
        plt.plot(true_cat, '--', label='Ground Truth')
        plt.plot(pred_cat, '-', label='Prediction')
        plt.title(f'{title_suffix} Predictions')
        plt.legend()
        plt.grid(True)


# =========================
# 融合模型最终评估
# =========================
def compute_multi_step_fusion_predictions(
        idm_model, leader_model_for_idm, lnn_ego_model, fusion_model,
        raw_data_test_slice, label_data_test_slice,
        dt, pred_steps, hist_len_idm_ego, hist_len_fusion, device_comp
):
    idm_model.eval();
    leader_model_for_idm.eval();
    lnn_ego_model.eval();
    fusion_model.eval()
    if raw_data_test_slice.shape[0] == 0: return

    idm_input = raw_data_test_slice[:, -hist_len_idm_ego:, [0, 1, 2, 3, 5]].clone() * 0.3048
    with torch.no_grad():
        y_idm = predict_multi_step_idm_for_fusion_training(
            idm_model, leader_model_for_idm, idm_input, raw_data_test_slice,
            pred_steps, dt, hist_len_idm_ego, device_comp
        ).to(device_comp)

    lnn_input = raw_data_test_slice[:, -hist_len_idm_ego:, [0, 1, 2, 3, 5]].clone() * 0.3048
    with torch.no_grad():
        y_lnn = lnn_ego_model.predict_speed(lnn_input.to(device_comp)).to(device_comp)

    ego_hist_fusion = raw_data_test_slice[:, -hist_len_fusion:, 0].unsqueeze(-1).clone() * 0.3048
    leader_hist_fusion = raw_data_test_slice[:, -hist_len_fusion:, 5].unsqueeze(-1).clone() * 0.3048

    with torch.no_grad():
        alpha = fusion_model(ego_hist_fusion.to(device_comp), leader_hist_fusion.to(device_comp)).to(device_comp)

    y_fusion = alpha * y_idm.detach() + (1 - alpha) * y_lnn.detach()
    y_fusion_np = y_fusion.cpu().numpy()
    true_np = label_data_test_slice[:, :pred_steps, 0].clone().cpu().numpy() * 0.3048

    mse = np.mean((y_fusion_np - true_np) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_fusion_np - true_np))

    print(f"\n最终融合模型预测 (RMSE: {rmse:.4f}, MAE: {mae:.4f})")
    plot_results(true_np, y_fusion_np, pred_steps, "Final Fusion Model")
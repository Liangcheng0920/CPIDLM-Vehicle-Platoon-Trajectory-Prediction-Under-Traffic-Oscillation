import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import glob
import os
import math

# Allow multiple OpenMP libraries to coexist to avoid conflicts in certain environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Global Path Definitions ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # Dataset storage directory
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim_modified_transformer_single_step"  # Results storage directory

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Global Constants ---
DT = 0.1  # Time step (s)
PRED_HORIZON = 1  # Prediction horizon K - Modified for single-step prediction


# --- Data Validation Function ---
def check_data(data, name="data"):
    """ Check if the data contains NaN or Inf values """
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# --- Fixed IDM Parameter Prediction Function ---
def idm_fixed(v_n, s_safe, delta_v,
              v_desired=10.13701546, T=0.50284384, a_max=0.10995557,
              b_safe=4.98369406, delta=5.35419582, s0=0.10337701,
              delta_t=DT):
    """
    Perform a single-step IDM (Intelligent Driver Model) prediction using fixed parameters.
    :param v_n: Current ego speed (m/s)
    :param s_safe: Current actual gap (m)
    :param delta_v: Current speed difference (Lead speed - Ego speed, m/s)
    :param v_desired: Desired speed (m/s)
    :param T: Safe time headway (s)
    :param a_max: Max acceleration (m/s^2)
    :param b_safe: Comfortable deceleration (m/s^2)
    :param delta: Acceleration exponent
    :param s0: Minimum static gap (m)
    :param delta_t: Time step (s)
    :return: Predicted ego speed for the next time step (m/s)
    """
    current_device = v_n.device
    # Convert IDM parameters to tensors matching the input device and type
    v_desired = torch.tensor(v_desired, device=current_device, dtype=v_n.dtype)
    T = torch.tensor(T, device=current_device, dtype=v_n.dtype)
    a_max = torch.tensor(a_max, device=current_device, dtype=v_n.dtype).clamp(min=1e-6)  # Avoid division by zero
    b_safe = torch.tensor(b_safe, device=current_device, dtype=v_n.dtype).clamp(min=1e-6)
    s0 = torch.tensor(s0, device=current_device, dtype=v_n.dtype)
    delta_param = torch.tensor(delta, device=current_device, dtype=v_n.dtype)
    delta_t_tensor = torch.tensor(delta_t, device=current_device, dtype=v_n.dtype)

    s_safe = s_safe.clamp(min=1e-6)  # Ensure gap is positive

    # Calculate desired gap s*
    s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe) + 1e-6)
    s_star = s_star.clamp(min=0.0)

    v_n_ratio = torch.zeros_like(v_n)
    mask_v_desired_nonzero = v_desired.abs() > 1e-6
    if mask_v_desired_nonzero.any():
        v_n_ratio[mask_v_desired_nonzero] = (v_n[mask_v_desired_nonzero] / v_desired[mask_v_desired_nonzero])

    # IDM acceleration formula
    acceleration_term = a_max * (
            1 - v_n_ratio ** delta_param - (s_star / s_safe) ** 2
    )
    v_follow = v_n + delta_t_tensor * acceleration_term
    return v_follow.clamp(min=0.0)


# --- Transformer Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):  # d_model is model dimension, max_len is max sequence length
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) -> (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Input tensor, shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- Define New Transformer-based Hybrid Model (Single-step) ---
class HybridIDMTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,  # Basic parameters
                 nhead=4, transformer_num_layers=2, dim_feedforward=512, dropout_transformer=0.1):  # Transformer specific parameters
        super(HybridIDMTransformerModel, self).__init__()
        self.pred_horizon = PRED_HORIZON  # Prediction horizon K, fixed at 1
        self.model_dim = hidden_dim  # Transformer internal dimension (d_model)

        self.input_fc = nn.Linear(input_dim, self.model_dim)
        self.pos_encoder = PositionalEncoding(self.model_dim, dropout_transformer)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_transformer,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=transformer_num_layers)
        # Output linear layer: map last time step output to 1 predicted value
        self.fc = nn.Linear(self.model_dim, self.pred_horizon)  # output_dim is 1

        # IDM parameters (fixed values)
        self.v_desired_idm = 12.64798288
        self.T_idm = 0.50284384
        self.a_max_idm = 0.10033688
        self.b_safe_idm = 4.98937183
        self.delta_idm = 1.0
        self.s0_idm = 0.13082412

    def forward(self, x, s_safe_initial, v_lead_initial):
        """
        Model forward pass (Single-step prediction).
        :param x: Input sequence, shape=(batch, seq_len, input_dim)
        :param s_safe_initial: Current safety gap (real observation for IDM), shape=(batch,)
        :param v_lead_initial: Current lead vehicle speed (real observation for IDM), shape=(batch,)
        :return:
          y_nn_pred: Transformer-based single-step speed prediction, shape=(batch, 1)
          y_idm_pred: IDM-based single-step speed prediction, shape=(batch, 1)
        """
        # Transformer Single-step prediction
        x_transformed = self.input_fc(x)
        x_transformed = self.pos_encoder(x_transformed)
        transformer_out = self.transformer_encoder(x_transformed)

        # Use the output of the last time step for prediction
        y_nn_pred = self.fc(transformer_out[:, -1, :])  # -> (batch, 1)

        # IDM Single-step prediction
        y_idm_pred_list = []
        v_ego_current_idm = x[:, -1, 0].clone()  # Current ego speed
        s_current_idm = s_safe_initial.clone()  # Current gap
        v_lead_constant_idm = v_lead_initial.clone()  # Lead vehicle speed

        # PRED_HORIZON = 1, so this loop runs once
        for _ in range(self.pred_horizon):
            delta_v_idm = v_lead_constant_idm - v_ego_current_idm
            v_ego_next_pred_idm = idm_fixed(
                v_ego_current_idm, s_current_idm, delta_v_idm,
                v_desired=self.v_desired_idm, T=self.T_idm, a_max=self.a_max_idm,
                b_safe=self.b_safe_idm, delta=self.delta_idm, s0=self.s0_idm, delta_t=DT
            )
            y_idm_pred_list.append(v_ego_next_pred_idm.unsqueeze(1))
            # For single-step, no need to update s_current_idm and v_ego_current_idm for next iteration

        y_idm_pred = torch.cat(y_idm_pred_list, dim=1)  # -> (batch, 1)

        return y_nn_pred, y_idm_pred


def initialize_weights(model):
    """ Initialize model weights """
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)


# --- Training Function (Single-step version) ---
def train_model(model, train_loader, device, num_epochs=30, alpha_decay_loss=0.1, lr_nn=5e-4):
    model.train()
    # For single-step PRED_HORIZON=1, loss_weights is essentially tensor([1.0])
    loss_weights_device = next(model.parameters()).device
    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(PRED_HORIZON, dtype=torch.float32)).to(
        loss_weights_device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * PRED_HORIZON

    nn_params = [param for name, param in model.named_parameters()]
    optimizer_nn = optim.Adam(nn_params, lr=lr_nn)

    alpha_fixed = 0.7  # Alpha fixed at 0.7

    print(f"--- Starting training with fixed Alpha (Single-step, Device: {device}) ---")
    print(f"NN parameters optimized via L_nn = alpha * L_true + (1-alpha) * L_idm.")
    print(f"Fixed Alpha: {alpha_fixed}")
    print(f"------------------------------------")

    for epoch in range(num_epochs):
        epoch_loss_nn_objective = 0.0

        for batch_x, batch_y_target, batch_s_safe_initial, batch_v_lead_initial in train_loader:
            batch_x = batch_x.to(device)
            batch_y_target = batch_y_target.to(device)  # batch_y_target shape: (batch, 1)
            batch_s_safe_initial = batch_s_safe_initial.to(device)
            batch_v_lead_initial = batch_v_lead_initial.to(device)

            optimizer_nn.zero_grad()

            y_nn_pred, y_idm_pred = model(batch_x, batch_s_safe_initial,
                                          batch_v_lead_initial)  # y_nn_pred, y_idm_pred shape: (batch, 1)

            # Loss 1: Difference between NN prediction and ground truth
            loss_nn_vs_true = ((y_nn_pred - batch_y_target).pow(2) * loss_weights.unsqueeze(0)).mean()
            # Loss 2: Difference between NN prediction and IDM prediction
            loss_nn_vs_idm = ((y_nn_pred - y_idm_pred.detach()).pow(2) * loss_weights.unsqueeze(0)).mean()

            loss_for_nn_params = alpha_fixed * loss_nn_vs_true + \
                                 (1 - alpha_fixed) * loss_nn_vs_idm

            loss_for_nn_params.backward()
            optimizer_nn.step()

            epoch_loss_nn_objective += loss_for_nn_params.item()

        avg_nn_loss = epoch_loss_nn_objective / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}  NN Objective Loss: {avg_nn_loss:.6f}  (α={alpha_fixed})")
    return model


# --- Evaluation Function (Single-step version) ---
def evaluate_model(model, test_loader, device, alpha_decay_loss=0.1, dataset_name="", results_dir=""):
    model.eval()
    all_pred_nn, all_pred_idm, all_true = [], [], []

    loss_weights_device = next(model.parameters()).device
    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(PRED_HORIZON, dtype=torch.float32)).to(
        loss_weights_device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * PRED_HORIZON  # Will be tensor([1.0])

    total_mse_nn_vs_true_weighted = 0
    total_mse_idm_vs_true_weighted = 0

    with torch.no_grad():
        for batch_x, batch_y_target, batch_s_safe_initial, batch_v_lead_initial in test_loader:
            batch_x = batch_x.to(device)
            batch_y_target = batch_y_target.to(device)  # shape (batch, 1)
            batch_s_safe_initial = batch_s_safe_initial.to(device)
            batch_v_lead_initial = batch_v_lead_initial.to(device)

            y_nn_pred, y_idm_pred = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)  # shape (batch, 1)

            all_pred_nn.append(y_nn_pred.cpu())
            all_pred_idm.append(y_idm_pred.cpu())
            all_true.append(batch_y_target.cpu())

            loss_nn_vs_true_batch_weighted = ((y_nn_pred - batch_y_target).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_nn_vs_true_weighted += loss_nn_vs_true_batch_weighted.item() * batch_x.size(0)

            loss_idm_vs_true_batch_weighted = ((y_idm_pred - batch_y_target).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_idm_vs_true_weighted += loss_idm_vs_true_batch_weighted.item() * batch_x.size(0)

    num_samples = len(test_loader.dataset)
    avg_mse_nn_vs_true_weighted = total_mse_nn_vs_true_weighted / num_samples
    avg_mse_idm_vs_true_weighted = total_mse_idm_vs_true_weighted / num_samples

    fixed_alpha_for_metrics = 0.7

    y_pred_nn_cat = torch.cat(all_pred_nn)  # shape (num_samples, 1)
    y_pred_idm_cat = torch.cat(all_pred_idm)  # shape (num_samples, 1)
    y_true_cat = torch.cat(all_true)  # shape (num_samples, 1)

    y_final_prediction_cat = y_pred_nn_cat  # Use NN output for final prediction

    # Calculate overall metrics (for single-step)
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

    print(f"\n--- Test Result Summary (Single-step, Final NN Prediction) ---")
    print(
        f"  NN Prediction vs True -- MSE: {mse_val_overall:.4f}, RMSE: {rmse_val_overall:.4f}, MAE: {mae_val_overall:.4f}, MAPE: {mape_p_overall if not np.isnan(mape_p_overall) else 'N/A'}%")
    print(
        f"  (Reference: IDM Prediction vs True MSE: {avg_mse_idm_vs_true_weighted:.4f})")  # MSE is weighted, but for single step it's same as unweighted
    print(f"  (Reference: NN Prediction vs True MSE (Train Metric): {avg_mse_nn_vs_true_weighted:.4f})")
    print(f"  Alpha Value Used (Fixed)={fixed_alpha_for_metrics:.4f}")

    # Detailed metrics for single-step (same as overall)
    print(f"\n--- Single-step Detailed Metrics (Step 1) ---")
    y_pred_nn_step_1 = y_pred_nn_cat[:, 0]
    y_pred_idm_step_1 = y_pred_idm_cat[:, 0]
    y_true_step_1 = y_true_cat[:, 0]

    mse_step_nn = nn.MSELoss()(y_pred_nn_step_1, y_true_step_1).item()
    rmse_step_nn = np.sqrt(mse_step_nn)
    mae_step_nn = torch.mean(torch.abs(y_pred_nn_step_1 - y_true_step_1)).item()

    abs_error_step = torch.abs(y_pred_nn_step_1 - y_true_step_1)
    abs_true_step = torch.abs(y_true_step_1)
    valid_mape_mask_step = abs_true_step > 1e-6
    mape_step_nn = float('nan')
    if torch.sum(valid_mape_mask_step) > 0:
        mape_step_nn = torch.mean(
            abs_error_step[valid_mape_mask_step] / abs_true_step[valid_mape_mask_step]
        ).item() * 100
    mse_step_idm = nn.MSELoss()(y_pred_idm_step_1, y_true_step_1).item()

    print(f"  Step 1:")
    print(
        f"    NN Prediction -- MSE: {mse_step_nn:.4f}, RMSE: {rmse_step_nn:.4f}, MAE: {mae_step_nn:.4f}, MAPE: {mape_step_nn if not np.isnan(mape_step_nn) else 'N/A'}%")
    print(f"    IDM (Reference) -- MSE: {mse_step_idm:.4f}")

    # Plot single-step comparison
    plt.figure(figsize=(12, 7))
    plt.plot(y_true_cat[:100, 0].numpy(), '--o', label=f'True (Step 1)')
    plt.plot(y_pred_nn_cat[:100, 0].numpy(), '-x', label=f'NN Prediction (Step 1) (Final)')
    plt.plot(y_pred_idm_cat[:100, 0].numpy(), '-s', label=f'IDM Prediction (Step 1) (Ref)')

    y_pred_combined_for_plot_cat = fixed_alpha_for_metrics * y_pred_nn_cat + \
                                   (1 - fixed_alpha_for_metrics) * y_pred_idm_cat
    plt.plot(y_pred_combined_for_plot_cat[:100, 0].numpy(), '-.',
             label=f'Hypothetical Fusion (Step 1, α={fixed_alpha_for_metrics:.2f}) (Ref Plot)')

    plt.title(f'Speed Prediction Comparison (First 100 Samples, Step 1) ({dataset_name})')
    plt.xlabel("Sample Index")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid()
    plot_filename = os.path.join(results_dir, f"{dataset_name}_speed_comparison_PITRANSFORMER_IDM_single_step.png")
    plt.savefig(plot_filename)
    print(f"Speed comparison plot saved to {plot_filename}")
    plt.close()

    return avg_mse_nn_vs_true_weighted, np.sqrt(avg_mse_nn_vs_true_weighted), mae_val_overall, mape_p_overall


# --- Modified compute_position_and_spacing_and_save (Single-step version) ---
def compute_position_and_spacing_and_save(model,
                                          test_loader,
                                          raw_data_all,  # Original dataset (for initial state)
                                          label_data_all,  # Label dataset (for true future pos/spacing)
                                          train_size,  # Training set size
                                          device,  # Device
                                          dt=0.1,  # Time step
                                          output_file="predictions_singlestep_extended.xlsx",
                                          dataset_name=""):
    model.eval()
    test_start_idx_in_all_data = train_size

    y_nn_list_mps, y_true_speeds_list_mps = [], []
    initial_ego_pos_ft_collected = []
    initial_lead_pos_ft_collected = []
    initial_ego_speed_ftps_collected = []
    initial_lead_speed_ftps_collected = []
    true_future_ego_pos_ft_collected = []  # True next ego position
    true_future_spacing_ft_collected = []  # True next spacing

    with torch.no_grad():
        for i, (batch_x_mps, batch_y_target_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps) in enumerate(
                test_loader):
            batch_x_mps = batch_x_mps.to(device)
            batch_s_safe_initial_m = batch_s_safe_initial_m.to(device)
            batch_v_lead_initial_mps = batch_v_lead_initial_mps.to(device)

            y_nn_pred_mps, _ = model(batch_x_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps)  # shape (batch, 1)

            y_nn_list_mps.append(y_nn_pred_mps.cpu())
            y_true_speeds_list_mps.append(batch_y_target_mps.cpu())  # batch_y_target_mps shape (batch, 1)

            current_batch_indices_in_all_data = np.arange(
                test_start_idx_in_all_data + i * test_loader.batch_size,
                test_start_idx_in_all_data + i * test_loader.batch_size + batch_x_mps.size(0)
            )
            # Extract initial state (ft/ftps)
            initial_ego_pos_ft_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 4].cpu())  # Ego Y pos
            initial_lead_pos_ft_collected.append(raw_data_all[current_batch_indices_in_all_data, -1, 4].cpu())  # Lead Y pos
            initial_ego_speed_ftps_collected.append(
                raw_data_all[current_batch_indices_in_all_data, -1, 0].cpu())  # Ego speed
            initial_lead_speed_ftps_collected.append(
                raw_data_all[current_batch_indices_in_all_data, -1, 5].cpu())  # Lead speed

            # Extract true next step position and spacing (ft)
            # label_data_all[:, 0, col_idx] takes label for the 1st future step
            true_future_ego_pos_ft_collected.append(
                label_data_all[current_batch_indices_in_all_data, 0, 3].cpu().unsqueeze(-1))  # Ego future Y pos, shape (batch,1)
            true_future_spacing_ft_collected.append(
                label_data_all[current_batch_indices_in_all_data, 0, 1].cpu().unsqueeze(-1))  # Future spacing, shape (batch,1)

    y_nn_all_mps = torch.cat(y_nn_list_mps, dim=0)  # (num_test_samples, 1)
    y_true_speeds_all_mps = torch.cat(y_true_speeds_list_mps, dim=0)  # (num_test_samples, 1)

    final_pred_speeds_mps = y_nn_all_mps  # (num_test_samples, 1)

    initial_ego_pos_ft = torch.cat(initial_ego_pos_ft_collected, dim=0)  # (num_test_samples,)
    initial_lead_pos_ft = torch.cat(initial_lead_pos_ft_collected, dim=0)  # (num_test_samples,)
    initial_ego_speed_ftps = torch.cat(initial_ego_speed_ftps_collected, dim=0)  # (num_test_samples,)
    initial_lead_speed_ftps = torch.cat(initial_lead_speed_ftps_collected, dim=0)  # (num_test_samples,)

    true_future_ego_pos_ft = torch.cat(true_future_ego_pos_ft_collected, dim=0)  # (num_test_samples, 1)
    true_future_spacing_ft = torch.cat(true_future_spacing_ft_collected, dim=0)  # (num_test_samples, 1)

    # Init tensors for single-step predicted position and spacing (ft)
    # Shape will be (num_test_samples, 1) as PRED_HORIZON is 1
    pred_ego_pos_next_step_ft = torch.zeros_like(final_pred_speeds_mps)
    pred_lead_pos_next_step_ft = torch.zeros_like(final_pred_speeds_mps)
    pred_spacing_next_step_ft = torch.zeros_like(final_pred_speeds_mps)

    final_pred_speeds_ftps = final_pred_speeds_mps / 0.3048  # (num_test_samples, 1)

    # Current iteration ego and lead position (ft), starting from initial observations
    current_ego_pos_ft = initial_ego_pos_ft.clone()  # (num_test_samples,)
    current_lead_pos_ft = initial_lead_pos_ft.clone()  # (num_test_samples,)
    lead_speed_constant_ftps = initial_lead_speed_ftps  # (num_test_samples,)

    # Calculate next step status (PRED_HORIZON=1, loop runs once for k=0)
    # k=0 : predict state at t+1
    # Use initial ego speed observation (at time t) to calculate displacement from t to t+1
    speed_ego_this_step_ftps = initial_ego_speed_ftps  # (num_test_samples,)

    disp_ego_ft = speed_ego_this_step_ftps * dt
    disp_lead_ft = lead_speed_constant_ftps * dt

    # Predicted position for next time step
    pred_ego_pos_next_step_ft[:, 0] = current_ego_pos_ft + disp_ego_ft
    pred_spacing_next_step_ft[:, 0] = pred_lead_pos_next_step_ft[:, 0] - pred_ego_pos_next_step_ft[:, 0]

    pred_ego_pos_m = pred_ego_pos_next_step_ft.numpy() * 0.3048  # (num_test_samples, 1)
    true_ego_pos_m = true_future_ego_pos_ft.numpy() * 0.3048  # (num_test_samples, 1)
    pred_spacing_m = pred_spacing_next_step_ft.numpy() * 0.3048  # (num_test_samples, 1)
    true_spacing_m = true_future_spacing_ft.numpy() * 0.3048  # (num_test_samples, 1)

    print(f"\n--- Error Evaluation of Position and Spacing based on NN Single-step Speed Prediction (Step 1) ---")
    # k_s = 0 for single step
    pos_err_sq_step = (pred_ego_pos_m[:, 0] - true_ego_pos_m[:, 0]) ** 2
    rmse_Y_step = np.sqrt(np.mean(pos_err_sq_step))
    valid_true_Y_step_mask = np.abs(true_ego_pos_m[:, 0]) > 1e-6
    mape_Y_step = float('nan')
    if np.sum(valid_true_Y_step_mask) > 0:
        mape_Y_step = np.mean(np.abs(
            (pred_ego_pos_m[valid_true_Y_step_mask, 0] - true_ego_pos_m[valid_true_Y_step_mask, 0]) /
            true_ego_pos_m[valid_true_Y_step_mask, 0])) * 100

    spacing_err_sq_step = (pred_spacing_m[:, 0] - true_spacing_m[:, 0]) ** 2
    rmse_sp_step = np.sqrt(np.mean(spacing_err_sq_step))
    valid_true_sp_step_mask = np.abs(true_spacing_m[:, 0]) > 1e-6
    mape_sp_step = float('nan')
    if np.sum(valid_true_sp_step_mask) > 0:
        mape_sp_step = np.mean(np.abs(
            (pred_spacing_m[valid_true_sp_step_mask, 0] - true_spacing_m[valid_true_sp_step_mask, 0]) /
            true_spacing_m[valid_true_sp_step_mask, 0])) * 100

    print(f"  Step 1:")
    print(f"    Position Error -- RMSE: {rmse_Y_step:.4f} m, MAPE: {mape_Y_step if not np.isnan(mape_Y_step) else 'N/A'}%")
    print(f"    Spacing Error -- RMSE: {rmse_sp_step:.4f} m, MAPE: {mape_sp_step if not np.isnan(mape_sp_step) else 'N/A'}%")

    # For single-step, "last_step" is the first and only step.
    rmse_p_last_step = rmse_Y_step
    mape_p_last_step = mape_Y_step

    df_data = {}
    df_data[f"NN Predicted Speed (m/s) Step 1"] = final_pred_speeds_mps[:, 0].numpy()
    df_data[f"True Speed (m/s) Step 1"] = y_true_speeds_all_mps[:, 0].numpy()
    df_data[f"Predicted Ego Pos Y (m) Step 1"] = pred_ego_pos_m[:, 0]
    df_data[f"True Ego Pos Y (m) Step 1"] = true_ego_pos_m[:, 0]
    df_data[f"Predicted Spacing (m) Step 1"] = pred_spacing_m[:, 0]
    df_data[f"True Spacing (m) Step 1"] = true_spacing_m[:, 0]
    df_pos = pd.DataFrame(df_data)

    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            df_pos.to_excel(writer, sheet_name=dataset_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df_pos.to_excel(writer, sheet_name=dataset_name, index=False)

    print(f"Single-step position and spacing predictions for {dataset_name} saved to '{output_file}' sheet '{dataset_name}'.")
    return rmse_p_last_step, mape_p_last_step


# --- Main Flow ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    if not data_files:
        print(f"No .mat files found in directory {DATA_DIR}. Program will exit.")
        exit()
    print(f"Found following dataset files: {data_files}")

    position_predictions_excel_path = os.path.join(RESULTS_DIR,
                                                   "pred_positions_all_datasets_pitransformer_idm_single_step1128.xlsx")
    LR_NN_PARAMS = 5e-4
    data = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')
    raw_all_ft = torch.tensor(data['train_data'], dtype=torch.float32)  # (samples, seq_len, features)
    lab_all_ft = torch.tensor(data['lable_data'], dtype=torch.float32)  # (samples, future_steps, label_features)

    # Feature extraction
    # Selected features: Ego speed(0), Spacing(1), Speed diff(2), Ego Y pos(3), Lead speed(5)
    seq_ft = raw_all_ft[:, :, [0, 1, 2, 3, 5]].clone()  # (samples, seq_len, 5 input features)

    # Target variable: Ego speed at next step (lable_v_NextTime)
    # lab_all_ft[:, 0, 0] -> first prediction step (index 0), first feature (index 0, i.e., speed)
    y_target_ftps = lab_all_ft[:, 0, 0].unsqueeze(-1).clone()  # (samples, 1)

    # Initial safety gap s_safe_initial_ft and lead speed v_lead_initial_ftps
    # Obtained from last time step of input sequence
    s_safe_initial_ft = seq_ft[:, -1, 1].clone()  # (samples,)
    v_lead_initial_ftps = raw_all_ft[:, -1, 5].clone()  # (samples,)

    # Unit Conversion: ft/ftps -> m/mps
    seq_mps = seq_ft.clone()
    seq_mps[:, :, [0, 2, 3, 4]] *= 0.3048  # v_ego, delta_v, a_ego, a_lead
    seq_mps[:, :, 1] *= 0.3048  # s_safe

    y_target_mps = y_target_ftps * 0.3048  # (samples, 1)
    s_safe_initial_m = s_safe_initial_ft * 0.3048  # (samples,)
    v_lead_initial_mps = v_lead_initial_ftps * 0.3048  # (samples,)

    N_total = seq_mps.size(0)
    N = int(N_total * 1)  # Use 10% for quick run
    print(f"Using {N} / {N_total} samples for training and testing.")

    seq_mps_selected = seq_mps[:N]
    y_target_mps_selected = y_target_mps[:N]
    s_safe_initial_m_selected = s_safe_initial_m[:N]
    v_lead_initial_mps_selected = v_lead_initial_mps[:N]
    raw_all_ft_selected = raw_all_ft[:N]  # For later position calculation
    lab_all_ft_selected = lab_all_ft[:N]  # For later position calculation

    split_ratio = 0.8
    train_size = int(N * split_ratio)

    train_seq = seq_mps_selected[:train_size]
    test_seq = seq_mps_selected[train_size:]
    train_y_target = y_target_mps_selected[:train_size]
    test_y_target = y_target_mps_selected[train_size:]
    train_s_safe_initial = s_safe_initial_m_selected[:train_size]
    test_s_safe_initial = s_safe_initial_m_selected[train_size:]
    train_v_lead_initial = v_lead_initial_mps_selected[:train_size]
    test_v_lead_initial = v_lead_initial_mps_selected[train_size:]

    batch_size = 32
    train_ds = torch.utils.data.TensorDataset(train_seq, train_y_target, train_s_safe_initial, train_v_lead_initial)
    test_ds = torch.utils.data.TensorDataset(test_seq, test_y_target, test_s_safe_initial, test_v_lead_initial)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = train_seq.size(2)  # Should be 5
    hidden_dim = 128
    n_head = 4
    transformer_layers = 2
    feedforward_dim = 512
    transformer_dropout = 0.1

    model = HybridIDMTransformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        nhead=n_head,
        transformer_num_layers=transformer_layers,
        dim_feedforward=feedforward_dim,
        dropout_transformer=transformer_dropout
    ).to(device)

    initialize_weights(model)

    # Train
    model = train_model(model, train_loader, device=device, num_epochs=50,
                        alpha_decay_loss=0.05, lr_nn=LR_NN_PARAMS)

    # Evaluate
    speed_mse_summary, speed_rmse_summary, speed_mae_overall, speed_mape_overall = evaluate_model(
        model, test_loader, device=device, alpha_decay_loss=0.05,
        dataset_name='data1', results_dir=RESULTS_DIR
    )

    print(f"Starting calculation and evaluation of position/spacing predictions: {'data1'}...")
    pos_rmse_next_step, pos_mape_next_step = compute_position_and_spacing_and_save(
        model, test_loader, raw_all_ft_selected, lab_all_ft_selected, train_size,
        device=device, dt=DT,
        output_file=position_predictions_excel_path, dataset_name='data1'
    )
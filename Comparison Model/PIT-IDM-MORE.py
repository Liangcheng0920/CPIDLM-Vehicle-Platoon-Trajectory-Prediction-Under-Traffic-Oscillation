import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import glob  # Used for finding file paths
import os  # OS interface for path operations and environment variables
import math  # Required for Transformer Positional Encoding

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow multiple OpenMP libraries to prevent conflicts

# --- Global Path Definitions ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # Dataset directory
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim_modified_transformer"  # Results directory

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Global Constants ---
DT = 0.1  # Time step (s)


# --- Data Validation Function ---
def check_data(data, name="data"):
    """ Check if the data contains NaN or Inf values """
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Contains NaN: {torch.isnan(data).any().item()}")
    print(f"Contains Inf: {torch.isinf(data).any().item()}")


# --- Fixed IDM Parameter Prediction Function ---
def idm_fixed(v_n, s_safe, delta_v,
              v_desired=10.13701546, T=0.50284384, a_max=0.10995557,
              b_safe=4.98369406, delta=5.35419582, s0=0.10337701,
              delta_t=DT):
    """
    Performs one-step IDM (Intelligent Driver Model) prediction using a fixed parameter set.
    :param v_n: Current ego speed (m/s)
    :param s_safe: Current actual gap (m)
    :param delta_v: Current speed difference (Lead - Ego, m/s)
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
    # Convert IDM parameters to tensors on the same device and type as input
    v_desired = torch.tensor(v_desired, device=current_device, dtype=v_n.dtype)
    T = torch.tensor(T, device=current_device, dtype=v_n.dtype)
    a_max = torch.tensor(a_max, device=current_device, dtype=v_n.dtype).clamp(min=1e-6)  # Avoid division by zero
    b_safe = torch.tensor(b_safe, device=current_device, dtype=v_n.dtype).clamp(min=1e-6)
    s0 = torch.tensor(s0, device=current_device, dtype=v_n.dtype)
    delta_param = torch.tensor(delta, device=current_device, dtype=v_n.dtype)
    delta_t_tensor = torch.tensor(delta_t, device=current_device, dtype=v_n.dtype)

    s_safe = s_safe.clamp(min=1e-6)  # Ensure spacing is positive

    # Calculate desired spacing s*
    s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe) + 1e-6)
    s_star = s_star.clamp(min=0.0)

    # Calculate velocity term (v_n / v_desired)^delta
    v_n_ratio = torch.zeros_like(v_n)
    mask_v_desired_nonzero = v_desired.abs() > 1e-6
    if mask_v_desired_nonzero.any():
        v_n_ratio[mask_v_desired_nonzero] = (v_n[mask_v_desired_nonzero] / v_desired[mask_v_desired_nonzero])

    # IDM Acceleration formula
    acceleration_term = a_max * (
            1 - v_n_ratio ** delta_param - (s_star / s_safe) ** 2
    )
    v_follow = v_n + delta_t_tensor * acceleration_term
    return v_follow.clamp(min=0.0)


# --- Transformer Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # buffer stays with the model but is not a trained parameter

    def forward(self, x):
        """ x shape: (batch_size, seq_len, d_model) """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- Define Hybrid Transformer-based Model ---
class HybridIDMTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 nhead=4, transformer_num_layers=2, dim_feedforward=512, dropout_transformer=0.1):
        super(HybridIDMTransformerModel, self).__init__()
        self.pred_horizon = output_dim  # Prediction horizon K
        self.model_dim = hidden_dim  # d_model

        # Projection layer: input_dim -> model_dim
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
        # Mapping last hidden state to K prediction steps
        self.fc = nn.Linear(self.model_dim, self.pred_horizon)

        # Fixed IDM Parameters
        self.v_desired_idm = 12.64798288
        self.T_idm = 0.50284384
        self.a_max_idm = 0.10033688
        self.b_safe_idm = 4.98937183
        self.delta_idm = 1.0
        self.s0_idm = 0.13082412

    def forward(self, x, s_safe_initial, v_lead_initial):
        """
        Forward Pass.
        :param x: Input sequence, shape=(batch, seq_len, input_dim)
        :param s_safe_initial: Observed safety gap (for IDM iteration), shape=(batch,)
        :param v_lead_initial: Lead vehicle speed (assumed constant for K steps), shape=(batch,)
        :return: NN K-step prediction, IDM K-step prediction
        """
        # Transformer Multi-step Prediction
        x_transformed = self.input_fc(x)
        x_transformed = self.pos_encoder(x_transformed)
        transformer_out = self.transformer_encoder(x_transformed)

        # Take the output of the last time step for prediction
        y_nn_multistep = self.fc(transformer_out[:, -1, :])

        # Iterated IDM Multi-step Prediction
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

            # Update state for next IDM iteration: s_new = s + (v_lead - v_ego) * dt
            s_current_idm = (s_current_idm + (v_lead_constant_idm - v_ego_current_idm) * DT).clamp(min=1e-6)
            v_ego_current_idm = v_ego_next_pred_idm

        y_idm_multistep = torch.cat(y_idm_multistep_list, dim=1)
        return y_nn_multistep, y_idm_multistep


def initialize_weights(model):
    """ Initialize weights using Xavier uniform """
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)

        # --- Training Function (Fixed Alpha) ---


def train_model(model, train_loader, pred_horizon, device, num_epochs=30, alpha_decay_loss=0.1, lr_nn=5e-4):
    model.train()
    # Loss weights favoring earlier steps in the sequence
    loss_weights_device = next(model.parameters()).device
    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(pred_horizon, dtype=torch.float32)).to(
        loss_weights_device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * pred_horizon

    nn_params = [param for name, param in model.named_parameters()]
    optimizer_nn = optim.Adam(nn_params, lr=lr_nn)

    alpha_fixed = 0.7

    print(f"--- Training with Fixed Alpha (Device: {device}) ---")
    print(f"NN optimized via L_nn = alpha * L_true + (1-alpha) * L_idm")
    print(f"Fixed Alpha: {alpha_fixed}")
    print(f"------------------------------------")

    for epoch in range(num_epochs):
        epoch_loss_nn_objective = 0.0

        for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in train_loader:
            batch_x, batch_y_multistep = batch_x.to(device), batch_y_multistep.to(device)
            batch_s_safe_initial, batch_v_lead_initial = batch_s_safe_initial.to(device), batch_v_lead_initial.to(
                device)

            optimizer_nn.zero_grad()

            # Forward pass
            y_nn_multistep, y_idm_multistep = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)

            # Calculate individual loss components
            # Loss 1: Difference between NN prediction and Ground Truth
            loss_nn_vs_true = ((y_nn_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            # Loss 2: Difference between NN prediction and Physics-based IDM prediction
            loss_nn_vs_idm = ((y_nn_multistep - y_idm_multistep.detach()).pow(2) * loss_weights.unsqueeze(0)).mean()

            # Final objective combining data and physics
            loss_for_nn_params = alpha_fixed * loss_nn_vs_true + (1 - alpha_fixed) * loss_nn_vs_idm

            loss_for_nn_params.backward()
            optimizer_nn.step()

            epoch_loss_nn_objective += loss_for_nn_params.item()

        avg_nn_loss = epoch_loss_nn_objective / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}  NN Objective Loss: {avg_nn_loss:.6f}  (α={alpha_fixed})")
    return model


# --- Evaluation Function ---
def evaluate_model(model, test_loader, pred_horizon, device, alpha_decay_loss=0.1, dataset_name="", results_dir=""):
    model.eval()
    all_pred_nn, all_pred_idm, all_true = [], [], []

    loss_weights_device = next(model.parameters()).device
    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(pred_horizon, dtype=torch.float32)).to(
        loss_weights_device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * pred_horizon

    total_mse_nn_vs_true_weighted = 0
    total_mse_idm_vs_true_weighted = 0

    with torch.no_grad():
        for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in test_loader:
            batch_x, batch_y_multistep = batch_x.to(device), batch_y_multistep.to(device)
            batch_s_safe_initial, batch_v_lead_initial = batch_s_safe_initial.to(device), batch_v_lead_initial.to(
                device)

            y_nn_multistep, y_idm_multistep = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)

            all_pred_nn.append(y_nn_multistep.cpu())
            all_pred_idm.append(y_idm_multistep.cpu())
            all_true.append(batch_y_multistep.cpu())

            loss_nn_vs_true_batch_weighted = (
                        (y_nn_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_nn_vs_true_weighted += loss_nn_vs_true_batch_weighted.item() * batch_x.size(0)

            loss_idm_vs_true_batch_weighted = (
                        (y_idm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_idm_vs_true_weighted += loss_idm_vs_true_batch_weighted.item() * batch_x.size(0)

    num_samples = len(test_loader.dataset)
    avg_mse_nn_vs_true_weighted = total_mse_nn_vs_true_weighted / num_samples
    avg_mse_idm_vs_true_weighted = total_mse_idm_vs_true_weighted / num_samples

    y_pred_nn_cat = torch.cat(all_pred_nn)
    y_pred_idm_cat = torch.cat(all_pred_idm)
    y_true_cat = torch.cat(all_true)

    # Final prediction results take the NN output
    y_final_prediction_cat = y_pred_nn_cat

    mse_val_overall = torch.mean((y_final_prediction_cat - y_true_cat).pow(2)).item()
    rmse_val_overall = np.sqrt(mse_val_overall)
    mae_val_overall = torch.mean(torch.abs(y_final_prediction_cat - y_true_cat)).item()

    # MAPE (%) Calculation
    abs_error_overall = torch.abs(y_final_prediction_cat - y_true_cat)
    abs_true_overall = torch.abs(y_true_cat)
    valid_mape_mask_overall = abs_true_overall > 1e-6
    mape_p_overall = float('nan')
    if torch.sum(valid_mape_mask_overall) > 0:
        mape_p_overall = torch.mean(
            abs_error_overall[valid_mape_mask_overall] / abs_true_overall[valid_mape_mask_overall]
        ).item() * 100

    print(f"\n--- Test Summary (Final NN Prediction, simple average across all steps) ---")
    print(
        f"  NN vs True -- MSE: {mse_val_overall:.4f}, RMSE: {rmse_val_overall:.4f}, MAE: {mae_val_overall:.4f}, MAPE: {mape_p_overall if not np.isnan(mape_p_overall) else 'N/A'}%")
    print(f"  (Ref: IDM vs True Weighted MSE: {avg_mse_idm_vs_true_weighted:.4f})")
    print(f"  (Ref: NN vs True Weighted MSE (Train Metric): {avg_mse_nn_vs_true_weighted:.4f})")

    print(f"\n--- Metrics Per Prediction Step ---")
    for k_step in range(pred_horizon):
        y_pred_nn_step_k = y_pred_nn_cat[:, k_step]
        y_pred_idm_step_k = y_pred_idm_cat[:, k_step]
        y_true_step_k = y_true_cat[:, k_step]

        mse_step_nn = nn.MSELoss()(y_pred_nn_step_k, y_true_step_k).item()
        rmse_step_nn = np.sqrt(mse_step_nn)
        mae_step_nn = torch.mean(torch.abs(y_pred_nn_step_k - y_true_step_k)).item()

        abs_error_step = torch.abs(y_pred_nn_step_k - y_true_step_k)
        abs_true_step = torch.abs(y_true_step_k)
        valid_mape_mask_step = abs_true_step > 1e-6
        mape_step_nn = float('nan')
        if torch.sum(valid_mape_mask_step) > 0:
            mape_step_nn = torch.mean(
                abs_error_step[valid_mape_mask_step] / abs_true_step[valid_mape_mask_step]).item() * 100

        print(
            f"  Step {k_step + 1}: NN RMSE: {rmse_step_nn:.4f}, MAE: {mae_step_nn:.4f}, MAPE: {mape_step_nn if not np.isnan(mape_step_nn) else 'N/A'}% | IDM MSE: {nn.MSELoss()(y_pred_idm_step_k, y_true_step_k).item():.4f}")

    # Plot Comparison (First 100 samples)
    k_plot = 0
    plt.figure(figsize=(12, 7))
    plt.plot(y_true_cat[:100, k_plot].numpy(), '--o', label=f'Ground Truth (Step {k_plot + 1})')
    plt.plot(y_pred_nn_cat[:100, k_plot].numpy(), '-x', label=f'NN Prediction (Step {k_plot + 1}) (Final)')
    plt.plot(y_pred_idm_cat[:100, k_plot].numpy(), '-s', label=f'IDM Prediction (Step {k_plot + 1}) (Ref)')

    plt.title(f'Speed Prediction Comparison (Step {k_plot + 1}) ({dataset_name})')
    plt.xlabel("Sample Index")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid()
    plot_filename = os.path.join(results_dir, f"{dataset_name}_speed_comparison.png")
    plt.savefig(plot_filename)
    print(f"Speed comparison plot saved to {plot_filename}")
    plt.close()

    return mse_val_overall, rmse_val_overall, mae_val_overall, mape_p_overall


# --- Compute Position and Spacing and Save ---
def compute_position_and_spacing_and_save(model,
                                          test_loader,
                                          raw_data_all,
                                          label_data_all,
                                          train_size,
                                          pred_horizon,
                                          device,
                                          dt=0.1,
                                          output_file="predictions_extended.xlsx",
                                          dataset_name=""):
    model.eval()
    test_start_idx_in_all_data = train_size

    y_nn_list_mps, y_true_speeds_list_mps = [], []
    initial_ego_pos_ft_collected = []
    initial_lead_pos_ft_collected = []
    initial_ego_speed_ftps_collected = []
    initial_lead_speed_ftps_collected = []
    true_future_ego_pos_ft_collected = []
    true_future_spacing_ft_collected = []

    with torch.no_grad():
        for i, (batch_x_mps, batch_y_multistep_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps) in enumerate(
                test_loader):
            batch_x_mps = batch_x_mps.to(device)
            batch_s_safe_initial_m = batch_s_safe_initial_m.to(device)
            batch_v_lead_initial_mps = batch_v_lead_initial_mps.to(device)

            y_nn_k_mps, _ = model(batch_x_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps)

            y_nn_list_mps.append(y_nn_k_mps.cpu())
            y_true_speeds_list_mps.append(batch_y_multistep_mps.cpu())

            current_batch_indices = np.arange(
                test_start_idx_in_all_data + i * test_loader.batch_size,
                test_start_idx_in_all_data + i * test_loader.batch_size + batch_x_mps.size(0)
            )

            # Extract initial status (ft/ftps)
            initial_ego_pos_ft_collected.append(raw_data_all[current_batch_indices, -1, 4].cpu())
            initial_lead_pos_ft_collected.append(raw_data_all[current_batch_indices, -1, 7].cpu())
            initial_ego_speed_ftps_collected.append(raw_data_all[current_batch_indices, -1, 0].cpu())
            initial_lead_speed_ftps_collected.append(raw_data_all[current_batch_indices, -1, 5].cpu())

            # Extract ground truth future status (ft)
            true_future_ego_pos_ft_collected.append(label_data_all[current_batch_indices, :pred_horizon, 3].cpu())
            true_future_spacing_ft_collected.append(label_data_all[current_batch_indices, :pred_horizon, 1].cpu())

    y_nn_all_mps = torch.cat(y_nn_list_mps, dim=0)
    y_true_speeds_all_mps = torch.cat(y_true_speeds_list_mps, dim=0)
    final_pred_speeds_ftps = y_nn_all_mps / 0.3048

    initial_ego_pos_ft = torch.cat(initial_ego_pos_ft_collected, dim=0)
    initial_lead_pos_ft = torch.cat(initial_lead_pos_ft_collected, dim=0)
    initial_ego_speed_ftps = torch.cat(initial_ego_speed_ftps_collected, dim=0)
    initial_lead_speed_ftps = torch.cat(initial_lead_speed_ftps_collected, dim=0)

    true_future_ego_pos_ft = torch.cat(true_future_ego_pos_ft_collected, dim=0)
    true_future_spacing_ft = torch.cat(true_future_spacing_ft_collected, dim=0)

    pred_ego_pos_ft = torch.zeros_like(final_pred_speeds_ftps)
    pred_spacing_ft = torch.zeros_like(final_pred_speeds_ftps)

    curr_ego_pos_ft = initial_ego_pos_ft.clone()
    curr_lead_pos_ft = initial_lead_pos_ft.clone()
    lead_speed_constant_ftps = initial_lead_speed_ftps

    # Recursive calculation of future positions and spacing
    for k in range(pred_horizon):
        ego_v_ftps = initial_ego_speed_ftps if k == 0 else final_pred_speeds_ftps[:, k - 1]

        curr_ego_pos_ft += ego_v_ftps * dt
        curr_lead_pos_ft += lead_speed_constant_ftps * dt  # Lead vehicle speed assumed constant

        pred_ego_pos_ft[:, k] = curr_ego_pos_ft
        pred_spacing_ft[:, k] = curr_lead_pos_ft - curr_ego_pos_ft

    # Convert to meters for metrics
    pred_ego_pos_m, true_ego_pos_m = pred_ego_pos_ft.numpy() * 0.3048, true_future_ego_pos_ft.numpy() * 0.3048
    pred_spacing_m, true_spacing_m = pred_spacing_ft.numpy() * 0.3048, true_future_spacing_ft.numpy() * 0.3048

    print(f"\n--- Detailed Position and Spacing Evaluation ---")
    for k_s in range(pred_horizon):
        rmse_Y = np.sqrt(np.mean((pred_ego_pos_m[:, k_s] - true_ego_pos_m[:, k_s]) ** 2))
        rmse_sp = np.sqrt(np.mean((pred_spacing_m[:, k_s] - true_spacing_m[:, k_s]) ** 2))
        print(f"  Step {k_s + 1}: Position RMSE: {rmse_Y:.4f} m | Spacing RMSE: {rmse_sp:.4f} m")

    rmse_p_final = np.sqrt(np.mean((pred_ego_pos_m - true_ego_pos_m) ** 2))

    # Prepare DataFrame for Excel
    df_data = {}
    for k_idx in range(pred_horizon):
        df_data[f"NN Pred Speed (m/s) Step {k_idx + 1}"] = y_nn_all_mps[:, k_idx].numpy()
        df_data[f"True Speed (m/s) Step {k_idx + 1}"] = y_true_speeds_all_mps[:, k_idx].numpy()
        df_data[f"Pred Ego Pos Y (m) Step {k_idx + 1}"] = pred_ego_pos_m[:, k_idx]
        df_data[f"True Ego Pos Y (m) Step {k_idx + 1}"] = true_ego_pos_m[:, k_idx]
        df_data[f"Pred Spacing (m) Step {k_idx + 1}"] = pred_spacing_m[:, k_idx]
        df_data[f"True Spacing (m) Step {k_idx + 1}"] = true_spacing_m[:, k_idx]

    df_pos = pd.DataFrame(df_data)
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            df_pos.to_excel(writer, sheet_name=dataset_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df_pos.to_excel(writer, sheet_name=dataset_name, index=False)

    print(f"{dataset_name} predictions saved to '{output_file}'.")
    return rmse_p_final, 0.0


# --- Helper for Metrics Summary ---
all_datasets_metrics_summary = []


def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse, pos_mape):
    metrics = {
        "Dataset": dataset_name,
        "Speed_MSE_NN": speed_mse,
        "Speed_RMSE_NN": speed_rmse,
        "Speed_MAE_NN": speed_mae,
        "Speed_MAPE_NN (%)": speed_mape,
        "Pos_RMSE_Final_m": pos_rmse,
        "Pos_MAPE_Final (%)": pos_mape
    }
    all_datasets_metrics_summary.append(metrics)


# --- Main Flow ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    if not data_files:
        print(f"No .mat files found in {DATA_DIR}.")
        exit()

    excel_path = os.path.join(RESULTS_DIR, "pred_results_summary.xlsx")

    for data_file_path in data_files:
        dataset_name = os.path.basename(data_file_path).replace(".mat", "")
        print(f"\n==================== Processing Dataset: {dataset_name} ====================")

        data = sio.loadmat(data_file_path)
        raw_all, lab_all = torch.tensor(data['train_data'], dtype=torch.float32), torch.tensor(data['lable_data'],
                                                                                               dtype=torch.float32)

        # Preprocessing: ft to meters conversion
        seq_mps = raw_all[:, :, [0, 1, 2, 3, 5]].clone()
        seq_mps[:, :, [0, 2, 3, 4]] *= 0.3048
        seq_mps[:, :, 1] *= 0.3048
        y_multistep_mps = lab_all[:, :, 0].clone() * 0.3048
        horizon = y_multistep_mps.shape[1]

        # Use 20% of data for training/testing
        N = int(seq_mps.size(0) * 0.2)
        train_size = int(N * 0.8)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                seq_mps[:train_size].to(device), y_multistep_mps[:train_size].to(device),
                (raw_all[:train_size, -1, 1] * 0.3048).to(device), (raw_all[:train_size, -1, 5] * 0.3048).to(device)
            ), batch_size=32, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                seq_mps[train_size:N].to(device), y_multistep_mps[train_size:N].to(device),
                (raw_all[train_size:N, -1, 1] * 0.3048).to(device), (raw_all[train_size:N, -1, 5] * 0.3048).to(device)
            ), batch_size=32, shuffle=False
        )

        model = HybridIDMTransformerModel(input_dim=5, hidden_dim=128, output_dim=horizon).to(device)
        initialize_weights(model)

        print(f"Training PIT-IDM model...")
        model = train_model(model, train_loader, pred_horizon=horizon, device=device, num_epochs=50,
                            alpha_decay_loss=0.05, lr_nn=5e-4)

        print(f"Evaluating Speed Predictions...")
        mse, rmse, mae, mape = evaluate_model(model, test_loader, pred_horizon=horizon, device=device,
                                              alpha_decay_loss=0.05, dataset_name=dataset_name, results_dir=RESULTS_DIR)

        print(f"Computing Position and Spacing...")
        pos_rmse, pos_mape = compute_position_and_spacing_and_save(model, test_loader, raw_all[:N], lab_all[:N],
                                                                   train_size, pred_horizon=horizon, device=device,
                                                                   dt=DT, output_file=excel_path,
                                                                   dataset_name=dataset_name)

        store_dataset_metrics(dataset_name, mse, rmse, mae, mape, pos_rmse, pos_mape)

    pd.DataFrame(all_datasets_metrics_summary).to_csv(os.path.join(RESULTS_DIR, "final_summary.csv"), index=False,
                                                      encoding='utf-8-sig')
    print("\nAll datasets processed. Evaluation summary saved.")
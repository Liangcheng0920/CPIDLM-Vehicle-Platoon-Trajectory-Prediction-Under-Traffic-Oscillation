import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import glob  # Used for file path searching
import os  # OS interface for path operations and environment variables

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow multiple OpenMP libraries to prevent conflicts

# --- Global Path Definitions ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # Directory for datasets
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim_modified"  # Directory for saving experimental results

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Global Constants ---
DT = 0.1  # Time step (s) - Inferred from original code


# --- Data Validation Function ---
def check_data(data, name="data"):
    """ Check if the data contains NaN or Inf values """
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Contains NaN: {torch.isnan(data).any().item()}")
    print(f"Contains Inf: {torch.isinf(data).any().item()}")


# --- Fixed IDM Parameter Prediction Function ---
def idm_fixed(v_n, s_safe, delta_v,
              v_desired=10.13701546, T=0.50290469, a_max=0.10995557,
              b_safe=4.98369406, delta=5.35419582, s0=0.10337701,
              delta_t=0.1):
    """
    Performs one-step IDM (Intelligent Driver Model) prediction using a fixed parameter set.
    :param v_n: Current ego vehicle speed (m/s)
    :param s_safe: Current actual gap (m)
    :param delta_v: Current speed difference (Lead speed - Ego speed, m/s)
    :param v_desired: Desired speed (m/s)
    :param T: Safe time headway (s)
    :param a_max: Max acceleration (m/s^2)
    :param b_safe: Comfortable deceleration (m/s^2)
    :param delta: Acceleration exponent
    :param s0: Minimum static gap (m)
    :param delta_t: Time step (s)
    :return: Predicted ego vehicle speed for the next step (m/s)
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


# --- Hybrid Model Definition (Modified for fixed alpha) ---
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):  # output_dim is prediction horizon K
        super(HybridIDMModel, self).__init__()
        self.pred_horizon = output_dim  # Save prediction horizon K
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.pred_horizon)  # LSTM produces K predictions

        # IDM Parameters (fixed values)
        self.v_desired_idm = 12.64798288
        self.T_idm = 0.50284384
        self.a_max_idm = 0.10033688
        self.b_safe_idm = 4.98937183
        self.delta_idm = 1.0
        self.s0_idm = 0.13082412

    def forward(self, x, s_safe_initial, v_lead_initial):
        """
        Forward propagation of the model.
        :param x: Input sequence, shape=(batch, seq_len, input_dim)
        :param s_safe_initial: Initial safe gap (observed, for multi-step IDM iteration), shape=(batch,)
        :param v_lead_initial: Current speed of lead vehicle (assumed constant for K steps), shape=(batch,)
        :return:
          y_lstm_multistep: Multi-step speed prediction from LSTM output, shape=(batch, K)
          y_idm_multistep: Multi-step speed prediction from iterated IDM, shape=(batch, K)
        """
        device = x.device

        # K-step LSTM prediction
        out, _ = self.lstm(x)
        y_lstm_multistep = self.fc(out[:, -1, :])

        # K-step iterated IDM prediction
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
            # Update gap for next step: s_next = s + (v_lead - v_ego) * dt
            s_current_idm = (s_current_idm + (v_lead_constant_idm - v_ego_current_idm) * DT).clamp(min=1e-6)
            v_ego_current_idm = v_ego_next_pred_idm

        y_idm_multistep = torch.cat(y_idm_multistep_list, dim=1)
        return y_lstm_multistep, y_idm_multistep


def initialize_weights(model):
    """ Initialize model weights """
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)


# --- Training Function (Modified for fixed alpha = 0.7) ---
def train_model(model, train_loader, pred_horizon, num_epochs=30, alpha_decay_loss=0.1, lr_lstm=5e-4):
    model.train()
    # Loss weights to emphasize earlier prediction steps
    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(pred_horizon, dtype=torch.float32)).to(
        next(model.parameters()).device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * pred_horizon

    lstm_params = [param for name, param in model.named_parameters()]
    optimizer_lstm = optim.Adam(lstm_params, lr=lr_lstm)

    alpha_fixed = 0.7  # Alpha set to a fixed value

    print(f"--- Training with Fixed Alpha ---")
    print(f"LSTM parameters will be optimized based on L_lstm = alpha * L_true + (1-alpha) * L_idm.")
    print(f"Alpha is fixed at: {alpha_fixed}")
    print(f"------------------------------------")

    for epoch in range(num_epochs):
        epoch_loss_lstm_objective = 0.0

        for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in train_loader:
            optimizer_lstm.zero_grad()

            # Forward propagation
            y_lstm_multistep, y_idm_multistep = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)

            # Calculate individual loss components
            loss_lstm_vs_true = ((y_lstm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            loss_lstm_vs_idm = ((y_lstm_multistep - y_idm_multistep.detach()).pow(2) * loss_weights.unsqueeze(0)).mean()

            # --- LSTM Parameter Update ---
            # Use combined loss with fixed alpha to update LSTM weights
            loss_for_lstm_params = alpha_fixed * loss_lstm_vs_true + \
                                   (1 - alpha_fixed) * loss_lstm_vs_idm

            loss_for_lstm_params.backward()
            optimizer_lstm.step()

            epoch_loss_lstm_objective += loss_for_lstm_params.item()

        avg_lstm_loss = epoch_loss_lstm_objective / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}  LSTM Objective Loss: {avg_lstm_loss:.6f}  (α={alpha_fixed})")
    return model


# --- Test/Evaluation Function (Modified for fixed alpha) ---
def evaluate_model(model, test_loader, pred_horizon, alpha_decay_loss=0.1, dataset_name="", results_dir=""):
    model.eval()
    all_pred_lstm, all_pred_idm, all_true = [], [], []

    loss_weights = torch.exp(-alpha_decay_loss * torch.arange(pred_horizon, dtype=torch.float32)).to(
        next(model.parameters()).device)
    loss_weights = loss_weights / (loss_weights.sum() + 1e-9) * pred_horizon

    total_mse_lstm_vs_true_weighted = 0
    total_mse_idm_vs_true_weighted = 0

    with torch.no_grad():
        for batch_x, batch_y_multistep, batch_s_safe_initial, batch_v_lead_initial in test_loader:
            y_lstm_multistep, y_idm_multistep = model(batch_x, batch_s_safe_initial, batch_v_lead_initial)

            all_pred_lstm.append(y_lstm_multistep.cpu())
            all_pred_idm.append(y_idm_multistep.cpu())
            all_true.append(batch_y_multistep.cpu())

            loss_lstm_vs_true_batch_weighted = (
                    (y_lstm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_lstm_vs_true_weighted += loss_lstm_vs_true_batch_weighted.item() * batch_x.size(0)

            loss_idm_vs_true_batch_weighted = (
                    (y_idm_multistep - batch_y_multistep).pow(2) * loss_weights.unsqueeze(0)).mean()
            total_mse_idm_vs_true_weighted += loss_idm_vs_true_batch_weighted.item() * batch_x.size(0)

    num_samples = len(test_loader.dataset)
    avg_mse_lstm_vs_true_weighted = total_mse_lstm_vs_true_weighted / num_samples
    avg_mse_idm_vs_true_weighted = total_mse_idm_vs_true_weighted / num_samples

    fixed_alpha_for_metrics = 0.7
    y_pred_lstm_cat = torch.cat(all_pred_lstm)
    y_pred_idm_cat = torch.cat(all_pred_idm)
    y_true_cat = torch.cat(all_true)

    # Use LSTM output as the final prediction
    y_final_prediction_cat = y_pred_lstm_cat

    mse_val_overall = torch.mean((y_final_prediction_cat - y_true_cat).pow(2)).item()
    rmse_val_overall = np.sqrt(mse_val_overall)
    mae_val_overall = torch.mean(torch.abs(y_final_prediction_cat - y_true_cat)).item()

    # MAPE calculation
    abs_error_overall = torch.abs(y_final_prediction_cat - y_true_cat)
    abs_true_overall = torch.abs(y_true_cat)
    valid_mape_mask_overall = abs_true_overall > 1e-6
    mape_p_overall = float('nan')
    if torch.sum(valid_mape_mask_overall) > 0:
        mape_p_overall = torch.mean(
            abs_error_overall[valid_mape_mask_overall] / abs_true_overall[valid_mape_mask_overall]
        ).item() * 100

    print(f"\n--- Test Results Summary (LSTM final prediction, simple average across all steps) ---")
    print(
        f"  LSTM Prediction vs True -- MSE: {mse_val_overall:.4f}, RMSE: {rmse_val_overall:.4f}, MAE: {mae_val_overall:.4f}, MAPE: {mape_p_overall if not np.isnan(mape_p_overall) else 'N/A'}%")
    print(f"  (Reference: IDM vs True Weighted MSE: {avg_mse_idm_vs_true_weighted:.4f})")
    print(f"  (Reference: LSTM vs True Weighted MSE (Training Metric): {avg_mse_lstm_vs_true_weighted:.4f})")
    print(f"  Fixed Alpha Used={fixed_alpha_for_metrics:.4f}")

    print(f"\n--- Detailed Metrics Per Step (LSTM speed prediction) ---")
    for k_step in range(pred_horizon):
        y_pred_lstm_step_k = y_pred_lstm_cat[:, k_step]
        y_true_step_k = y_true_cat[:, k_step]

        mse_step_lstm = nn.MSELoss()(y_pred_lstm_step_k, y_true_step_k).item()
        rmse_step_lstm = np.sqrt(mse_step_lstm)
        mae_step_lstm = torch.mean(torch.abs(y_pred_lstm_step_k - y_true_step_k)).item()

        # Step-wise MAPE
        abs_error_step = torch.abs(y_pred_lstm_step_k - y_true_step_k)
        abs_true_step = torch.abs(y_true_step_k)
        valid_mape_mask_step = abs_true_step > 1e-6
        mape_step_lstm = float('nan')
        if torch.sum(valid_mape_mask_step) > 0:
            mape_step_lstm = torch.mean(
                abs_error_step[valid_mape_mask_step] / abs_true_step[valid_mape_mask_step]
            ).item() * 100

        mse_step_idm = nn.MSELoss()(y_pred_idm_cat[:, k_step], y_true_step_k).item()

        print(f"  Step {k_step + 1}:")
        print(
            f"    LSTM Prediction -- MSE: {mse_step_lstm:.4f}, RMSE: {rmse_step_lstm:.4f}, MAE: {mae_step_lstm:.4f}, MAPE: {mape_step_lstm if not np.isnan(mape_step_lstm) else 'N/A'}%")
        print(f"    IDM (Reference) -- MSE: {mse_step_idm:.4f}")

    # Plotting first prediction step
    k_plot = 0
    plt.figure(figsize=(12, 7))
    plt.plot(y_true_cat[:100, k_plot].numpy(), '--o', label=f'True Value (Step {k_plot + 1})')
    plt.plot(y_pred_lstm_cat[:100, k_plot].numpy(), '-x', label=f'LSTM Prediction (Step {k_plot + 1}) (Final)')
    plt.plot(y_pred_idm_cat[:100, k_plot].numpy(), '-s', label=f'IDM Prediction (Step {k_plot + 1}) (Reference)')

    plt.title(f'Speed Prediction Comparison (First 100 samples, Step {k_plot + 1})')
    plt.xlabel("Sample Index")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid()
    plot_filename = os.path.join(results_dir, f"{dataset_name}_speed_comparison_LSTM_final_fixed_alpha.png")
    plt.savefig(plot_filename)
    print(f"Speed comparison plot saved to {plot_filename}")
    plt.close()

    return mse_val_overall, rmse_val_overall, mae_val_overall, mape_p_overall


# === Multi-step Position and Spacing Calculation (Using LSTM output) ===
def compute_position_and_spacing_and_save(model,
                                          test_loader,
                                          raw_data_all,
                                          label_data_all,
                                          train_size,
                                          pred_horizon,
                                          dt=0.1,
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
            y_lstm_k_mps, _ = model(batch_x_mps, batch_s_safe_initial_m, batch_v_lead_initial_mps)

            y_lstm_list_mps.append(y_lstm_k_mps.cpu())
            y_true_speeds_list_mps.append(batch_y_multistep_mps.cpu())

            batch_start_idx_in_loader = i * test_loader.batch_size
            current_batch_indices_in_all_data = np.arange(
                test_start_idx_in_all_data + batch_start_idx_in_loader,
                test_start_idx_in_all_data + batch_start_idx_in_loader + batch_x_mps.size(0)
            )

            # Data extraction for position calculation
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

    final_pred_speeds_ftps = final_pred_speeds_mps / 0.3048

    current_ego_pos_ft = initial_ego_pos_ft.clone()
    current_lead_pos_ft = initial_lead_pos_ft.clone()
    lead_speed_constant_ftps = initial_lead_speed_ftps

    # Recursive position calculation across prediction steps
    for k in range(pred_horizon):
        speed_ego_this_step_ftps = initial_ego_speed_ftps if k == 0 else final_pred_speeds_ftps[:, k - 1]

        disp_ego_ft = speed_ego_this_step_ftps * dt
        disp_lead_ft = lead_speed_constant_ftps * dt

        current_ego_pos_ft += disp_ego_ft
        current_lead_pos_ft += disp_lead_ft

        pred_ego_pos_k_steps_ft[:, k] = current_ego_pos_ft
        pred_lead_pos_k_steps_ft[:, k] = current_lead_pos_ft
        pred_spacing_k_steps_ft[:, k] = current_lead_pos_ft - current_ego_pos_ft

    # Conversion to meters for error reporting
    pred_ego_pos_m = pred_ego_pos_k_steps_ft.numpy() * 0.3048
    true_ego_pos_m = true_future_ego_pos_ft.numpy() * 0.3048
    pred_spacing_m = pred_spacing_k_steps_ft.numpy() * 0.3048
    true_spacing_m = true_future_spacing_ft.numpy() * 0.3048

    print(f"\n--- Multi-step Position and Spacing Evaluation (LSTM speed prediction) ---")
    for k_s in range(pred_horizon):
        rmse_Y_step = np.sqrt(np.mean((pred_ego_pos_m[:, k_s] - true_ego_pos_m[:, k_s]) ** 2))
        valid_Y_mask = np.abs(true_ego_pos_m[:, k_s]) > 1e-6
        mape_Y_step = np.mean(np.abs(
            (pred_ego_pos_m[valid_Y_mask, k_s] - true_ego_pos_m[valid_Y_mask, k_s]) / true_ego_pos_m[
                valid_Y_mask, k_s])) * 100 if np.sum(valid_Y_mask) > 0 else float('nan')

        rmse_sp_step = np.sqrt(np.mean((pred_spacing_m[:, k_s] - true_spacing_m[:, k_s]) ** 2))
        valid_sp_mask = np.abs(true_spacing_m[:, k_s]) > 1e-6
        mape_sp_step = np.mean(np.abs(
            (pred_spacing_m[valid_sp_mask, k_s] - true_spacing_m[valid_sp_mask, k_s]) / true_spacing_m[
                valid_sp_mask, k_s])) * 100 if np.sum(valid_sp_mask) > 0 else float('nan')

        print(f"  Step {k_s + 1}: Position RMSE: {rmse_Y_step:.4f} m, Spacing RMSE: {rmse_sp_step:.4f} m")

    rmse_p_overall = np.sqrt(np.mean((pred_ego_pos_m - true_ego_pos_m) ** 2))
    print(f"\n--- Overall Position RMSE: {rmse_p_overall:.4f} m ---")

    # Save results to Excel
    df_data = {}
    for k_idx in range(pred_horizon):
        df_data[f"Pred Speed LSTM (m/s) Step {k_idx + 1}"] = final_pred_speeds_mps[:, k_idx].numpy()
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

    print(f"Predictions for {dataset_name} saved to '{output_file}'.")
    return rmse_p_overall, 0.0  # MAPE summary simplified


# --- Metrics Storage Helpers ---
all_datasets_metrics_summary = []


def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse, pos_mape):
    metrics = {
        "Dataset": dataset_name,
        "Speed_MSE": speed_mse,
        "Speed_RMSE": speed_rmse,
        "Speed_MAE": speed_mae,
        "Speed_MAPE (%)": speed_mape,
        "Position_RMSE (m)": pos_rmse,
        "Position_MAPE (%)": pos_mape
    }
    all_datasets_metrics_summary.append(metrics)


def save_all_metrics_to_csv(filepath="evaluation_summary_LSTM_final.csv"):
    if all_datasets_metrics_summary:
        pd.DataFrame(all_datasets_metrics_summary).to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Summary metrics saved to {filepath}")


# --- Main Entry Point ---
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    if not data_files:
        print(f"No .mat files found in {DATA_DIR}. Exiting.")
        exit()

    position_predictions_excel_path = os.path.join(RESULTS_DIR, "pred_positions_all_datasets_LSTM_final.xlsx")

    for data_file_path in data_files:
        dataset_filename = os.path.basename(data_file_path)
        dataset_name_clean = dataset_filename.replace(".mat", "")
        print(f"\n==================== Processing: {dataset_filename} ====================")

        data = sio.loadmat(data_file_path)
        raw_all_ft = torch.tensor(data['train_data'], dtype=torch.float32)
        lab_all_ft = torch.tensor(data['lable_data'], dtype=torch.float32)

        # Preprocessing: ft to meters
        seq_mps = raw_all_ft[:, :, [0, 1, 2, 3, 5]].clone()
        seq_mps[:, :, [0, 1, 2, 3, 4]] *= 0.3048
        y_multistep_mps = lab_all_ft[:, :, 0].clone() * 0.3048

        current_pred_horizon = y_multistep_mps.shape[1]
        s_safe_initial_m = raw_all_ft[:, -1, 1].clone() * 0.3048
        v_lead_initial_mps = raw_all_ft[:, -1, 4].clone() * 0.3048

        # Sample fraction (20% for demonstration)
        N = int(seq_mps.size(0) * 0.2)
        train_size = int(N * 0.8)

        train_seq, test_seq = seq_mps[:train_size].to(device), seq_mps[train_size:N].to(device)
        train_y, test_y = y_multistep_mps[:train_size].to(device), y_multistep_mps[train_size:N].to(device)
        train_s, test_s = s_safe_initial_m[:train_size].to(device), s_safe_initial_m[train_size:N].to(device)
        train_v, test_v = v_lead_initial_mps[:train_size].to(device), v_lead_initial_mps[train_size:N].to(device)

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_seq, train_y, train_s, train_v),
                                                   batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_seq, test_y, test_s, test_v),
                                                  batch_size=32, shuffle=False)

        model = HybridIDMModel(train_seq.size(2), 128, output_dim=current_pred_horizon, num_layers=1).to(device)
        initialize_weights(model)

        print(f"Training model: {dataset_name_clean}...")
        model = train_model(model, train_loader, pred_horizon=current_pred_horizon, num_epochs=50,
                            alpha_decay_loss=0.05, lr_lstm=5e-4)

        print(f"Evaluating Speed: {dataset_name_clean}...")
        speed_mse, speed_rmse, speed_mae, speed_mape = evaluate_model(model, test_loader,
                                                                      pred_horizon=current_pred_horizon,
                                                                      dataset_name=dataset_name_clean,
                                                                      results_dir=RESULTS_DIR)

        print(f"Computing Positions: {dataset_name_clean}...")
        pos_rmse, pos_mape = compute_position_and_spacing_and_save(model, test_loader, raw_all_ft[:N], lab_all_ft[:N],
                                                                   train_size, pred_horizon=current_pred_horizon, dt=DT,
                                                                   output_file=position_predictions_excel_path,
                                                                   dataset_name=dataset_name_clean)

        store_dataset_metrics(dataset_name_clean, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse, pos_mape)

    save_all_metrics_to_csv(os.path.join(RESULTS_DIR, "evaluation_summary_final.csv"))
    print("\nProcessing complete for all datasets.")
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np

# Set environment variable to allow duplicate OpenMP libraries, avoiding conflicts in certain environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --- Data Validation Function ---
def check_data(data, name="data"):
    """
    Check if a PyTorch tensor contains NaN (Not a Number) or Inf (Infinity) values.
    :param data: PyTorch tensor to check.
    :param name: Data name for logging.
    """
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# --- Fixed IDM Parameter Prediction Function ---
def idm_fixed(v_n, s_safe, delta_v,
              v_desired=10.13701546, T=0.50290469, a_max=0.10995557,
              b_safe=4.98369406, delta=5.35419582, s0=0.10337701,
              delta_t=0.1):
    """
    Performs a single-step IDM (Intelligent Driver Model) prediction using a fixed parameter set.
    Calculates the speed for the next time step based on the current vehicle state (speed, gap, speed difference).

    :param v_n: Current vehicle speed (m/s) - Tensor.
    :param s_safe: Current actual headway/spacing (m) - Tensor.
    :param delta_v: Current speed difference (Lead speed - Subject speed, m/s) - Tensor.
    :param v_desired: Desired speed (m/s) - IDM parameter.
    :param T: Safe time headway (s) - IDM parameter.
    :param a_max: Maximum acceleration (m/s^2) - IDM parameter.
    :param b_safe: Comfortable deceleration (m/s^2) - IDM parameter.
    :param delta: Acceleration exponent (dimensionless) - IDM parameter.
    :param s0: Minimum static gap (m) - IDM parameter.
    :param delta_t: Time step (s) - Prediction interval.
    :return: Predicted speed for the next time step (m/s) - Tensor.
    """
    device = v_n.device  # Get device (CPU or CUDA) from input tensor
    # Convert IDM constant parameters to tensors on the same device
    # clamp(min=1e-6) prevents division by zero or invalid values
    a_max_t = torch.tensor(a_max, device=device, dtype=v_n.dtype).clamp(min=1e-6)
    b_safe_t = torch.tensor(b_safe, device=device, dtype=v_n.dtype).clamp(min=1e-6)
    s0_t = torch.tensor(s0, device=device, dtype=v_n.dtype)
    v_desired_t = torch.tensor(v_desired, device=device, dtype=v_n.dtype)
    T_t = torch.tensor(T, device=device, dtype=v_n.dtype)
    delta_param_t = torch.tensor(delta, device=device, dtype=v_n.dtype)
    delta_t_tensor = torch.tensor(delta_t, device=device, dtype=v_n.dtype)

    s_safe = s_safe.clamp(min=1e-6)  # Ensure gap is positive to prevent calculation errors

    # Calculate desired gap s* (s_star)
    s_star = s0_t + v_n * T_t + (v_n * delta_v) / (2 * torch.sqrt(a_max_t * b_safe_t) + 1e-6)
    s_star = s_star.clamp(min=0.0)  # Desired gap cannot be negative

    # Handle cases where v_desired might be zero
    v_n_ratio = torch.zeros_like(v_n)
    mask_v_desired_nonzero = v_desired_t.abs() > 1e-6
    if mask_v_desired_nonzero.any():
        v_n_ratio[mask_v_desired_nonzero] = (v_n[mask_v_desired_nonzero] / v_desired_t[mask_v_desired_nonzero])

    # Calculate acceleration term
    acceleration_term = a_max_t * (
            1 - v_n_ratio ** delta_param_t - (s_star / s_safe) ** 2
    )
    # Update speed based on acceleration
    v_follow = v_n + delta_t_tensor * acceleration_term
    return v_follow.clamp(min=0.0)  # Speed cannot be negative


# --- Hybrid Model Definition ---
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        """
        Hybrid model combining LSTM and IDM.
        :param input_dim: Feature dimension of LSTM input.
        :param hidden_dim: Dimension of LSTM hidden layers.
        :param num_layers: Number of LSTM layers.
        """
        super(HybridIDMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer maps LSTM output to predicted speed
        self.fc = nn.Linear(hidden_dim, 1)
        # Fixed alpha value, no longer a learnable parameter
        self.alpha = torch.tensor(0.7, dtype=torch.float32)  # Fixed alpha at 0.7

    def forward(self, x, s_safe):
        """
        Model Forward Pass.

        :param x: Input sequence, shape=(batch_size, seq_len, input_dim).
                  x[:, -1, 0] is current ego speed.
                  x[:, -1, 2] is current speed difference (Lead - Subject).
        :param s_safe: Current safety distance (m), shape=(batch_size,).
        :return: LSTM prediction output (batch_size,),
                 IDM prediction output (for internal calculation),
                 and fixed alpha value.
        """
        batch_size = x.size(0)
        # LSTM forward pass
        out, _ = self.lstm(x)  # out shape: (batch_size, seq_len, hidden_dim)
        # Get output of the last time step and map to speed via FC layer
        y_lstm = self.fc(out[:, -1, :]).squeeze(1)  # y_lstm shape: (batch_size,)

        # Calculate parameters required for fixed IDM prediction
        v_n = x[:, -1, 0]  # Current subject speed
        delta_v = x[:, -1, 2]  # Current speed difference
        # Calculate predicted speed using the fixed parameter IDM function
        y_idm = idm_fixed(v_n, s_safe, delta_v)

        # Final output uses LSTM prediction directly
        return y_lstm, y_idm, self.alpha.to(x.device)  # Ensure alpha is on the correct device


def initialize_weights(model):
    """
    Initialize model weights to improve training stability.
    :param model: PyTorch model to initialize.
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            # Use Xavier uniform initialization for weights
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            # Initialize biases to 0
            nn.init.constant_(param, 0)


# --- Training Function ---
def train_model(model, train_loader, optimizer, num_epochs=30, device='cpu'):
    """
    Train the model.

    :param model: Model to train.
    :param train_loader: Training data loader.
    :param optimizer: Optimizer.
    :param num_epochs: Number of training epochs.
    :param device: Target device ('cpu' or 'cuda:0').
    :return: Trained model.
    """
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y, batch_s_safe in train_loader:
            # Move data to target device
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_s_safe = batch_s_safe.to(device)

            optimizer.zero_grad()  # Zero out gradients

            # Forward pass; y_lstm is the final output
            y_lstm, y_idm, alpha_fixed = model(batch_x, batch_s_safe)

            # Loss function calculates MSE between y_lstm and target batch_y
            loss = 0.7 * (y_lstm - batch_y).pow(2).mean() + (1 - 0.7) * (y_lstm - y_idm).pow(2).mean()

            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters
            epoch_loss += loss.item()

        # Print average loss for the current epoch
        print(f"Epoch {epoch + 1}/{num_epochs}  Loss: {epoch_loss / len(train_loader):.6f}")
    return model


# --- Evaluation Function ---
def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model performance on the test set.

    :param model: Model to evaluate.
    :param test_loader: Test data loader.
    :param device: Target device ('cpu' or 'cuda:0').
    """
    model.eval()  # Set model to evaluation mode
    all_pred, all_true = [], []
    with torch.no_grad():  # Disable gradient calculation during evaluation
        for batch_x, batch_y, batch_s_safe in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_s_safe = batch_s_safe.to(device)

            # Forward pass
            y_lstm, y_idm, alpha_fixed = model(batch_x, batch_s_safe)
            # Collect results (move back to CPU for concatenation)
            all_pred.append(y_lstm.cpu())
            all_true.append(batch_y.cpu())

    y_pred = torch.cat(all_pred)
    y_true = torch.cat(all_true)

    # Compute metrics
    mse = nn.MSELoss()(y_pred, y_true).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    mae = torch.mean(torch.abs(y_pred - y_true)).item()

    print(f"\nTest Results -- MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:100].numpy(), '--o', label='True')  # First 100 samples
    plt.plot(y_pred[:100].numpy(), '-x', label='Pred')
    plt.title('Predicted vs. True Speed (First 100 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()


# --- Position and Spacing Calculation Function ---
def compute_position_and_spacing_and_save(model,
                                          test_data,
                                          test_real_speed,
                                          raw_data,
                                          label_data,
                                          train_size,
                                          test_s_safe,
                                          dt=0.1,
                                          output_file="predictions_extended.xlsx",
                                          device='cpu'):
    """
    Calculates future ego position and spacing based on predicted speed.
    Saves detailed results to an Excel file.
    """
    model.eval()
    with torch.no_grad():
        test_data = test_data.to(device)
        test_s_safe = test_s_safe.to(device)

        # Final output is directly y_lstm
        pred_speed_lstm, _, _ = model(test_data, test_s_safe)
        pred_speed = pred_speed_lstm.cpu().numpy()

    true_speed = test_real_speed.cpu().numpy()
    N_test = test_data.size(0)
    # Locate index in the original dataset
    idx = np.arange(train_size, train_size + N_test)

    # Extract current position (ft) and convert speed to ft/s for displacement calc
    current_Y_ft = raw_data[idx, -1, 4].numpy()
    current_speed_ftps = test_data[:, -1, 0].cpu().numpy() / 0.3048

    # Extract true future position and spacing from original labels (ft)
    true_Y_ft = label_data[idx, -1, 3].numpy()
    true_spacing_ft = label_data[idx, -1, 1].numpy()
    true_spacing_m = true_spacing_ft * 0.3048

    # Convert predicted speed to ft/s
    pred_speed_ftps = pred_speed / 0.3048

    # Calculate ego displacement (using average of current and predicted speed)
    disp_ft = ((current_speed_ftps + pred_speed_ftps) / 2) * dt

    # Predicted position for the next time step
    pred_Y_ft = current_Y_ft + disp_ft

    # Unit conversion: ft to m
    pred_Y_m = pred_Y_ft * 0.3048
    true_Y_m = true_Y_ft * 0.3048

    # Calculate predicted spacing
    pred_spacing_m = (true_Y_ft - pred_Y_ft) * 0.3048 + true_spacing_m

    # Error Metrics
    rmse_Y = np.sqrt(np.mean((pred_Y_m - true_Y_m) ** 2))
    valid_mask_Y = np.abs(true_Y_m) > 1e-6
    mape_Y = np.mean(np.abs((pred_Y_m[valid_mask_Y] - true_Y_m[valid_mask_Y]) / true_Y_m[valid_mask_Y])) * 100 if np.sum(valid_mask_Y) > 0 else float('nan')

    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m) ** 2))
    valid_mask_sp = np.abs(true_spacing_m) > 1e-6
    mape_sp = np.mean(np.abs((pred_spacing_m[valid_mask_sp] - true_spacing_m[valid_mask_sp]) / true_spacing_m[valid_mask_sp])) * 100 if np.sum(valid_mask_sp) > 0 else float('nan')

    print(f"Position Error -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y if not np.isnan(mape_Y) else 'N/A'}%")
    print(f" Spacing  Error -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp if not np.isnan(mape_sp) else 'N/A'}%")

    # Export to Excel
    df = pd.DataFrame({
        "Pred Speed (m/s)": pred_speed,
        "True Speed (m/s)": true_speed,
        "Predicted Y (m)": pred_Y_m,
        "True Y (m)": true_Y_m,
        "Pred Spacing (m)": pred_spacing_m,
        "True Spacing (m)": true_spacing_m,
    })
    sheet_name = "PID-LSTM-IDM"
    mode = "a" if os.path.exists(output_file) else "w"
    with pd.ExcelWriter(output_file, engine="openpyxl", mode=mode) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Results saved to '{output_file}' sheet '{sheet_name}'.")


# --- Main Execution Flow ---
if __name__ == "__main__":
    torch.manual_seed(42)

    # --- 1. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load MAT Data ---
    data_file_path = 'E:\\pythonProject1\\data_fine_0.1.mat'
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
        exit()

    data = sio.loadmat(data_file_path)
    raw = torch.tensor(data['train_data'], dtype=torch.float32)
    lab = torch.tensor(data['lable_data'], dtype=torch.float32)

    # --- 3. Preprocessing and Feature Selection ---
    seq = raw[:, -50:, [0, 1, 2, 3, -1]].clone()
    y = lab[:, -1, 0].clone().squeeze()
    s_safe = seq[:, -1, 1]

    # --- 4. Unit Conversion: ft/s to m/s ---
    seq *= 0.3048
    y *= 0.3048
    s_safe *= 0.3048

    # --- 5. Data Sampling (First 10% for quick example) ---
    N = int(seq.size(0) * 1)
    seq, y, s_safe = seq[:N], y[:N], s_safe[:N]
    print(f"Using {N} samples (first 10% of {seq.size(0)} total).")

    # --- 6. Dataset Split and Loader Preparation ---
    split_ratio = 0.8
    train_size = int(N * split_ratio)

    train_seq, test_seq = seq[:train_size], seq[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    train_s_safe, test_s_safe = s_safe[:train_size], s_safe[train_size:]

    train_ds = torch.utils.data.TensorDataset(train_seq, train_y, train_s_safe)
    test_ds = torch.utils.data.TensorDataset(test_seq, test_y, test_s_safe)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    # --- 7. Model Instantiation and Initialization ---
    input_dim = seq.size(2)
    hidden_dim = 128
    model = HybridIDMModel(input_dim, hidden_dim, num_layers=1).to(device)
    initialize_weights(model)

    # --- 8. Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # --- 9. Training & Evaluation ---
    print("\n--- Starting Model Training ---")
    model = train_model(model, train_loader, optimizer, num_epochs=100, device=device)
    print("\n--- Starting Model Evaluation ---")
    evaluate_model(model, test_loader, device=device)

    # --- 10. Position and Spacing Calculation ---
    print("\n--- Computing and Saving Position/Spacing Predictions ---")
    compute_position_and_spacing_and_save(
        model,
        test_seq,
        test_y,
        raw,
        lab,
        train_size,
        test_s_safe,
        dt=0.1,
        output_file="predictions_extended.xlsx",
        device=device
    )

    print("\nAll operations completed.")
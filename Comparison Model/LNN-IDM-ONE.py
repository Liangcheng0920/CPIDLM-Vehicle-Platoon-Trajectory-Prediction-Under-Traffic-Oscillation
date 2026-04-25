import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --- Data Validation Function ---
def check_data(data, name="data"):
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# --- Define Liquid Neural Network (LNN) Layer ---
class LNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, delta_t=0.1):
        super(LNNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.delta_t = delta_t

        # State update: pre-activation = W_xh x + W_hh h + b_h
        self.W_xh = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Time constant τ, initialized to 1, ensured positive via softplus
        self.log_tau = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        Returns: Final hidden state h_T [batch_size, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            xt = x[:, t, :]
            # Continuous-time dynamics: dh = ( -h + f(W_xh x + W_hh h + b) ) / τ * Δt
            preact = self.W_xh(xt) + self.W_hh(h)
            f = torch.tanh(preact)
            tau = nn.functional.softplus(self.log_tau) + 1e-3  # Prevent division by zero
            dh = (-h + f) / tau * self.delta_t
            h = h + dh

        return h  # [batch_size, hidden_dim]


# --- Define Hybrid Model (LNN replacing LSTM) ---
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(HybridIDMModel, self).__init__()
        # Using a single LNN layer; num_layers kept for structural compatibility
        self.lnn = LNNLayer(input_dim, hidden_dim, delta_t=0.1)
        self.fc = nn.Linear(hidden_dim, 6)  # IDM Parameters: [v_desired, T, a_max, b_safe, delta, s0]
        self.softplus = nn.Softplus()
        self.delta_t = 0.1  # Time step

    def forward(self, x):
        h = self.lnn(x)  # [batch, hidden_dim]
        params = self.fc(h)  # [batch, 6]
        params = self.softplus(params)  # Ensure parameters are positive
        return params

    def predict_speed(self, x, s_safe):
        params = self.forward(x)
        v_n, delta_v = x[:, -1, 0], x[:, -1, 2]
        v_desired, T, a_max, b_safe, delta, s0 = (
            params[:, 0], params[:, 1], params[:, 2],
            params[:, 3], params[:, 4], params[:, 5]
        )

        # IDM calculation for safe gap
        s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe))
        s_star = torch.clamp(s_star, min=0.0)

        # Predicted speed based on IDM acceleration formula
        v_follow = v_n + self.delta_t * a_max * (
                1 - (v_n / v_desired) ** delta - (s_star / s_safe) ** 2
        )
        predicted_speed = torch.clamp(v_follow, min=0.0)
        return predicted_speed, params


# --- Initialize Weights and Biases ---
def initialize_weights(model):
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            # Apply Xavier initialization for weight matrices
            nn.init.xavier_uniform_(param)
        else:
            # Initialize biases or 1D parameters to zero
            nn.init.constant_(param, 0)


# --- Model Training (with Gradient Clipping) ---
def train_model(model, train_loader, optimizer, criterion, num_epochs=30, clip_norm=1.0):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_data, batch_speed, batch_s_safe in train_loader:
            optimizer.zero_grad()
            predicted_speed, _ = model.predict_speed(batch_data, batch_s_safe)
            loss = criterion(predicted_speed, batch_speed)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return model


# --- Model Evaluation ---
def evaluate_model(model, test_loader):
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0.0
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch_idx, (batch_data, batch_speed, batch_s_safe) in enumerate(test_loader):
            pred_speed, params = model.predict_speed(batch_data, batch_s_safe)
            total_mse += mse_loss(pred_speed, batch_speed).item()
            all_pred.append(pred_speed)
            all_true.append(batch_speed)

            print(f"\n--- Batch {batch_idx + 1} Predicted IDM Parameters ---")
            for i, name in enumerate(["v_desired", "T", "a_max", "b_safe", "delta", "s0"]):
                print(f"{name}: {params[:, i].cpu().numpy()}")

    mse = total_mse / len(test_loader)
    rmse = torch.sqrt(torch.tensor(mse))
    mae = torch.mean(torch.abs(torch.cat(all_pred) - torch.cat(all_true))).item()
    print(f"\nEvaluation Metrics -- MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(torch.cat(all_true)[:100].cpu().numpy(), '--o', label='True Speed')
    plt.plot(torch.cat(all_pred)[:100].cpu().numpy(), '-x', label='Predicted Speed')
    plt.title('True vs Predicted Speed (Hybrid IDM-LNN)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()


# --- Save Predictions to CSV ---
def save_predictions_to_csv(model, test_loader, output_file="predictions.csv"):
    model.eval()
    all_true, all_pred, all_params = [], [], []
    with torch.no_grad():
        for batch_data, batch_speed, batch_s_safe in test_loader:
            pred_speed, params = model.predict_speed(batch_data, batch_s_safe)
            all_true.append(batch_speed.cpu())
            all_pred.append(pred_speed.cpu())
            all_params.append(params.cpu())

    true_np = torch.cat(all_true).numpy()
    pred_np = torch.cat(all_pred).numpy()
    params_np = torch.cat(all_params).numpy()

    df = pd.DataFrame({
        "True Speed": true_np,
        "Pred Speed": pred_np,
        "v_desired": params_np[:, 0],
        "T": params_np[:, 1],
        "a_max": params_np[:, 2],
        "b_safe": params_np[:, 3],
        "delta": params_np[:, 4],
        "s0": params_np[:, 5],
    })
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# --- Compute Position, Spacing and Save to Excel ---
def compute_position_and_spacing_and_save(model,
                                          test_data,
                                          test_real_speed,
                                          raw_data,
                                          label_data,
                                          train_size,
                                          test_s_safe,
                                          dt=0.1,
                                          output_file="predictions_extended.xlsx"
                                          ):
    model.eval()
    with torch.no_grad():
        pred_speed_tensor, params_tensor = model.predict_speed(test_data, test_s_safe)
        pred_speed = pred_speed_tensor.squeeze().cpu().numpy()

    true_speed = test_real_speed.cpu().numpy()
    N_test = test_data.shape[0]
    idx = np.arange(train_size, train_size + N_test)

    # Convert current Y position to meters
    current_Y_ft = raw_data[idx, -1, 4].numpy()  # 4th dimension index
    current_Y_ft *= 0.3048
    current_speed_m = test_data[:, -1, 0].numpy()

    # Get true Y position and spacing in meters
    true_Y_ft = label_data[idx, -1, 3].numpy()
    true_Y_ft *= 0.3048
    true_spacing_m = label_data[idx, -1, 1].numpy()
    true_spacing_m *= 0.3048

    # Calculate displacement based on predicted speed (using constant acceleration approximation)
    disp_m = current_speed_m * dt + 0.5 * ((pred_speed - current_speed_m) / dt) * dt ** 2
    pred_Y_m = current_Y_ft + disp_m
    true_Y_m = true_Y_ft

    # Calculate predicted spacing
    pred_spacing_m = (true_Y_ft - pred_Y_m) + true_spacing_m

    # Error calculation
    rmse_Y = np.sqrt(np.mean((pred_Y_m - true_Y_m) ** 2))
    mape_Y = np.mean(np.abs((pred_Y_m - true_Y_m) / true_Y_m)) * 100
    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m) ** 2))
    mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m) / true_spacing_m)) * 100

    print(f"Position Error    -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
    print(f"Spacing  Error    -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")

    df = pd.DataFrame({
        "Pred Speed (m/s)": pred_speed.flatten(),
        "True Speed m(m/s)": true_speed.flatten(),
        "Predicted Y (m)": pred_Y_m.flatten(),
        "True Y (m)": true_Y_m.flatten(),
        "Predicted Spacing (m)": pred_spacing_m.flatten(),
        "True Spacing (m)": true_spacing_m.flatten(),
    })

    sheet_name = "LNN-IDM_1"  # Target worksheet name

    # Append sheet if file exists, otherwise create new file
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a" if os.path.exists(output_file) else "w") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"New model results saved to '{output_file}' in sheet '{sheet_name}'.")


# --- Main Logic ---
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load and preprocess data
    data = sio.loadmat('E:\\pythonProject1\\data_fine_0.1.mat')
    raw_data = torch.tensor(data['train_data'], dtype=torch.float32)
    label_data = torch.tensor(data['lable_data'], dtype=torch.float32)

    # Select specific features: [0, 1, 2, 3, -1] and time steps
    train_data = raw_data[:, -50:, [0, 1, 2, 3, -1]]
    train_real_speed = label_data[:, -1, 0]
    train_s_safe = train_data[:, -1, 1].clone()

    # Unit conversion: feet to meters
    train_data *= 0.3048
    train_real_speed *= 0.3048
    train_s_safe *= 0.3048

    # Use first 10% of data for the sample
    sample_size = int(train_data.shape[0] * 1)
    train_data = train_data[:sample_size]
    train_real_speed = train_real_speed[:sample_size]
    train_s_safe = train_s_safe[:sample_size]

    # Data Check
    check_data(train_data, "train_data")
    check_data(train_real_speed, "train_real_speed")
    check_data(train_s_safe, "train_s_safe")

    # Split into training and testing sets
    dataset_size = train_data.shape[0]
    train_size = int(dataset_size * 0.8)
    test_size = dataset_size - train_size

    train_set = torch.utils.data.TensorDataset(
        train_data[:train_size],
        train_real_speed[:train_size],
        train_s_safe[:train_size]
    )
    test_set = torch.utils.data.TensorDataset(
        train_data[train_size:],
        train_real_speed[train_size:],
        train_s_safe[train_size:]
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    input_dim, hidden_dim = train_data.shape[2], 128
    model = HybridIDMModel(input_dim, hidden_dim, num_layers=1)
    initialize_weights(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Training & Evaluation
    model = train_model(model, train_loader, optimizer, criterion, num_epochs=100, clip_norm=1.0)
    evaluate_model(model, test_loader)

    # Extended position and spacing calculation
    compute_position_and_spacing_and_save(
        model,
        train_data[train_size:],
        train_real_speed[train_size:],
        raw_data,
        label_data,
        train_size,
        train_s_safe[train_size:],
        dt=0.1,
        output_file="predictions_extended.xlsx"
    )
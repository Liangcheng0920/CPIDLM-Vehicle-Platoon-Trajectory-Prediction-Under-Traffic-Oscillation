import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ------------------------------
# Data Validation Function
def check_data(data, name="data"):
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# ------------------------------
# Liquid Cell Module
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
        # dh = -h + f(Wh*h + Wu*u + b)
        dh = -h + self.activation(self.W_h(h) + self.W_u(u) + self.bias)
        h_new = h + self.dt * dh
        return h_new


# ------------------------------
# Multi-step Prediction Liquid Neural Network
class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1, num_steps=50):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.prediction_steps = prediction_steps

        self.liquid_cells = nn.ModuleList([
            LiquidCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, prediction_steps)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        T = min(seq_len, self.num_steps)
        h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t, :]
            for i, cell in enumerate(self.liquid_cells):
                h[i] = cell(inp if i == 0 else h[i - 1], h[i])

        out = self.fc(h[-1])
        return out

    def predict_speed(self, x):
        return self.forward(x)


# ------------------------------
# Weight Initialization
def initialize_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)


# ------------------------------
# Training Function
def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed in train_loader:
            optimizer.zero_grad()
            pred = model.predict_speed(batch_data)
            loss = criterion(pred, batch_speed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    return model


# ------------------------------
# Evaluation Function
def evaluate_model(model, train_real_speed, test_loader):
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    all_predicted = []
    all_true = []

    with torch.no_grad():
        for batch_idx, (batch_data, batch_speed) in enumerate(test_loader):
            predicted_speed = model.predict_speed(batch_data)
            loss = mse_loss(predicted_speed, batch_speed)
            total_mse += loss.item()
            all_predicted.append(predicted_speed)
            all_true.append(batch_speed)

    # Concatenate predictions and ground truth across all batches
    all_predicted = torch.cat(all_predicted).cpu().numpy()  # shape: (N, prediction_steps)
    all_true = torch.cat(all_true).cpu().numpy()  # shape: (N, prediction_steps)

    # Calculate global evaluation metrics
    mse_val = np.mean((all_predicted - all_true) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted - all_true))
    print(f"Evaluation Metrics:\nMSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")

    # Concatenate data for visualization (30 samples)
    step = train_real_speed.shape[1]
    num_samples = 30
    true_concat = []
    pred_concat = []

    for i in range(num_samples):
        idx = i * step
        if idx >= all_true.shape[0]:
            break
        # Append true and predicted values for each sample
        true_concat.extend(all_true[idx])
        pred_concat.extend(all_predicted[idx])

    true_concat = np.array(true_concat)
    pred_concat = np.array(pred_concat)

    # Plot the concatenated curves
    plt.figure(figsize=(12, 8))
    plt.plot(true_concat, linestyle='--', marker='o', label='True')
    plt.plot(pred_concat, linestyle='-', marker='x', label='Predicted')

    plt.title('True vs Predicted Speed for 30 Samples (Step 1-150)')
    plt.xlabel('Time Step')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()


# ------------------------------
# Save Speed Predictions to CSV
def save_predictions_to_csv(model, test_loader, output_file="predictions.csv"):
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for bd, bt in test_loader:
            p = model.predict_speed(bd)
            all_t.append(bt.cpu())
            all_p.append(p.cpu())
    all_t = torch.cat(all_t).numpy()
    all_p = torch.cat(all_p).numpy()
    df = pd.DataFrame({
        **{f"True_Step{j + 1}": all_t[:, j] for j in range(all_t.shape[1])},
        **{f"Pred_Step{j + 1}": all_p[:, j] for j in range(all_p.shape[1])}
    })
    df.to_csv(output_file, index=False)
    print(f"Speeds saved to {output_file}")


# ------------------------------
# Calculate and Save Future Positions of Leading Vehicle
def compute_future_positions_and_save(model,
                                      test_data,
                                      raw_data,
                                      label_data,
                                      train_size,
                                      dt=0.1,
                                      output_file="pred_positions.xlsx"):
    model.eval()
    with torch.no_grad():
        pred_speeds = model.predict_speed(test_data).cpu().numpy()  # (N_test, steps)

    N_test, steps = pred_speeds.shape
    idx = np.arange(train_size, train_size + N_test)

    # Current speed (m/s)
    curr_speed = test_data[:, -1, 0].cpu().numpy()
    # Current lead vehicle position (ft -> m)
    curr_pos_ft = raw_data[idx, -1, 7].cpu().numpy()
    curr_pos_m = curr_pos_ft * 0.3048

    # Ground truth future positions (ft -> m), label_data[:, :, 5] stores lead vehicle future positions
    true_pos_ft = label_data[idx, :, 5].cpu().numpy()
    true_pos_m = true_pos_ft * 0.3048

    # Ground truth speeds (m/s), label_data[:, :, 4] stores lead vehicle speeds
    true_speeds = label_data[idx, :, 4].cpu().numpy()

    # Initialize storage for predicted positions
    pred_pos_m = np.zeros((N_test, steps))

    # Calculate future positions using recursive kinematics
    for i in range(N_test):
        prev_speed = curr_speed[i]
        prev_pos = curr_pos_m[i]

        for k in range(steps):
            v_pred = pred_speeds[i, k]  # Predicted speed for current sample at step k
            # Use kinematic equations for displacement
            a = (v_pred - prev_speed) / dt  # Calculate acceleration
            disp = prev_speed * dt + 0.5 * a * dt ** 2  # Calculate displacement
            pos = prev_pos + disp  # Update position
            pred_pos_m[i, k] = pos
            prev_speed = v_pred  # Update for next iteration
            prev_pos = pos

    # Error evaluation
    rmse_p = np.sqrt(np.mean((pred_pos_m - true_pos_m) ** 2))
    mape_p = np.mean(np.abs((pred_pos_m - true_pos_m) / true_pos_m)) * 100
    print(f"Future Position Error -- RMSE: {rmse_p:.4f} m, MAPE: {mape_p:.2f}%")

    # Save results to Excel
    data_dict = {}
    for i in range(steps):
        data_dict[f"Pred_Speed_step{i + 1}(m/s)"] = pred_speeds[:, i]
        data_dict[f"Pred_Pos_step{i + 1}(m)"] = pred_pos_m[:, i]
        data_dict[f"True_Pos_step{i + 1}(m)"] = true_pos_m[:, i]
        data_dict[f"True_Speed_step{i + 1}(m/s)"] = true_speeds[:, i]

    df_pos = pd.DataFrame(data_dict)
    sheet_name = "LNN_1"  # Target sheet name

    # Append to existing file or create new one
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a" if os.path.exists(output_file) else "w") as writer:
        df_pos.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"New model results saved to '{output_file}' in sheet '{sheet_name}'.")


# ------------------------------
# Main Execution
if __name__ == "__main__":
    # Load data from .mat file
    data = sio.loadmat('E:\\pythonProject1\\data_ngsim\\data_10.mat')
    raw_data = torch.tensor(data['train_data'], dtype=torch.float32)
    label_data = torch.tensor(data['lable_data'], dtype=torch.float32)

    # Extract multi-step speed labels (Index 0 in label_data)
    train_real_speed_all = label_data[:, :, 0]
    print("train_real_speed_all shape:", train_real_speed_all.shape)

    # Construct multi-step input: Last 50 steps of specific features [0,1,2,3,5]
    train_data = raw_data[:, -50:, [0, 1, 2, 3, 5]].clone() * 0.3048
    train_real_speed = train_real_speed_all.clone()
    print("train_data shape:", train_data.shape)

    # Unit conversion: ft/s -> m/s
    train_real_speed *= 0.3048

    # Full sampling (set to 1.0)
    total_samples = train_data.shape[0]
    sample_size = int(total_samples * 1.0)
    train_data = train_data[:sample_size]
    train_real_speed = train_real_speed[:sample_size]

    # Data integrity check
    check_data(train_data, "train_data")
    check_data(train_real_speed, "train_real_speed")

    # Split Train/Test sets (80/20)
    dataset_size = train_data.shape[0]
    train_size = int(dataset_size * 0.8)
    test_data = train_data[train_size:]
    test_real_speed = train_real_speed[train_size:]
    train_data = train_data[:train_size]
    train_real_speed = train_real_speed[:train_size]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_real_speed),
        batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data, test_real_speed),
        batch_size=32, shuffle=False
    )

    # Model Configuration
    input_dim = train_data.shape[2]
    hidden_dim = 128
    prediction_steps = train_real_speed.shape[1]
    model = LiquidNeuralNetwork(input_dim, hidden_dim, prediction_steps,
                                num_layers=1, num_steps=train_data.shape[1])
    initialize_weights(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Training & Evaluation
    model = train_model(model, train_loader, optimizer, criterion, num_epochs=10)
    evaluate_model(model, train_real_speed, test_loader)
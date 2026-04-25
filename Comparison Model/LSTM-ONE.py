import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np

# Resolve multi-threading conflicts for macOS / some specific environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --- Data Validation Function ---
def check_data(data, name="data"):
    """
    Check for NaN or Inf values within a tensor
    """
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# --- Define LSTM Model (Predicts speed for the next time step only) ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Taking the hidden state of the last time step
        return self.fc(out[:, -1, :])

    def predict_speed(self, x):
        return self.forward(x)


# --- Initialize Model Weights and Biases ---
def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)


# --- Train Model ---
def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_data, batch_speed in train_loader:
            optimizer.zero_grad()
            pred = model.predict_speed(batch_data)
            loss = criterion(pred.squeeze(), batch_speed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}]  Loss: {avg_loss:.4f}")
    return model


# --- Evaluate Speed Prediction Performance on Test Set ---
def evaluate_model(model, test_loader):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch_data, batch_speed in test_loader:
            pred = model.predict_speed(batch_data).squeeze()
            all_pred.append(pred.cpu())
            all_true.append(batch_speed.cpu())

    all_pred = torch.cat(all_pred).numpy()
    all_true = torch.cat(all_true).numpy()

    mse_val = np.mean((all_pred - all_true) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_pred - all_true))
    mape_speed = np.mean(np.abs((all_pred - all_true) / all_true)) * 100

    print(f"Speed Prediction -- MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_speed:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(all_true[:100], linestyle='--', marker='o', label='True Speed')
    plt.plot(all_pred[:100], linestyle='-', marker='x', label='Pred Speed')
    plt.title('True vs Predicted Speed (LSTM)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid(True)
    plt.show()


# --- Calculate next Y position and spacing based on predicted speed, then save results ---
def compute_position_and_spacing_and_save(model,
                                          test_data,
                                          test_real_speed,
                                          raw_data,
                                          label_data,
                                          train_size,
                                          dt=0.1,
                                          output_file="LSTM_Final_1.xlsx"):
    model.eval()
    with torch.no_grad():
        pred_speed = model.predict_speed(test_data).squeeze().cpu().numpy()
    true_speed = test_real_speed.cpu().numpy()

    N_test = test_data.shape[0]
    idx = np.arange(train_size, train_size + N_test)

    # Current Y position in feet, convert to meters (index 4)
    current_Y_ft = raw_data[idx, -1, 4].numpy()
    current_Y_ft *= 0.3048
    current_speed_m = test_data[:, -1, 0].numpy()

    # Ground truth Y and Spacing in feet, convert to meters
    true_Y_ft = label_data[idx, -1, 3].numpy()
    true_Y_ft *= 0.3048
    true_spacing_m = label_data[idx, -1, 1].numpy()
    true_spacing_m *= 0.3048

    # Calculate displacement using kinematic formula: s = v0*t + 0.5*a*t^2
    # Where a = (v_pred - v_current) / dt
    disp_m = current_speed_m * dt + 0.5 * ((pred_speed - current_speed_m) / dt) * dt ** 2
    pred_Y_m = current_Y_ft + disp_m

    true_Y_m = true_Y_ft
    # Predicted Spacing = (True_Y_Leading_Vehicle - Pred_Y_Following_Vehicle)
    pred_spacing_m = (true_Y_ft - pred_Y_m) + true_spacing_m

    # Error calculation
    rmse_Y = np.sqrt(np.mean((pred_Y_m - true_Y_m) ** 2))
    mape_Y = np.mean(np.abs((pred_Y_m - true_Y_m) / true_Y_m)) * 100
    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m) ** 2))
    mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m) / true_spacing_m)) * 100

    print(f"Position Error    -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
    print(f"Spacing  Error    -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")

    df = pd.DataFrame({
        "Pred Speed (m/s)": pred_speed,
        "True Speed (m/s)": true_speed,
        "Predicted Y (m)": pred_Y_m,
        "True Y (m)": true_Y_m,
        "Predicted Spacing (m)": pred_spacing_m,
        "True Spacing (m)": true_spacing_m,
    })

    sheet_name = "LSTM_1"  # Define worksheet name

    # Always open in write mode; if the file exists, it will be overwritten
    with pd.ExcelWriter(output_file, engine="openpyxl", mode='w') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Results saved to '{output_file}' in sheet '{sheet_name}'.")


# --- Main Entry Point ---
if __name__ == "__main__":
    torch.manual_seed(42)

    # --- Load Data ---
    mat = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')
    raw_data = torch.tensor(mat['train_data'], dtype=torch.float32)
    label_data = torch.tensor(mat['lable_data'], dtype=torch.float32)

    # Construct training inputs: selecting specific columns [0, 1, 2, 3, -1]
    train_data = raw_data[:, -50:, [0, 1, 2, 3, -1]].clone()
    train_labels = label_data[:, :, 0].clone()

    # Conversion from feet to meters (0.3048)
    train_data *= 0.3048
    train_real_speed = train_labels[:, -1] * 0.3048

    # --- Sample 10% of data to speed up the experiment ---
    total_samples = train_data.shape[0]
    sample_size = int(total_samples * 1)
    print(f"Total samples: {total_samples}, using only first {sample_size} samples for quick run.")
    train_data = train_data[:sample_size]
    train_real_speed = train_real_speed[:sample_size]

    # Split Train/Test sets (80% / 20%)
    N = train_data.shape[0]
    train_size = int(N * 0.8)
    test_size = N - train_size

    train_x = train_data[:train_size]
    test_x = train_data[train_size:]
    train_y = train_real_speed[:train_size]
    test_y = train_real_speed[train_size:]

    # DataLoader setup
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=batch_size, shuffle=False
    )

    # Model, Loss, and Optimizer
    input_dim = train_x.shape[2]
    hidden_dim = 128
    model = LSTMModel(input_dim, hidden_dim, num_layers=1)
    initialize_weights(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Train & Evaluate
    model = train_model(model, train_loader, optimizer, criterion, num_epochs=100)
    evaluate_model(model, test_loader)

    # Position & Spacing prediction and save results
    compute_position_and_spacing_and_save(
        model,
        test_x,
        test_y,
        raw_data,
        label_data,
        train_size,
        dt=0.1,
        output_file="LSTM_Final_1.xlsx"
    )
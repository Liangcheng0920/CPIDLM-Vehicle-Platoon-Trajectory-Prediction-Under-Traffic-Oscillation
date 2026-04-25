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
# This cell simulates continuous-time dynamics, updating state via Euler integration:
# h_new = h + dt * (-h + tanh(W_h * h + W_u * u + b))
class LiquidCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        """
        :param input_dim: Current input feature count (raw input for 1st layer, hidden state of prev layer for subsequent)
        :param hidden_dim: Dimension of the hidden state
        :param dt: Time step size used for Euler integration updates
        """
        super(LiquidCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_u = nn.Linear(input_dim, hidden_dim)
        # Use a separate bias parameter
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.activation = nn.Tanh()

    def forward(self, u, h):
        # u: (batch, input_dim); h: (batch, hidden_dim)
        dh = -h + self.activation(self.W_h(h) + self.W_u(u) + self.bias)
        h_new = h + self.dt * dh
        return h_new


# ------------------------------
# Liquid Neural Network Model
# Iteratively updates hidden states along time steps using Liquid Cells to produce predictions.
class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_steps=50, output_dim=1):
        """
        :param input_dim: Number of input features
        :param hidden_dim: Dimension of the hidden state
        :param num_layers: Number of liquid layers (stacked Liquid Cells)
        :param num_steps: Number of time steps in sequence used for state updates
        :param output_dim: Output dimension (e.g., 1D for speed prediction)
        """
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_steps

        # Construct multiple liquid cells; first cell takes raw input, others take previous layer's hidden state
        self.liquid_cells = nn.ModuleList([
            LiquidCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # Final fully connected layer for output prediction
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        :param x: Input data, shape (batch_size, seq_len, input_dim)
        :return: Predicted results, shape (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        T = min(seq_len, self.num_steps)  # Use the first T steps of the sequence for state updates

        # Initialize hidden states for each layer as zero vectors
        h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

        # Iteratively update hidden states along time steps
        for t in range(T):
            input_t = x[:, t, :]  # Current time step input
            for i, cell in enumerate(self.liquid_cells):
                # First layer takes raw input; subsequent layers take the output of the previous layer
                if i == 0:
                    h[i] = cell(input_t, h[i])
                else:
                    h[i] = cell(h[i - 1], h[i])

        # Use the hidden state of the final layer as the global representation
        out = self.fc(h[-1])
        return out

    def predict_speed(self, x):
        return self.forward(x)


# ------------------------------
# Weight Initialization Function
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
# Model Training Function
def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed in train_loader:
            optimizer.zero_grad()
            predicted_speed = model.predict_speed(batch_data)
            loss = criterion(predicted_speed, batch_speed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    return model


# ------------------------------
# Model Evaluation Function
def evaluate_model(model, test_loader):
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

    # Concatenate predicted and true values for all batches
    all_predicted = torch.cat(all_predicted).cpu().numpy()
    all_true = torch.cat(all_true).cpu().numpy()

    mse_val = np.mean((all_predicted - all_true) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted - all_true))
    print(f"Evaluation Metrics:\nMSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(all_true[:100], label='True Speed', linestyle='--', marker='o')
    plt.plot(all_predicted[:100], label='Predicted Speed', linestyle='-', marker='x')
    plt.title('True vs Predicted Speed (Liquid Neural Network)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()


# ------------------------------
# Save Predictions to CSV
def save_predictions_to_csv(model, test_loader, output_file="predictions.csv"):
    model.eval()
    all_true_speeds = []
    all_predicted_speeds = []

    with torch.no_grad():
        for batch_data, batch_speed in test_loader:
            predicted_speed = model.predict_speed(batch_data)
            all_true_speeds.append(batch_speed.cpu())
            all_predicted_speeds.append(predicted_speed.cpu())

    all_true_speeds = torch.cat(all_true_speeds).numpy()
    all_predicted_speeds = torch.cat(all_predicted_speeds).numpy()

    result_data = pd.DataFrame({
        "True Speed": all_true_speeds.flatten(),
        "Predicted Speed": all_predicted_speeds.flatten(),
    })

    result_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# --- Calculate next Y-coordinate and spacing based on predicted speed, then save results ---
def compute_position_and_spacing_and_save(model,
                                          test_data,
                                          test_real_speed,
                                          raw_data,
                                          label_data,
                                          train_size,
                                          dt=0.1,
                                          output_file="predictions_extended.xlsx"):
    model.eval()
    with torch.no_grad():
        pred_speed = model.predict_speed(test_data).squeeze().cpu().numpy()
    true_speed = test_real_speed.cpu().numpy()

    N_test = test_data.shape[0]
    idx = np.arange(train_size, train_size + N_test)

    # current_Y position in feet, convert to meters (4th index)
    current_Y_ft = raw_data[idx, -1, 4].numpy()
    current_Y_ft *= 0.3048
    current_speed_m = test_data[:, -1, 0].numpy()

    # Ground truth position and spacing
    true_Y_ft = label_data[idx, -1, 3].numpy()
    true_Y_ft *= 0.3048
    true_spacing_m = label_data[idx, -1, 1].numpy()
    true_spacing_m *= 0.3048

    # Calculate displacement using kinematic equation
    disp_m = current_speed_m * dt + 0.5 * ((pred_speed - current_speed_m) / dt) * dt ** 2
    pred_Y_m = current_Y_ft + disp_m

    true_Y_m = true_Y_ft
    # Calculate predicted spacing
    pred_spacing_m = (true_Y_ft - pred_Y_m) + true_spacing_m

    # Compute errors
    rmse_Y = np.sqrt(np.mean((pred_Y_m - true_Y_m) ** 2))
    mape_Y = np.mean(np.abs((pred_Y_m - true_Y_m) / true_Y_m)) * 100
    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m) ** 2))
    mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m) / true_spacing_m)) * 100

    print(f"Position Error    -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
    print(f"Spacing  Error    -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")

    df = pd.DataFrame({
        "Pred Speed (m/s)": pred_speed.flatten(),
        "True Speed (m/s)": true_speed.flatten(),
        "Predicted Y (m)": pred_Y_m.flatten(),
        "True Y (m)": true_Y_m.flatten(),
        "Predicted Spacing (m)": pred_spacing_m.flatten(),
        "True Spacing (m)": true_spacing_m.flatten(),
    })
    sheet_name = "LNN_1"  # Target worksheet name

    # Append to existing file or create new
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a" if os.path.exists(output_file) else "w") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"New model results saved to '{output_file}' in sheet '{sheet_name}'.")


# ------------------------------
# Main Function
if __name__ == "__main__":
    torch.manual_seed(42)
    # Load MAT file
    mat = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')
    raw_data = torch.tensor(mat['train_data'], dtype=torch.float32)
    label_data = torch.tensor(mat['lable_data'], dtype=torch.float32)

    # Construct training inputs: selecting specific feature columns [0, 1, 2, 3, -1]
    train_data = raw_data[:, -50:, [0, 1, 2, 3, -1]]
    train_real_speed = label_data[:, :, 0]

    # Unit conversion: feet to meters
    train_data[:, :, 0] *= 0.3048
    train_data[:, :, 1] *= 0.3048
    train_data[:, :, 2] *= 0.3048
    train_data[:, :, 3] *= 0.3048
    train_real_speed *= 0.3048

    # --- Sample 10% to speed up experimentation ---
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

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=batch_size, shuffle=False
    )

    # Model Parameters
    input_dim = train_data.shape[2]  # Feature count
    hidden_dim = 128  # Hidden state dimension
    num_layers = 1  # Stacked layers
    num_steps = train_data.shape[1]  # Steps for iteration

    # Initialize LNN model
    model = LiquidNeuralNetwork(input_dim, hidden_dim, num_layers=num_layers, num_steps=num_steps, output_dim=1)
    initialize_weights(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Train model
    num_epochs = 100
    model = train_model(model, train_loader, optimizer, criterion, num_epochs)

    # Evaluate model
    evaluate_model(model, test_loader)

    # Position & Spacing prediction and save
    compute_position_and_spacing_and_save(
        model,
        test_x,
        test_y,
        raw_data,
        label_data,
        train_size,
        dt=0.1,
        output_file="predictions_extended.xlsx"
    )
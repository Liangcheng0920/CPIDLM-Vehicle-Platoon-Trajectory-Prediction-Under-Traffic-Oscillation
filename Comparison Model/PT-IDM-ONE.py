import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np

# Resolve potential conflicts between Intel MKL and PyTorch libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --- Data Validation Function ---
def check_data(data, name="data"):
    """
    Checks if a PyTorch tensor contains any NaN or Inf values.

    Args:
        data (torch.Tensor): The tensor to be checked.
        name (str): Identifier name for logging purposes.
    """
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# --- Hybrid Model Definition (Transformer + IDM) ---
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        """
        A hybrid model combining a Transformer Encoder and the Intelligent Driver Model (IDM).

        Args:
            input_dim (int): Input feature dimension.
            model_dim (int): Hidden dimension for the Transformer.
            num_heads (int): Number of heads in Multi-Head Attention.
            num_layers (int): Number of Transformer Encoder layers.
            dropout (float): Dropout probability.
        """
        super(HybridIDMModel, self).__init__()
        self.model_dim = model_dim
        # Linear layer to project input to model_dim
        self.input_linear = nn.Linear(input_dim, model_dim)
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # FC layer to output IDM parameters: [v_desired, T, a_max, b_safe, delta, s0]
        self.fc = nn.Linear(model_dim, 6)
        self.softplus = nn.Softplus()  # Ensures parameters stay positive
        self.delta_t = 0.1  # Time step (seconds)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input sequence data, shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Predicted IDM parameters.
        """
        # Map input dimension to Transformer dimension
        x = self.input_linear(x)
        # Pass through Transformer encoder
        out = self.transformer_encoder(x)
        # Use the output of the last time step to predict IDM parameters
        params = self.fc(out[:, -1, :])
        params = self.softplus(params)  # Constraint to non-negative space
        return params

    def predict_speed(self, x, s_safe):
        """
        Predicts velocity at the next time step using the IDM formula.

        Args:
            x (torch.Tensor): Input data, shape (batch_size, seq_len, input_dim).
                              x[:, -1, 0] = current velocity v_n
                              x[:, -1, 2] = velocity difference delta_v
            s_safe (torch.Tensor): Actual head gap (s_safe), shape (batch_size,).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted speed and IDM parameters.
        """
        params = self.forward(x)
        v_n = x[:, -1, 0]  # Current velocity
        delta_v = x[:, -1, 2]  # Velocity difference (v_leader - v_ego)

        # Extract IDM parameters
        v_desired, T, a_max, b_safe, delta, s0 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:,
                                                                                                         4], params[:,
                                                                                                             5]

        # IDM Core Logic: Calculation of desired dynamic gap (s_star)
        # Formula: $s^* = s_0 + v_n T + \frac{v_n \Delta v}{2 \sqrt{a_{max} b_{safe}}}$
        s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe))
        s_star = torch.clamp(s_star, min=0)  # Ensure physical plausibility

        # Speed update: $v_{n+1} = v_n + \Delta t \cdot a_{max} [1 - (v_n / v_{des})^\delta - (s^* / s_{safe})^2]$
        v_follow = v_n + self.delta_t * a_max * (1 - (v_n / v_desired) ** delta - (s_star / s_safe) ** 2)
        predicted_speed = torch.clamp(v_follow, min=0)  # Ensure non-negative speed
        return predicted_speed, params


# --- Weight and Bias Initialization ---
def initialize_weights(model):
    """
    Initializes model weights using Xavier Uniform distribution and biases to constant 0.

    Args:
        model (nn.Module): The PyTorch model to initialize.
    """
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:  # Apply Xavier only to weights with dim >= 2
            nn.init.xavier_uniform_(param)
        elif "bias" in name:  # Set all biases to zero
            nn.init.constant_(param, 0)


# --- Model Training ---
def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
    """
    Standard training loop for the model.
    """
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed, batch_s_safe in train_loader:
            optimizer.zero_grad()  # Reset gradients
            predicted_speed, _ = model.predict_speed(batch_data, batch_s_safe)
            loss = criterion(predicted_speed, batch_speed)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {total_loss / len(train_loader):.4f}")
    return model


# --- Model Evaluation ---
def evaluate_model(model, test_loader):
    """
    Evaluates performance on the test set and plots ground truth vs. predictions.
    """
    model.eval()  # Set model to evaluation mode
    mse_loss = nn.MSELoss()
    total_mse = 0
    all_predicted = []
    all_true = []

    with torch.no_grad():  # Disable gradient tracking for inference
        for batch_idx, (batch_data, batch_speed, batch_s_safe) in enumerate(test_loader):
            predicted_speed, params = model.predict_speed(batch_data, batch_s_safe)
            loss = mse_loss(predicted_speed, batch_speed)
            total_mse += loss.item()
            all_predicted.append(predicted_speed)
            all_true.append(batch_speed)

    mse = total_mse / len(test_loader)
    rmse = torch.sqrt(torch.tensor(mse))
    mae = torch.mean(torch.abs(torch.cat(all_predicted) - torch.cat(all_true))).item()
    print(f"Evaluation Metrics:\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Plot Comparison (First 100 samples)
    plt.figure(figsize=(10, 6))
    plt.plot(torch.cat(all_true)[:100].cpu().numpy(), label='True Speed', linestyle='--', marker='o')
    plt.plot(torch.cat(all_predicted)[:100].cpu().numpy(), label='Predicted Speed', linestyle='-', marker='x')
    plt.title('True vs Predicted Speed (Hybrid IDM-Transformer)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()


def save_predictions_to_csv(model, test_loader, output_file="predictions.csv"):
    """
    Saves prediction results and IDM parameters to a CSV file.
    """
    model.eval()
    all_true_speeds = []
    all_predicted_speeds = []
    all_params = []

    with torch.no_grad():
        for batch_data, batch_speed, batch_s_safe in test_loader:
            predicted_speed, params = model.predict_speed(batch_data, batch_s_safe)
            all_true_speeds.append(batch_speed.cpu())
            all_predicted_speeds.append(predicted_speed.cpu())
            all_params.append(params.cpu())

    all_true_speeds = torch.cat(all_true_speeds).numpy()
    all_predicted_speeds = torch.cat(all_predicted_speeds).numpy()
    all_params = torch.cat(all_params).numpy()

    result_data = pd.DataFrame({
        "True Speed": all_true_speeds.flatten(),
        "Predicted Speed": all_predicted_speeds.flatten(),
        "v_desired": all_params[:, 0],
        "T": all_params[:, 1],
        "a_max": all_params[:, 2],
        "b_safe": all_params[:, 3],
        "delta": all_params[:, 4],
        "s0": all_params[:, 5]
    })

    result_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


def compute_position_and_spacing_and_save(model, test_data, test_real_speed, raw_data, label_data, train_size,
                                          test_s_safe, dt=0.1, output_file="predictions_extended.xlsx"):
    """
    Infers next-step Y-coordinate and gap spacing based on predicted speed, saving results to Excel.

    Args:
        model: Trained model.
        test_data: Test input features.
        test_real_speed: Ground truth speed for test set.
        raw_data: Original input (to retrieve current Y positions).
        label_data: Original labels (to retrieve GT next Y and gap).
        train_size: Offset for indexing test samples within raw_data.
    """
    model.eval()
    with torch.no_grad():
        pred_speed_tensor, _ = model.predict_speed(test_data, test_s_safe)
        pred_speed = pred_speed_tensor.squeeze().cpu().numpy()
    true_speed = test_real_speed.cpu().numpy()

    N_test = test_data.shape[0]
    idx = np.arange(train_size, train_size + N_test)

    # Convert current Y coordinates (ft to meters)
    current_Y_ft = raw_data[idx, -1, 4].cpu().numpy()  # ego Y coordinate
    current_Y_m = current_Y_ft * 0.3048

    # Retrieve current speed (m/s)
    current_speed_m = test_data[:, -1, 0].cpu().numpy()

    # Retrieve Ground Truth next-step Y and spacing (ft to meters)
    true_Y_ft = label_data[idx, -1, 3].cpu().numpy()  # GT Next-step Ego Y
    true_Y_m = true_Y_ft * 0.3048
    true_spacing_ft = label_data[idx, -1, 1].cpu().numpy()  # GT Next-step Gap
    true_spacing_m = true_spacing_ft * 0.3048

    # Infer displacement using constant acceleration approximation
    # Acceleration a = (v_next - v_current) / dt
    acceleration_m = (pred_speed - current_speed_m) / dt
    disp_m = current_speed_m * dt + 0.5 * acceleration_m * dt ** 2

    # Predict next-step Y position
    pred_Y_m = current_Y_m + disp_m

    # Logic: Target Spacing = (True Leader Y) - (Predicted Ego Y)
    # True Leader Y = True Ego Y + True Spacing
    true_leader_Y_m = true_Y_m + true_spacing_m
    pred_spacing_m = true_leader_Y_m - pred_Y_m

    # Error Metrics
    rmse_Y = np.sqrt(np.mean((pred_Y_m - true_Y_m) ** 2))
    mape_Y = np.mean(np.abs((pred_Y_m - true_Y_m) / (true_Y_m + 1e-6))) * 100
    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m) ** 2))
    mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m) / (true_spacing_m + 1e-6))) * 100

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

    sheet_name = "Transformer-IDM_1"
    # Append to existing file if available, otherwise create new
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a" if os.path.exists(output_file) else "w") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Results saved to '{output_file}' in sheet '{sheet_name}'.")


# --- Main Block ---
if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducibility

    # Load MATLAB data file
    data = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')
    raw_data = torch.tensor(data['train_data'], dtype=torch.float32)  # Full inputs
    label_data = torch.tensor(data['lable_data'], dtype=torch.float32)  # Full labels

    print("Original raw_data shape:", raw_data.shape)
    print("Original label_data shape:", label_data.shape)

    # Feature Engineering: Extract 50-step sequence with specific columns
    # [0] Ego Speed, [1] Gap, [2] Speed Diff, [3] Ego Accel, [-1] Leader Speed
    train_data = raw_data[:, -50:, [0, 1, 2, 3, -1]]
    # Next-step GT Speed is the first feature in label_data at the final time step
    train_real_speed = label_data[:, -1, 0]

    # Current gap (s_safe) retrieved from the last step of sequence (index 1)
    train_s_safe = torch.tensor(train_data[:, -1, 1].clone(), dtype=torch.float32)

    print(f"train_data shape: {train_data.shape}")
    print(f"train_real_speed shape: {train_real_speed.shape}")
    print(f"train_s_safe shape: {train_s_safe.shape}")

    # Unit Conversion: Imperial (ft, ft/s) to Metric (m, m/s)
    train_data[:, :, 0] *= 0.3048  # Ego Speed
    train_data[:, :, 1] *= 0.3048  # Gap
    train_data[:, :, 2] *= 0.3048  # Speed Difference
    train_data[:, :, 3] *= 0.3048  # Ego Acceleration
    train_data[:, :, 4] *= 0.3048  # Leader Speed
    train_real_speed *= 0.3048  # Next-step GT Speed
    train_s_safe *= 0.3048  # Current Gap

    # Subset the data for testing (using 10% of total samples for faster iteration)
    total_samples = train_data.shape[0]
    sample_size = int(total_samples * 1)
    train_data = train_data[:sample_size]
    train_real_speed = train_real_speed[:sample_size]
    train_s_safe = train_s_safe[:sample_size]

    # Validate data integrity
    check_data(train_data, "train_data")
    check_data(train_real_speed, "train_real_speed")
    check_data(train_s_safe, "train_s_safe")

    # Split: 80% Training, 20% Testing
    dataset_size = train_data.shape[0]
    train_size = int(dataset_size * 0.8)

    train_x, test_x = train_data[:train_size], train_data[train_size:]
    train_y, test_y = train_real_speed[:train_size], train_real_speed[train_size:]
    train_s_safe_split, test_s_safe_split = train_s_safe[:train_size], train_s_safe[train_size:]

    # Define DataLoaders
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y, train_s_safe_split)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y, test_s_safe_split)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Hyperparameters
    input_dim = train_data.shape[2]
    model_dim = 128
    num_heads = 4
    num_layers = 2

    # Instantiate Transformer-IDM Model
    model = HybridIDMModel(input_dim, model_dim, num_heads, num_layers)
    initialize_weights(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Start Training
    num_epochs = 100
    print("\n--- Starting Model Training ---")
    model = train_model(model, train_loader, optimizer, criterion, num_epochs)

    # Start Evaluation
    print("\n--- Starting Model Evaluation ---")
    evaluate_model(model, test_loader)

    # Trajectory and Gap Inference + Save to Excel
    output_excel_file = "predictions_extended.xlsx"
    compute_position_and_spacing_and_save(
        model,
        test_x,
        test_y,
        raw_data,
        label_data,
        train_size,
        test_s_safe_split,
        dt=0.1,
        output_file=output_excel_file
    )
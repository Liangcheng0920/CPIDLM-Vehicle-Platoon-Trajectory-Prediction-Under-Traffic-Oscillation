import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np

# Set environment variable to prevent errors caused by duplicate library loading
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ------------------------------
# Data Validation Function
def check_data(data, name="data"):
    """
    Check if the tensor contains NaN or Inf values.
    Args:
        data (torch.Tensor): Data to check.
        name (str): Name of the data for logging.
    """
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Contains NaN: {torch.isnan(data).any().item()}")
    print(f"Contains Inf: {torch.isinf(data).any().item()}")


# ------------------------------
# LSTM-based Multi-step Prediction Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1):
        """
        LSTM Model Initialization.
        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Dimension of LSTM hidden layers.
            prediction_steps (int): Number of future time steps to predict.
            num_layers (int): Number of layers in the LSTM network.
        """
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_steps = prediction_steps

        # LSTM Layer
        # batch_first=True implies input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer to map the last LSTM output to the prediction steps
        self.fc = nn.Linear(hidden_dim, prediction_steps)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input data, shape (batch_size, seq_len, input_dim).
        Returns:
            torch.Tensor: Predicted result, shape (batch_size, prediction_steps).
        """
        batch_size = x.size(0)

        # Initialize LSTM hidden state and cell state
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        # LSTM Forward Pass
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        lstm_out, (hn, cn) = self.lstm(x, (h_0, c_0))

        # Use the hidden state of the last time step from the last LSTM layer (hn[-1])
        # hn shape is (num_layers, batch_size, hidden_dim)
        out = self.fc(hn[-1])

        return out

    def predict_speed(self, x):
        """
        Helper function for speed prediction, calling the forward method.
        """
        return self.forward(x)


# ------------------------------
# Weight Initialization
def initialize_weights(model):
    """
    Initialize model weights.
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            if param.dim() >= 2:
                # Use Xavier Uniform initialization for multidimensional weights
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param)
        elif "bias" in name:
            # Initialize biases to 0
            nn.init.constant_(param, 0)


# ------------------------------
# Training Function
def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
    """
    Train the model.
    """
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed in train_loader:
            optimizer.zero_grad()  # Clear gradients
            pred = model.predict_speed(batch_data)
            loss = criterion(pred, batch_speed)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    return model


# ------------------------------
# Evaluation Function
def evaluate_model(model, test_loader):
    """
    Evaluate model performance.
    """
    model.eval()  # Set model to evaluation mode
    mse_loss = nn.MSELoss()
    total_mse = 0
    all_predicted = []
    all_true = []

    with torch.no_grad():  # No gradient calculation during evaluation
        for batch_idx, (batch_data, batch_speed) in enumerate(test_loader):
            predicted_speed = model.predict_speed(batch_data)
            loss = mse_loss(predicted_speed, batch_speed)
            total_mse += loss.item()
            all_predicted.append(predicted_speed)
            all_true.append(batch_speed)

    # Concatenate predictions and true values from all batches
    all_predicted = torch.cat(all_predicted).cpu().numpy()  # Shape: (N, prediction_steps)
    all_true = torch.cat(all_true).cpu().numpy()  # Shape: (N, prediction_steps)

    # Calculate overall evaluation metrics
    mse_val = np.mean((all_predicted - all_true) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted - all_true))
    print(f"Evaluation Metrics:\nMSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")

    # Concatenate sample data for plotting comparisons
    step = 5  # Sampling step
    num_samples_to_plot = 30
    true_concat = []
    pred_concat = []

    for i in range(num_samples_to_plot):
        idx = i * step
        if idx >= all_true.shape[0]:
            break
        true_concat.extend(all_true[idx])
        pred_concat.extend(all_predicted[idx])

    true_concat = np.array(true_concat)
    pred_concat = np.array(pred_concat)

    # Plot the comparison curves
    plt.figure(figsize=(12, 8))
    plt.plot(true_concat, linestyle='--', marker='o', label='True Value')
    plt.plot(pred_concat, linestyle='-', marker='x', label='Predicted Value')

    plt.title(f'Comparison of True and Predicted Speed for {num_samples_to_plot} Samples (Steps 1-{len(true_concat)})')
    plt.xlabel('Time Step')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()


# ------------------------------
# Save Prediction Results to CSV
def save_predictions_to_csv(model, test_loader, output_file="predictions.csv"):
    """
    Save predicted and true speeds to a CSV file.
    """
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for bd, bt in test_loader:
            p = model.predict_speed(bd)
            all_t.append(bt.cpu())
            all_p.append(p.cpu())
    all_t = torch.cat(all_t).numpy()
    all_p = torch.cat(all_p).numpy()

    df_data = {}
    for j in range(all_t.shape[1]):
        df_data[f"True_Step{j + 1}"] = all_t[:, j]
    for j in range(all_p.shape[1]):
        df_data[f"Pred_Step{j + 1}"] = all_p[:, j]
    df = pd.DataFrame(df_data)

    df.to_csv(output_file, index=False)
    print(f"Speed prediction results saved to {output_file}")


# ------------------------------
# Compute and Save Future Positions for Lead Vehicle
def compute_future_positions_and_save(model,
                                      test_data,
                                      raw_data_all,
                                      label_data_all,
                                      original_train_size,
                                      dt=0.1,
                                      output_file="pred_positions.xlsx",
                                      model_name_sheet="LSTM_1"):
    """
    Calculate and save future positions based on multi-step speed predictions.
    """
    model.eval()
    with torch.no_grad():
        pred_speeds = model.predict_speed(test_data).cpu().numpy()  # Shape: (N_test, prediction_steps)

    N_test, steps = pred_speeds.shape

    # Determine index in the original dataset for the test set
    idx_in_original_data = np.arange(original_train_size, original_train_size + N_test)

    # Current speed (m/s) from input features
    curr_speed = test_data[:, -1, 0].cpu().numpy()

    # Current lead vehicle position (ft -> m)
    curr_pos_ft = raw_data_all[idx_in_original_data, -1, 7].cpu().numpy()
    curr_pos_m = curr_pos_ft * 0.3048

    # Ground truth future positions (ft -> m)
    true_pos_ft = label_data_all[idx_in_original_data, :, 5].cpu().numpy()
    true_pos_m = true_pos_ft * 0.3048

    # Ground truth speeds (m/s)
    true_speeds = label_data_all[idx_in_original_data, :, 4].cpu().numpy() * 0.3048

    # Initialize storage for predicted positions
    pred_pos_m = np.zeros((N_test, steps))

    # Perform recursive position calculation for each sample
    for i in range(N_test):
        prev_speed_for_calc = curr_speed[i]
        prev_pos_for_calc = curr_pos_m[i]

        for k in range(steps):
            v_pred_current_step = pred_speeds[i, k]

            # Kinematic displacement formula: s = v0*t + 0.5*a*t^2
            # Acceleration a = (v_final - v_initial) / dt
            acceleration = (v_pred_current_step - prev_speed_for_calc) / dt
            displacement = prev_speed_for_calc * dt + 0.5 * acceleration * dt ** 2

            current_pos = prev_pos_for_calc + displacement
            pred_pos_m[i, k] = current_pos

            # Update state for next step
            prev_speed_for_calc = v_pred_current_step
            prev_pos_for_calc = current_pos

    # Error Evaluation
    mask = true_pos_m != 0
    rmse_p = np.sqrt(np.mean((pred_pos_m[mask] - true_pos_m[mask]) ** 2))
    mape_p = np.mean(np.abs((pred_pos_m[mask] - true_pos_m[mask]) / true_pos_m[mask])) * 100
    print(f"Future Position Prediction Error -- RMSE: {rmse_p:.4f} m, MAPE: {mape_p:.2f}%")

    # Save to Excel
    data_dict = {}
    for i in range(steps):
        data_dict[f"Pred_Speed_step{i + 1}(m/s)"] = pred_speeds[:, i]
        data_dict[f"Pred_Pos_step{i + 1}(m)"] = pred_pos_m[:, i]
        data_dict[f"True_Pos_step{i + 1}(m)"] = true_pos_m[:, i]
        data_dict[f"True_Speed_step{i + 1}(m/s)"] = true_speeds[:, i]

    df_pos = pd.DataFrame(data_dict)

    if os.path.exists(output_file):
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            df_pos.to_excel(writer, sheet_name=model_name_sheet, index=False)
    else:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df_pos.to_excel(writer, sheet_name=model_name_sheet, index=False)

    print(f"Model results saved to '{output_file}' in worksheet '{model_name_sheet}'.")


# ------------------------------
# Main Function
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load Data
    try:
        data = sio.loadmat('E:\\pythonProject1\\data_ngsim\\data_5.mat')
    except FileNotFoundError:
        print("Error: data_5.mat file not found. Please check the path.")
        exit()

    raw_data_full = torch.tensor(data['train_data'], dtype=torch.float32)
    label_data_full = torch.tensor(data['lable_data'], dtype=torch.float32)

    # Extract multi-step speed labels (Future speeds are in index 0 of label data)
    train_real_speed_all_steps = label_data_full[:, :, 0]
    print("Shape of future true speeds (train_real_speed_all_steps):", train_real_speed_all_steps.shape)

    # Construct multi-step inputs: Last 50 steps of specific features
    input_features_data = raw_data_full[:, -50:, [0, 1, 2, 3, 5]].clone() * 0.3048
    target_speeds_data = train_real_speed_all_steps.clone() * 0.3048
    print("Shape of input features (input_features_data):", input_features_data.shape)

    # Data Sampling (e.g., using 30% of data for a quick test)
    total_samples_original = input_features_data.shape[0]
    sample_size = int(total_samples_original * 0.3)

    sampled_input_features = input_features_data[:sample_size]
    sampled_target_speeds = target_speeds_data[:sample_size]

    print(f"Original total samples: {total_samples_original}")
    print(f"Samples after sampling: {sample_size}")

    check_data(sampled_input_features, "Sampled Input Features")
    check_data(sampled_target_speeds, "Sampled Target Speeds")

    # Split Train/Test sets
    current_dataset_size = sampled_input_features.shape[0]
    train_split_ratio = 0.8
    train_size_for_split = int(current_dataset_size * train_split_ratio)

    train_data = sampled_input_features[:train_size_for_split]
    train_real_speed = sampled_target_speeds[:train_size_for_split]

    test_data = sampled_input_features[train_size_for_split:]
    test_real_speed = sampled_target_speeds[train_size_for_split:]

    print(f"Train set size: {train_data.shape[0]}")
    print(f"Test set size: {test_data.shape[0]}")

    # Create DataLoaders
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
    num_lstm_layers = 1

    # Initialize Model
    model = LSTMModel(input_dim, hidden_dim, prediction_steps, num_layers=num_lstm_layers)
    initialize_weights(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # Train Model
    print("Starting LSTM Model training...")
    model = train_model(model, train_loader, optimizer, criterion, num_epochs=100)

    # Evaluate Model
    print("\nStarting LSTM Model evaluation...")
    evaluate_model(model, test_loader)

    print("\nProcessing complete.")
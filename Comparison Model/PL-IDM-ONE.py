import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import seaborn as sns  # Import a more aesthetic plotting library

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --- Data Check Function ---
def check_data(data, name="data"):
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# --- Define Hybrid Model ---
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(HybridIDMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 6)  # IDM parameters: [v_desired, T, a_max, b_safe, delta, s0]
        self.softplus = nn.Softplus()
        self.delta_t = 0.1  # Time step

    def forward(self, x):
        out, _ = self.lstm(x)
        params = self.fc(out[:, -1, :])
        params = self.softplus(params)  # Ensure parameters are positive
        return params

    def predict_speed(self, x, s_safe):
        params = self.forward(x)
        v_n = x[:, -1, 0]
        delta_v = x[:, -1, 2]
        v_desired, T, a_max, b_safe, delta, s0 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:,
                                                                                                         4], params[:,
                                                                                                             5]

        # IDM formula for desired gap
        s_star = s0 + v_n * T + (v_n * delta_v) / (2 * torch.sqrt(a_max * b_safe))
        s_star = torch.clamp(s_star, min=0)

        # Calculate speed for the next time step
        v_follow = v_n + self.delta_t * a_max * (1 - (v_n / v_desired) ** delta - (s_star / s_safe) ** 2)
        predicted_speed = torch.clamp(v_follow, min=0)
        return predicted_speed, params


# --- Initialize Weights and Biases ---
def initialize_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)


# --- Model Training ---
def train_model(model, train_loader, optimizer, criterion, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed, batch_s_safe in train_loader:
            optimizer.zero_grad()
            predicted_speed, _ = model.predict_speed(batch_data, batch_s_safe)
            loss = criterion(predicted_speed, batch_speed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    return model


# --- Model Evaluation ---
def evaluate_model(model, test_loader):
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    all_predicted = []
    all_true = []
    all_predicted_params = []  # Store all predicted parameters

    with torch.no_grad():
        for batch_idx, (batch_data, batch_speed, batch_s_safe) in enumerate(test_loader):
            predicted_speed, params = model.predict_speed(batch_data, batch_s_safe)
            loss = mse_loss(predicted_speed, batch_speed)
            total_mse += loss.item()
            all_predicted.append(predicted_speed)
            all_true.append(batch_speed)
            all_predicted_params.append(params)  # Append parameters for each batch

    mse = total_mse / len(test_loader)
    rmse = torch.sqrt(torch.tensor(mse))
    mae = torch.mean(torch.abs(torch.cat(all_predicted) - torch.cat(all_true))).item()
    print(f"Evaluation Metrics:\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Plot comparison chart
    plt.figure(figsize=(10, 6))
    plt.plot(torch.cat(all_true)[:100].cpu().numpy(), label='True Speed', linestyle='--', marker='o')
    plt.plot(torch.cat(all_predicted)[:100].cpu().numpy(), label='Predicted Speed', linestyle='-', marker='x')
    plt.title('True vs Predicted Speed (Hybrid IDM-LSTM)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()

    # Concatenate all predicted parameters and return them
    return torch.cat(all_predicted_params, dim=0).cpu().numpy()


def save_predictions_to_csv(model, test_loader, output_file="predictions_param.csv"):
    """
    Save test data prediction results to a CSV file.
    """
    model.eval()
    all_true_speeds = []
    all_predicted_speeds = []
    all_params = []

    with torch.no_grad():
        for batch_data, batch_speed, batch_s_safe in test_loader:
            # Get model prediction output
            predicted_speed, params = model.predict_speed(batch_data, batch_s_safe)

            # Store true speed and predicted parameters
            all_true_speeds.append(batch_speed.cpu())
            all_predicted_speeds.append(predicted_speed.cpu())
            all_params.append(params.cpu())

    # Convert to Tensor format and concatenate
    all_true_speeds = torch.cat(all_true_speeds).numpy()
    all_predicted_speeds = torch.cat(all_predicted_speeds).numpy()
    all_params = torch.cat(all_params).numpy()

    # Merge into a single DataFrame
    result_data = pd.DataFrame({
        "True Speed": all_true_speeds,
        "Predicted Speed": all_predicted_speeds,
        "v_desired": all_params[:, 0],
        "T": all_params[:, 1],
        "a_max": all_params[:, 2],
        "b_safe": all_params[:, 3],
        "delta": all_params[:, 4],
        "s0": all_params[:, 5]
    })

    # Save as CSV file
    result_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


# Based on predicted speed, calculate Y-coordinate and headway spacing for the next moment, then save results
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
        # Squeeze the speed tensor and move to CPU numpy
        pred_speed = pred_speed_tensor.squeeze().cpu().numpy()
    true_speed = test_real_speed.cpu().numpy()

    N_test = test_data.shape[0]
    idx = np.arange(train_size, train_size + N_test)

    current_Y_ft = raw_data[idx, -1, 4].numpy()  # Index of the 4th dimension (adjust if necessary)
    current_Y_ft *= 0.3048  # Convert ft to m
    current_speed_m = test_data[:, -1, 0].numpy()

    true_Y_ft = label_data[idx, -1, 3].numpy()
    true_Y_ft *= 0.3048
    true_spacing_m = label_data[idx, -1, 1].numpy()
    true_spacing_m *= 0.3048

    # Calculate displacement based on predicted speed (using constant acceleration approximation)
    disp_m = current_speed_m * dt + 0.5 * ((pred_speed - current_speed_m) / dt) * dt ** 2
    disp_ft = disp_m
    pred_Y_ft = current_Y_ft + disp_ft

    pred_Y_m = pred_Y_ft
    true_Y_m = true_Y_ft
    pred_spacing_m = (true_Y_ft - pred_Y_ft) + true_spacing_m

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
    sheet_name = "LSTM-IDM_1"  # Target sheet name

    # Append to existing file if it exists, otherwise create new
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a" if os.path.exists(output_file) else "w") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"New model results saved to '{output_file}' in sheet '{sheet_name}'.")


def plot_parameter_distributions(predicted_params):
    """
    Plot the distributions of the six IDM parameters predicted by the LSTM-IDM model on the test set:
    [v_desired, T, a_max, b_safe, delta, s0]
    """
    # ---- Style Settings ----
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11

    # Parameter names (including units)
    params_names = [
        r'$v_{\mathrm{desired}}$ (m/s)',
        r'$T$ (s)',
        r'$a_{\mathrm{max}}$ (m/s$^2$)',
        r'$b_{\mathrm{safe}}$ (m/s$^2$)',
        r'$\delta$',
        r'$s_{0}$ (m)'
    ]

    # Color scheme: Histogram in blue (semi-transparent), KDE curve in orange
    hist_color = "#1f77b4"
    kde_color = "#ff7f0e"
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        sns.histplot(
            predicted_params[:, i],
            ax=ax,
            stat='density',
            bins=30,
            color=hist_color,
            alpha=0.6,
            edgecolor='none'
        )
        sns.kdeplot(
            predicted_params[:, i],
            ax=ax,
            color=kde_color,
            lw=2
        )
        ax.set_title(f'Distribution of {params_names[i]}')
        ax.set_xlabel(params_names[i])
        ax.set_ylabel('Density')
        # Show full scale, no truncation
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Global layout adjustment and saving
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Distribution of Predicted IDM Parameters on Test Set', fontsize=16, y=1.02)
    plt.savefig('idm_params_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


# --- Main Function ---
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load data
    data = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')  # Replace with actual file path
    raw_data = torch.tensor(data['train_data'], dtype=torch.float32)  # Expecting shape (samples, time_steps, features)
    label_data = torch.tensor(data['lable_data'], dtype=torch.float32)

    # Note: Original code used train_data = raw_data[:, -5:, :] taking only last 5 steps.
    # If looking for a reach of 0-50 for tau, historical steps should be long enough. Keeping sequence length 50 here.
    print(raw_data.shape)
    # Features: 1. Ego speed, 2. Distance to leader, 3. Speed diff, 4. Ego acc, 5. Leader speed
    train_data = raw_data[:, -50:, [0, 1, 2, 3, -1]]

    train_real_speed1 = torch.tensor(data['lable_data'], dtype=torch.float32)

    print(train_real_speed1.shape)
    train_real_speed = train_real_speed1[:, -1, 0]

    # Modification: Extract the 2nd column data (index 1) from the last frame of train_data as s_safe
    train_s_safe = torch.tensor(train_data[:, -1, 1].clone(), dtype=torch.float32)

    # Verify time-sequence order
    print("Processed time sequence example (first sample):")
    print(train_data[0, :, 0])  # Assuming first column is speed

    # Print shapes
    print(f"train_data shape: {train_data.shape}")
    print(f"train_real_speed shape: {train_real_speed.shape}")
    print(f"train_s_safe shape: {train_s_safe.shape}")

    # Preprocessing: Unit conversion (ft to m)
    train_data[:, :, 0] *= 0.3048
    train_data[:, :, 1] *= 0.3048
    train_data[:, :, 2] *= -0.3048  # Speed diff conversion
    train_data[:, :, 3] *= 0.3048
    train_data[:, :, 4] *= 0.3048
    train_real_speed *= 0.3048
    train_s_safe *= 0.3048

    # Select first 10% of data for training/testing
    total_samples = train_data.shape[0]
    sample_size = int(total_samples * 1)
    train_data = train_data[:sample_size]
    train_real_speed = train_real_speed[:sample_size]
    train_s_safe = train_s_safe[:sample_size]

    # Check data for errors
    check_data(train_data, "train_data")
    check_data(train_real_speed, "train_real_speed")
    check_data(train_s_safe, "train_s_safe")

    # Split dataset into training (80%) and testing (20%)
    dataset_size = train_data.shape[0]
    train_size = int(dataset_size * 0.8)

    train_x = train_data[:train_size]
    test_x = train_data[train_size:]
    train_y = train_real_speed[:train_size]
    test_y = train_real_speed[train_size:]

    train_data, test_data = train_data[:train_size], train_data[train_size:]
    train_real_speed, test_real_speed = train_real_speed[:train_size], train_real_speed[train_size:]
    train_s_safe, test_s_safe = train_s_safe[:train_size], train_s_safe[train_size:]

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(train_data, train_real_speed, train_s_safe)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_real_speed, test_s_safe)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, weights, loss, and optimizer
    input_dim = train_data.shape[2]  # Input feature count
    hidden_dim = 128  # Increased hidden units
    num_layers = 1  # LSTM layer count
    model = HybridIDMModel(input_dim, hidden_dim, num_layers)
    initialize_weights(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adjusted learning rate

    # Train the model
    num_epochs = 100  # Increased training epochs
    model = train_model(model, train_loader, optimizer, criterion, num_epochs)

    # Evaluate model and get predicted parameters
    predicted_params = evaluate_model(model, test_loader)

    # Plot parameter distributions
    plot_parameter_distributions(predicted_params)

    # Save to CSV
    output_file = "test_predictions_param.csv"
    save_predictions_to_csv(model, test_loader, output_file=output_file)
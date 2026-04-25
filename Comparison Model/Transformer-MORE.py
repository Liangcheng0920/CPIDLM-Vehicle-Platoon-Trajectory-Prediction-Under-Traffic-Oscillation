import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import glob  # Used for file path searching
import math  # Used for mathematical constants like pi

# Set environment variable to allow multiple OpenMP libraries (prevents conflicts in some environments)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device in use: {device}")

# --- Global Path Definitions ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # Directory for dataset
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim"  # Directory for saving experimental results

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------
# Data Validation Function
def check_data(data, name="data"):
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


# ------------------------------
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # Tensor to store positional encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Index tensor [0, max_len-1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Frequency term
        pe[:, 0::2] = torch.sin(position * div_term)  # Sine encoding for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosine encoding for odd indices
        pe = pe.unsqueeze(0).transpose(0, 1)  # Reshape to (seq_len, batch_size, d_model)
        self.register_buffer('pe', pe)  # Register as model buffer (saved but not as a parameter)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]  # Add positional encoding to input
        return self.dropout(x)


# ------------------------------
# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 prediction_steps, dropout=0.1, num_steps=50):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.input_dim = input_dim
        self.model_dim = model_dim  # Internal dimension (d_model)
        self.num_steps = num_steps  # Input sequence length

        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_len=num_steps)
        self.embedding = nn.Linear(input_dim, model_dim)  # Linear embedding layer

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers,
                                                         num_layers=num_encoder_layers)

        # Output layer mapping model dimension to multi-step prediction
        self.fc_out = nn.Linear(model_dim * num_steps, prediction_steps)  # Option 1: Flattened strategy

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()

    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        src = self.embedding(src) * math.sqrt(self.model_dim)  # Embedding and scaling

        # Transpose to (seq_len, batch_size, model_dim) for PositionalEncoding
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # Back to (batch_size, seq_len, model_dim)

        output = self.transformer_encoder(src)
        # output: (batch_size, seq_len, model_dim)

        # Use flattened output for final prediction
        output = output.reshape(output.size(0), -1)  # (batch_size, seq_len * model_dim)
        output = self.fc_out(output)  # (batch_size, prediction_steps)

        return output

    def predict_speed(self, x):
        return self.forward(x)


# ------------------------------
# Weight Initialization (General)
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
def evaluate_model(model, train_real_speed, test_loader, dataset_name="", results_dir=""):
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

    # Combine all batches
    all_predicted = torch.cat(all_predicted).cpu().numpy()
    all_true = torch.cat(all_true).cpu().numpy()

    # Calculate overall metrics
    mse_val = np.mean((all_predicted - all_true) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted - all_true))

    # Handle MAPE calculation (avoiding division by zero)
    true_non_zero = all_true[all_true != 0]
    pred_non_zero = all_predicted[all_true != 0]
    mape_p = np.mean(np.abs((pred_non_zero - true_non_zero) / true_non_zero)) * 100 if len(
        true_non_zero) > 0 else float('inf')

    print(f"Evaluation Metrics:\nMSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, MAPE: {mape_p:.2f}%")

    # Plot concatenation logic for 30 samples
    num_pred_steps = all_true.shape[1]
    num_samples_to_plot = 30
    true_concat, pred_concat = [], []

    for i in range(min(num_samples_to_plot, all_true.shape[0])):
        true_concat.extend(all_true[i])
        pred_concat.extend(all_predicted[i])

    true_concat = np.array(true_concat)
    pred_concat = np.array(pred_concat)

    plt.figure(figsize=(12, 8))
    plt.plot(true_concat, linestyle='--', marker='o', label='True')
    plt.plot(pred_concat, linestyle='-', marker='x', label='Predicted')

    plt.title(f'True vs Predicted Speed for {num_samples_to_plot} Samples (Multi-step)')
    plt.xlabel(f'Time Step (concatenated over {num_pred_steps} steps per sample)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()

    plot_filename = os.path.join(results_dir, f"{dataset_name}_transformer_speed_comparison.png")
    plt.savefig(plot_filename)
    print(f"Speed comparison plot saved to {plot_filename}")
    plt.close()
    return mse_val, rmse_val, mae_val, mape_p


# ------------------------------
# Future Position Calculation and Saving
def compute_future_positions_and_save(model, test_data, raw_data, label_data, train_size, dt=0.1,
                                      output_file="pred_positions.xlsx", dataset_name=""):
    model.eval()
    with torch.no_grad():
        pred_speeds = model.predict_speed(test_data).cpu().numpy()  # (N_test, prediction_steps)

    N_test, steps = pred_speeds.shape
    test_start_idx_in_raw = train_size
    test_end_idx_in_raw = train_size + N_test

    # Current speed in m/s (from input features)
    curr_speed_m_s = test_data[:, -1, 0].cpu().numpy()

    # Current global position of lead vehicle (ft -> m)
    curr_pos_ft = raw_data[test_start_idx_in_raw:test_end_idx_in_raw, -1, 7].cpu().numpy()
    curr_pos_m = curr_pos_ft * 0.3048

    # True future position of lead vehicle (ft -> m)
    true_pos_ft = label_data[test_start_idx_in_raw:test_end_idx_in_raw, :, 5].cpu().numpy()
    true_pos_m = true_pos_ft * 0.3048

    # True future speeds of lead vehicle (m/s)
    true_speeds_m_s = label_data[test_start_idx_in_raw:test_end_idx_in_raw, :, 0].cpu().numpy() * 0.3048

    pred_pos_m = np.zeros((N_test, steps))

    # Kinematic recursive calculation for each sample
    for i in range(N_test):
        current_prec_speed_m_s = curr_speed_m_s[i]
        current_prec_pos_m = curr_pos_m[i]

        for k in range(steps):
            v_pred_prec = pred_speeds[i, k]

            # Assume constant acceleration over small dt interval
            a_prec_step = (v_pred_prec - current_prec_speed_m_s) / dt
            disp_step = current_prec_speed_m_s * dt + 0.5 * a_prec_step * dt ** 2
            current_prec_pos_m += disp_step
            pred_pos_m[i, k] = current_prec_pos_m

            current_prec_speed_m_s = v_pred_prec  # Update speed for next step

    # Error Evaluation
    rmse_p = np.sqrt(np.mean((pred_pos_m - true_pos_m) ** 2))
    mask = true_pos_m != 0
    mape_p = np.mean(np.abs((pred_pos_m[mask] - true_pos_m[mask]) / true_pos_m[mask])) * 100 if np.any(mask) else float(
        'inf')
    print(f"Future Position Error -- RMSE: {rmse_p:.4f} m, MAPE: {mape_p:.2f}%")

    # Data construction for Export
    data_dict = {}
    for i in range(steps):
        data_dict[f"Pred_Speed_step{i + 1}(m/s)"] = pred_speeds[:, i]
        data_dict[f"Pred_Pos_step{i + 1}(m)"] = pred_pos_m[:, i]
        data_dict[f"True_Pos_step{i + 1}(m)"] = true_pos_m[:, i]
        data_dict[f"True_Speed_step{i + 1}(m/s)"] = true_speeds_m_s[:, i]

    df_pos = pd.DataFrame(data_dict)
    sheet_name = dataset_name

    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
            df_pos.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df_pos.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Position prediction results for {dataset_name} saved to '{output_file}' in sheet '{sheet_name}'.")
    return rmse_p, mape_p


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


# ------------------------------
# Main Entry Point
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    if not data_files:
        print(f"No .mat files found in directory {DATA_DIR}.")
        exit()

    position_predictions_excel_path = os.path.join(RESULTS_DIR, "pred_positions_Transformer_Summary.xlsx")
    summary_metrics_csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_Transformer.csv")

    # Clean up old result files
    for path in [position_predictions_excel_path, summary_metrics_csv_path]:
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted old file: {path}")

    for data_file_path in data_files:
        dataset_filename = os.path.basename(data_file_path)
        dataset_name_clean = dataset_filename.replace(".mat", "")

        print(f"\n==================== Processing: {dataset_filename} (Transformer) ====================")

        data = sio.loadmat(data_file_path)
        raw_data = torch.tensor(data['train_data'], dtype=torch.float32)
        label_data = torch.tensor(data['lable_data'], dtype=torch.float32)

        # Feature selection and unit conversion
        # Indices: [Preceding Speed, Following Speed, Speed Difference, Spacing, Preceding Acceleration]
        input_features_indices = [0, 1, 2, 3, 5]
        train_data_m = raw_data[:, -50:, input_features_indices].clone() * 0.3048
        train_real_speed_m_s = label_data[:, :, 0].clone() * 0.3048  # Target multi-step speeds

        # Fraction for sampling (set to 0.2 for quick validation)
        sample_size = int(train_data_m.shape[0] * 0.2)
        train_data_sampled = train_data_m[:sample_size]
        train_real_speed_sampled = train_real_speed_m_s[:sample_size]

        check_data(train_data_sampled, "Sampled Input (m)")

        train_size = int(sample_size * 0.8)
        train_X, train_Y = train_data_sampled[:train_size], train_real_speed_sampled[:train_size]
        test_X, test_Y = train_data_sampled[train_size:], train_real_speed_sampled[train_size:]

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_Y), batch_size=32,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_X, test_Y), batch_size=32,
                                                  shuffle=False)

        # Transformer Config
        input_dim = train_X.shape[2]
        model_dim = 128
        nhead = 4  # Ensure model_dim is divisible by nhead

        model = TransformerModel(
            input_dim=input_dim,
            model_dim=model_dim,
            nhead=nhead,
            num_encoder_layers=3,
            num_decoder_layers=0,
            dim_feedforward=256,
            prediction_steps=train_Y.shape[1],
            dropout=0.1,
            num_steps=50
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        print("Starting training...")
        model = train_model(model, train_loader, optimizer, criterion, num_epochs=30)

        print("Evaluating...")
        speed_mse, speed_rmse, speed_mae, speed_mape = evaluate_model(model, train_Y, test_loader,
                                                                      dataset_name=dataset_name_clean,
                                                                      results_dir=RESULTS_DIR)

        print("Computing positions...")
        pos_rmse, pos_mape = compute_future_positions_and_save(model, test_X, raw_data, label_data, train_size, dt=0.1,
                                                               output_file=position_predictions_excel_path,
                                                               dataset_name=dataset_name_clean)

        store_dataset_metrics(dataset_name_clean, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse, pos_mape)

    # Save summary
    pd.DataFrame(all_datasets_metrics_summary).to_csv(summary_metrics_csv_path, index=False)
    print(f"\nSummary metrics saved to {summary_metrics_csv_path}")
    print("\nAll datasets processed successfully.")
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ------------------------------
# Utility Class: Early Stopping
# ------------------------------
class EarlyStopping:
    """
    Stops training when the validation loss does not improve within a given patience period,
    and saves the best model.
    """

    def __init__(self, patience=7, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases'''
        torch.save(model.state_dict(), self.path)


# ------------------------------
# Improved Time Series Transformer Model
# ------------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, num_steps=50, output_dim=1, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model

        # 1. Input Projection: Map feature dimension to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Learnable Positional Encoding
        # For fixed-length time windows, learnable PE often adapts better to data features than sine/cosine
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, d_model))

        self.dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder
        # Using batch_first=True for easier handling of (Batch, Seq, Feature)
        # activation='gelu' often performs better in modern Transformers
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model,
                                                    nhead=nhead,
                                                    dim_feedforward=d_model * 4,
                                                    dropout=dropout,
                                                    activation='gelu',
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 4. MLP Head (Decoder)
        # Using MLP enhances the non-linear capacity for regression compared to a single linear layer
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        # Initialize positional encoding and projection layers
        self.input_proj.weight.data.uniform_(-initrange, initrange)
        self.input_proj.bias.data.zero_()
        self.decoder[0].weight.data.uniform_(-initrange, initrange)
        self.decoder[3].weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        :param src: (batch_size, seq_len, input_dim)
        :return: (batch_size, output_dim)
        """
        # Embedding
        src = self.input_proj(src)  # (Batch, Seq, d_model)

        # Add Positional Encoding (Broadcasting)
        src = src + self.pos_embedding[:, :src.size(1), :]
        src = self.dropout(src)

        # Transformer Encoding
        output = self.transformer_encoder(src)  # (Batch, Seq, d_model)

        # Take the features from the last time step for prediction (Many-to-One)
        # Alternative: output.mean(dim=1) for global average pooling
        final_feature = output[:, -1, :]

        # MLP Decoding
        prediction = self.decoder(final_feature)
        return prediction

    def predict_speed(self, x):
        return self.forward(x)


# ------------------------------
# Advanced Model Training Function
# ------------------------------
def train_model_advanced(model, train_loader, val_loader, optimizer, criterion, num_epochs=50, patience=10):
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=patience, path='best_transformer_model.pt')

    # Initialize Learning Rate Scheduler (Reduce LR when Val Loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    print("--- Start Advanced Training ---")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for batch_data, batch_speed in train_loader:
            if batch_speed.dim() == 1:
                batch_speed = batch_speed.unsqueeze(1)

            optimizer.zero_grad()
            predicted_speed = model(batch_data)
            loss = criterion(predicted_speed, batch_speed)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_data, val_speed in val_loader:
                if val_speed.dim() == 1:
                    val_speed = val_speed.unsqueeze(1)
                val_pred = model(val_data)
                v_loss = criterion(val_pred, val_speed)
                val_running_loss += v_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print Progress
        print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("--- Training Complete ---")

    # Load best model weights
    model.load_state_dict(torch.load('best_transformer_model.pt'))

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    return model


# ------------------------------
# Model Evaluation Function
# ------------------------------
def evaluate_model(model, test_loader):
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    all_predicted = []
    all_true = []
    with torch.no_grad():
        for batch_data, batch_speed in test_loader:
            if batch_speed.dim() == 1:
                batch_speed = batch_speed.unsqueeze(1)

            predicted_speed = model.predict_speed(batch_data)
            loss = mse_loss(predicted_speed, batch_speed)
            total_mse += loss.item()
            all_predicted.append(predicted_speed)
            all_true.append(batch_speed)

    all_predicted = torch.cat(all_predicted).cpu().numpy()
    all_true = torch.cat(all_true).cpu().numpy()

    mse_val = np.mean((all_predicted - all_true) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted - all_true))
    print(f"Evaluation Metrics:\nMSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")

    N_plot = min(100, len(all_true))
    plt.figure(figsize=(10, 6))
    plt.plot(all_true[:N_plot], label='True Speed', linestyle='--', marker='o')
    plt.plot(all_predicted[:N_plot], label='Predicted Speed', linestyle='-', marker='x')
    plt.title('True vs Predicted Speed (Improved Transformer)')
    plt.xlabel('Sample Index')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.show()


# ------------------------------
# Position and Spacing Calculation Function
# ------------------------------
def compute_position_and_spacing_and_save(model,
                                          test_data,
                                          test_real_speed_next_step,
                                          raw_data_full,
                                          label_data_full,
                                          train_size,
                                          seq_len,
                                          dt=0.1,
                                          output_file="predictions_extended.xlsx"):
    model.eval()
    with torch.no_grad():
        pred_speed_next = model.predict_speed(test_data).squeeze().cpu().numpy()
    true_speed_next = test_real_speed_next_step.squeeze().cpu().numpy()

    N_test = test_data.shape[0]
    idx_test_start = train_size
    idx_test_end = train_size + N_test

    current_speed_m = test_data[:, -1, 0].cpu().numpy()
    current_Y_ft = raw_data_full[idx_test_start:idx_test_end, seq_len - 1, 4].cpu().numpy()
    current_Y_m = current_Y_ft * 0.3048

    true_Y_ft_next = label_data_full[idx_test_start:idx_test_end, 0, 3].cpu().numpy()
    true_spacing_ft_next = label_data_full[idx_test_start:idx_test_end, 0, 1].cpu().numpy()

    true_Y_m_next = true_Y_ft_next * 0.3048
    true_spacing_m_next = true_spacing_ft_next * 0.3048

    # Using kinematic formula: displacement = v0*t + 0.5*a*t^2
    accel_m = (pred_speed_next - current_speed_m) / dt
    disp_m = current_speed_m * dt + 0.5 * accel_m * dt ** 2

    pred_Y_m_next = current_Y_m + disp_m
    pred_spacing_m_next = true_spacing_m_next + (pred_Y_m_next - true_Y_m_next)

    rmse_Y = np.sqrt(np.mean((pred_Y_m_next - true_Y_m_next) ** 2))
    mape_Y = np.mean(np.abs((pred_Y_m_next - true_Y_m_next) / true_Y_m_next)) * 100
    rmse_sp = np.sqrt(np.mean((pred_spacing_m_next - true_spacing_m_next) ** 2))
    mape_sp = np.mean(np.abs((pred_spacing_m_next - true_spacing_m_next) / true_spacing_m_next)) * 100

    print(f"Position Error    -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
    print(f"Spacing  Error    -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")

    df = pd.DataFrame({
        "Pred Speed (m/s)": pred_speed_next.flatten(),
        "True Speed (m/s)": true_speed_next.flatten(),
        "Predicted Y (m)": pred_Y_m_next.flatten(),
        "True Y (m)": true_Y_m_next.flatten(),
        "Predicted Spacing (m)": pred_spacing_m_next.flatten(),
        "True Spacing (m)": true_spacing_m_next.flatten(),
    })
    sheet_name = "Improved_Transformer_new"

    if os.path.exists(output_file):
        try:
            with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        except Exception:
            with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"New model results saved to '{output_file}' in sheet '{sheet_name}'.")


# ------------------------------
# Main Function
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    seq_len = 50

    # Load Data
    try:
        mat = sio.loadmat('E:\pythonProject1\data_fine_0.1.mat')
        raw_data_full = torch.tensor(mat['train_data'], dtype=torch.float32)
        label_data_full = torch.tensor(mat['lable_data'], dtype=torch.float32)
    except FileNotFoundError:
        print("Error: Data file not found. Please check the path.")
        exit()

    # Feature selection
    input_features = [0, 1, 2, 3, -1]
    train_x_input = raw_data_full[:, :seq_len, input_features]
    train_y_target = label_data_full[:, 0, 0].unsqueeze(1)  # (N, 1)

    # Unit conversion (ft -> m)
    train_x_input[:, :, :5] *= 0.3048
    train_y_target *= 0.3048

    # --- Sample 10% to speed up experimentation ---
    total_samples = train_x_input.shape[0]
    sample_size = int(total_samples * 1)
    print(f"Total samples: {total_samples}, using first {sample_size} samples for training/testing.")

    train_x_input = train_x_input[:sample_size]
    train_y_target = train_y_target[:sample_size]

    # Dataset split: 80% Train, 20% Test
    # Validation set will be further split from the training set later
    N = train_x_input.shape[0]
    train_size = int(N * 0.8)

    train_x = train_x_input[:train_size]
    test_x = train_x_input[train_size:]
    train_y = train_y_target[:train_size]
    test_y = train_y_target[train_size:]

    # Create Dataset
    full_train_dataset = TensorDataset(train_x, train_y)

    # Further split validation set from training set (Train: 85%, Val: 15% of train_size)
    n_train_full = len(full_train_dataset)
    n_val = int(n_train_full * 0.15)
    n_train = n_train_full - n_val
    train_dataset, val_dataset = random_split(full_train_dataset, [n_train, n_val])

    batch_size = 64  # Increased batch size is generally more stable for Transformers

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

    # Model Initialization
    input_dim = train_x.shape[2]

    model = TimeSeriesTransformer(input_dim=input_dim,
                                  d_model=128,  # Model dimension
                                  nhead=4,  # Number of attention heads
                                  num_layers=3,  # Number of encoder layers
                                  num_steps=seq_len,
                                  output_dim=1,
                                  dropout=0.2)  # Increased dropout to prevent overfitting

    # Using Huber Loss (SmoothL1) for increased robustness
    criterion = nn.SmoothL1Loss()

    # Using AdamW optimizer (Adam with Weight Decay) for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Start Advanced Training
    model = train_model_advanced(model, train_loader, val_loader, optimizer, criterion, num_epochs=50, patience=10)

    # Evaluate Model
    evaluate_model(model, test_loader)

    # Save prediction and calculation results
    compute_position_and_spacing_and_save(
        model,
        test_x,
        test_y,
        raw_data_full,
        label_data_full,
        train_size,
        seq_len=seq_len,
        dt=0.1,
        output_file="predictions_extended.xlsx"
    )
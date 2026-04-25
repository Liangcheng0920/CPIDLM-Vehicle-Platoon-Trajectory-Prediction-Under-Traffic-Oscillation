import os  # Import os module for OS interaction, such as file path operations
import torch  # Import PyTorch library for deep learning
import torch.nn as nn  # Import PyTorch neural network module
import torch.optim as optim  # Import PyTorch optimizer module
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import scipy.io as sio  # Import scipy.io for loading .mat data files (MATLAB format)
import pandas as pd  # Import pandas for data processing and analysis
import numpy as np  # Import numpy for numerical computation
import glob  # Used to find file paths

# Set environment variable KMP_DUPLICATE_LIB_OK to TRUE.
# This usually resolves conflicts between Intel MKL and PyTorch's internal libraries,
# allowing duplicate DLLs to be loaded.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Global Path Definitions ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"  # Directory for datasets
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim_lstm_idm_only"  # Directory for saving experimental results

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Device selection: Use CUDA GPU if available, otherwise fallback to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")


# =========================
# Data Validation & Initialization
# =========================
def check_data(data, name="data"):
    """
    Checks the input tensor for NaN (Not a Number) or Inf (Infinity) values.

    Args:
        data (torch.Tensor): PyTorch tensor to be checked.
        name (str): Label for the data source used in logging.
    """
    print(f"Checking {name} for NaN or Inf values...")
    has_nan = torch.isnan(data).any().item()
    has_inf = torch.isinf(data).any().item()
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    if has_nan or has_inf:
        print(f"Warning: {name} contains invalid (NaN/Inf) values!")


def initialize_weights(model):
    """
    Initializes model weights using Xavier Uniform initialization and sets biases to zero.
    Good initialization helps with convergence, especially for tanh/sigmoid activation functions.

    Args:
        model (nn.Module): PyTorch model to initialize.
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            if param.data.dim() > 1:
                nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)


# =========================
# 1. Hybrid IDM Model (LSTM-based Intelligent Driver Model)
# =========================
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dt=0.1):
        """
        Initializes the Hybrid IDM model.
        Args:
            input_dim (int): Input feature dimension for the LSTM.
            hidden_dim (int): Hidden state dimension for the LSTM.
            num_layers (int): Number of LSTM layers.
            dt (float): Simulation time step (seconds).
        """
        super(HybridIDMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 6)  # Outputs 6 IDM parameters
        self.softplus = nn.Softplus()  # Ensures parameters are positive
        self.delta_t = dt

    def forward(self, x):
        """
        Forward pass. LSTM processes temporal info; FC layer outputs IDM parameters.
        Args:
            x (torch.Tensor): Historical sequence data (batch_size, seq_len, input_dim).
        Returns:
            torch.Tensor: Activated IDM parameters (batch_size, 6).
        """
        out, (hn, cn) = self.lstm(x)
        # Use the hidden state of the last time step
        params_raw = self.fc(out[:, -1, :])
        # Softplus ensures physical plausibility (non-negative values)
        params_activated = self.softplus(params_raw)
        return params_activated

    def predict_speed(self, x, s_actual):
        """
        Predicts the follower's speed at the next time step using the IDM formula.
        Args:
            x (torch.Tensor): Historical input sequence (v_f, s, dv, a_f, v_l).
            s_actual (torch.Tensor): Current actual gap distance.
        Returns:
            torch.Tensor: Predicted speed for the next time step (batch, 1).
            torch.Tensor: IDM parameters (batch, 6).
        """
        params = self.forward(x)
        v_n = x[:, -1, 0]  # Current follower speed
        delta_v_hist = x[:, -1, 2]  # Current relative speed (v_leader - v_follower)

        # Parse and clamp IDM parameters to reasonable ranges for stability
        v_des_raw, T_raw, a_max_raw, b_safe_raw, delta_idm_raw, s0_raw = \
            params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]

        v_des = torch.clamp(v_des_raw, min=0.1, max=50.0)  # Desired speed (m/s)
        T = torch.clamp(T_raw, min=0.1, max=5.0)  # Safety time headway (s)
        a_max = torch.clamp(a_max_raw, min=0.1, max=5.0)  # Max acceleration (m/s^2)
        b_safe = torch.clamp(b_safe_raw, min=0.1, max=9.0)  # Comfortable deceleration (m/s^2)
        delta_idm = torch.clamp(delta_idm_raw, min=1.0, max=10.0)  # Acceleration exponent
        s0 = torch.clamp(s0_raw, min=0.0, max=10.0)  # Minimum jam distance (m)
        s_actual_clamped = torch.clamp(s_actual, min=0.5)

        # IDM Calculation
        # Desired dynamic gap s_star
        sqrt_ab_clamped = torch.clamp(torch.sqrt(a_max * b_safe), min=1e-6)
        # Note: interaction term uses relative velocity (v_f - v_l), which is -delta_v_hist
        interaction_term = (v_n * (-delta_v_hist)) / (2 * sqrt_ab_clamped + 1e-9)
        s_star = s0 + torch.clamp(v_n * T, min=0.0) + interaction_term
        s_star = torch.clamp(s_star, min=s0)

        # Acceleration calculation
        v_n_clamped = torch.clamp(v_n, min=0.0)
        speed_ratio = (v_n_clamped + 1e-6) / (v_des + 1e-6)
        term_speed_ratio = speed_ratio.pow(delta_idm)
        spacing_ratio = s_star / (s_actual_clamped + 1e-6)
        term_spacing_ratio = spacing_ratio.pow(2)

        accel_component = 1.0 - term_speed_ratio - term_spacing_ratio
        a_idm_val = a_max * accel_component  # Resulting IDM acceleration

        # Speed update: v(t+dt) = v(t) + a(t)*dt
        v_follow = v_n + a_idm_val * self.delta_t
        v_follow = torch.clamp(v_follow, min=0.0, max=60.0)  # Clamp to reasonable highway speeds

        if torch.isnan(v_follow).any() or torch.isinf(v_follow).any():
            print("Warning: NaN/Inf detected in HybridIDMModel.predict_speed.")

        return v_follow.unsqueeze(1), params


# =========================
# 2. LNN Model (Liquid Neural Network) for Leader Prediction
# =========================
class LiquidCellMulti(nn.Module):  # Single LNN Cell
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        super(LiquidCellMulti, self).__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt  # Time step for ODE solver
        # Linear layers without bias
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Hidden-to-Hidden
        self.W_u = nn.Linear(input_dim, hidden_dim, bias=False)  # Input-to-Hidden
        self.bias = nn.Parameter(torch.zeros(hidden_dim))  # Learnable bias
        self.act = nn.Tanh()

    def forward(self, u, h):
        # ODE: h_dot = -h + act(W_h*h + W_u*u + bias)
        # Euler Integration: h_new = h_old + dt * h_dot
        if h.shape[-1] != self.hidden_dim:
            h = torch.zeros(u.shape[0], self.hidden_dim, device=u.device)
        dh = -h + self.act(self.W_h(h) + self.W_u(u) + self.bias)
        return h + self.dt * dh


class LiquidNeuralNetworkMultiStep(nn.Module):  # Multi-step Leader Prediction
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1, num_steps=50, dt=0.1):
        super(LiquidNeuralNetworkMultiStep, self).__init__()
        self.input_dim = input_dim
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(LiquidCellMulti(current_input_dim, hidden_dim, dt=dt))
        self.fc = nn.Linear(hidden_dim, prediction_steps)  # Final prediction layer
        self.num_steps = num_steps  # Internal simulation steps
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch, seq, features = x.shape
        h_states = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        effective_seq_len = min(seq, self.num_steps)

        for t in range(effective_seq_len):  # Iterate through sequence
            u_t_layer = x[:, t, :]
            for i in range(self.num_layers):  # Propagate through LNN layers
                input_signal_for_cell = h_states[i - 1] if i > 0 else u_t_layer
                h_states[i] = self.cells[i](input_signal_for_cell, h_states[i])
        return self.fc(h_states[-1])  # Predict using last hidden state

    def predict_speed(self, x):
        return self.forward(x)


# =========================
# Training Functions
# =========================
def train_generic_model(model, loader, optimizer, criterion, epochs=30, model_name="Generic Model", clip_value=1.0):
    """
    Generic training function for the Leader LNN model.
    """
    model.train()  # Set to training mode
    for ep in range(epochs):
        tot_loss = 0
        num_batches_processed = 0

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Predict using the custom predict_speed method or standard forward
            pred = model.predict_speed(x_batch) if hasattr(model, 'predict_speed') else model(x_batch)
            loss = criterion(pred, y_batch)

            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: {model_name} Epoch {ep + 1} produced NaN/Inf loss. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            tot_loss += loss.item()
            num_batches_processed += 1

        avg_loss = tot_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
        print(f"[{model_name}] Epoch {ep + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        if np.isnan(avg_loss) and ep > 0:
            print(f"Warning: {model_name} average loss is NaN. Stopping training.")
            break
    return model


def precompute_leader_trajectories_for_idm_training(
        leader_model, raw_data_slice, pred_steps_K, dt, device, hist_len=50
):
    """
    Precomputes leader trajectories (velocity and position) to speed up IDM training.
    """
    leader_model.eval()
    num_samples = raw_data_slice.shape[0]

    if num_samples == 0:
        _IDM_INPUT_DIM_placeholder = 5
        empty_tensor_k = torch.empty(0, pred_steps_K, dtype=torch.float32, device=device)
        return torch.empty(0, hist_len, _IDM_INPUT_DIM_placeholder, device=device), \
            torch.empty(0, device=device), torch.empty(0, device=device), \
            empty_tensor_k, empty_tensor_k, torch.empty(0, device=device)

    # Prepare IDM initial input sequence (Features: v_f, s, dv, a_f, v_l)
    initial_idm_input_seqs = raw_data_slice[:, -hist_len:, [0, 1, 2, 3, 5]].clone() * 0.3048  # Convert feet to meters
    initial_follower_poses = raw_data_slice[:, -1, 4].clone() * 0.3048
    initial_leader_poses_val = raw_data_slice[:, -1, -1].clone() * 0.3048
    initial_s_safes = initial_idm_input_seqs[:, -1, 1].clone()

    # Calculate d1 offset: difference between coordinate distance and gap distance
    batch_d1 = initial_leader_poses_val - initial_follower_poses - initial_s_safes

    # Prepare Leader LNN input (Features: v_l, a_l)
    leader_hist_for_lnn = raw_data_slice[:, -hist_len:, [5, 6]].clone() * 0.3048

    pred_leader_speeds_K_list = []
    pred_leader_pos_K_list = []
    current_dt = dt if dt > 1e-6 else 1e-6

    with torch.no_grad():
        all_pred_l_speeds_k_steps_tensor = leader_model.predict_speed(leader_hist_for_lnn.to(device)).cpu()

        for i in range(num_samples):
            pred_l_speeds_k_i = all_pred_l_speeds_k_steps_tensor[i]
            pred_leader_speeds_K_list.append(pred_l_speeds_k_i)

            # Iteratively calculate leader positions based on predicted speeds
            current_l_pos = initial_leader_poses_val[i].item()
            prev_l_v = leader_hist_for_lnn[i, -1, 0].item()
            l_pos_k_steps = []

            for k_idx in range(pred_steps_K):
                vp = pred_l_speeds_k_i[k_idx].item()
                a_leader = (vp - prev_l_v) / current_dt
                displacement = prev_l_v * current_dt + 0.5 * a_leader * current_dt ** 2
                next_l_pos = current_l_pos + displacement
                l_pos_k_steps.append(next_l_pos)
                prev_l_v, current_l_pos = vp, next_l_pos

            pred_leader_pos_K_list.append(torch.tensor(l_pos_k_steps, dtype=torch.float32))

    pred_leader_speeds_K = torch.stack(pred_leader_speeds_K_list)
    pred_leader_pos_K = torch.stack(pred_leader_pos_K_list)

    return initial_idm_input_seqs.to(device), initial_follower_poses.to(device), \
        initial_s_safes.to(device), pred_leader_speeds_K.to(device), \
        pred_leader_pos_K.to(device), batch_d1.to(device)


def train_idm_model_multistep(
        model, train_loader, optimizer, num_epochs=30, pred_steps_K=5, dt=0.1,
        alpha_decay=0.0, teacher_forcing_initial_ratio=1.0, min_teacher_forcing_ratio=0.0,
        teacher_forcing_decay_epochs_ratio=0.75, clip_value=1.0
):
    """
    Trains LSTM-IDM using multi-step prediction and Scheduled Sampling (Teacher Forcing).
    """
    model.train()
    criterion_mse_elementwise = nn.MSELoss(reduction='none')
    # Decay weights for further prediction steps
    loss_weights = torch.exp(-alpha_decay * torch.arange(pred_steps_K, device=device).float())
    decay_epochs = int(num_epochs * teacher_forcing_decay_epochs_ratio)
    current_dt = dt if dt > 1e-6 else 1e-6

    for epoch in range(num_epochs):
        total_loss_epoch, num_valid_batches = 0, 0

        # Linearly decay Teacher Forcing ratio
        current_tf_ratio = teacher_forcing_initial_ratio - \
                           (teacher_forcing_initial_ratio - min_teacher_forcing_ratio) * \
                           (float(epoch) / decay_epochs if decay_epochs > 0 else 0)
        current_tf_ratio = max(min_teacher_forcing_ratio, current_tf_ratio)
        print(f"[LSTM-IDM Training] Epoch [{epoch + 1}/{num_epochs}], TF Ratio: {current_tf_ratio:.4f}")

        for batch_idx, batch_data in enumerate(train_loader):
            (initial_idm_input, true_f_speeds_K, initial_f_pos, initial_s_safe,
             pred_l_speeds_K, pred_l_pos_K, d1_offset, true_f_features_K, true_f_pos_K) = batch_data

            optimizer.zero_grad()
            batch_current_idm_input = initial_idm_input.clone()
            batch_current_f_speed = batch_current_idm_input[:, -1, 0].clone()
            batch_current_f_pos = initial_f_pos.clone()
            batch_current_s_actual = initial_s_safe.clone()

            all_preds = []
            skip_batch = False

            for k_step in range(pred_steps_K):
                # Stability check
                if torch.isnan(batch_current_idm_input).any() or torch.isinf(batch_current_idm_input).any():
                    skip_batch = True;
                    break

                # Predict single step
                v_f_next_pred_unsq, _ = model.predict_speed(batch_current_idm_input, batch_current_s_actual)
                v_f_next_pred = v_f_next_pred_unsq.squeeze(1)

                if torch.isnan(v_f_next_pred).any():
                    skip_batch = True;
                    break

                all_preds.append(v_f_next_pred.unsqueeze(1))

                # Prepare next time step input
                if k_step < pred_steps_K - 1:
                    use_gt = torch.rand(1).item() < current_tf_ratio
                    v_l_next = pred_l_speeds_K[:, k_step]

                    if use_gt:
                        new_feature_slice = torch.stack([
                            true_f_features_K[:, k_step, 0], true_f_features_K[:, k_step, 1],
                            true_f_features_K[:, k_step, 2], true_f_features_K[:, k_step, 3], v_l_next
                        ], dim=1)
                        batch_current_f_speed = true_f_features_K[:, k_step, 0].clone()
                        batch_current_f_pos = true_f_pos_K[:, k_step].clone()
                        batch_current_s_actual = true_f_features_K[:, k_step, 1].clone()
                    else:
                        a_f_pred = torch.clamp((v_f_next_pred - batch_current_f_speed) / current_dt, -10.0, 10.0)
                        disp_f = batch_current_f_speed * current_dt + 0.5 * a_f_pred * current_dt ** 2
                        pos_f_next = batch_current_f_pos + disp_f
                        s_next = torch.clamp(pred_l_pos_K[:, k_step] - pos_f_next - d1_offset, min=0.1)
                        dv_next = v_l_next - v_f_next_pred

                        new_feature_slice = torch.stack([v_f_next_pred, s_next, dv_next, a_f_pred, v_l_next], dim=1)
                        batch_current_f_speed, batch_current_f_pos, batch_current_s_actual = v_f_next_pred.clone(), pos_f_next.clone(), s_next.clone()

                    batch_current_idm_input = torch.cat(
                        [batch_current_idm_input[:, 1:, :], new_feature_slice.unsqueeze(1)], dim=1)

            if skip_batch: continue

            batch_pred_speeds = torch.cat(all_preds, dim=1)
            sq_errors = criterion_mse_elementwise(batch_pred_speeds, true_f_speeds_K)
            loss = (sq_errors * loss_weights.unsqueeze(0)).sum(dim=1).mean()

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                total_loss_epoch += loss.item()
                num_valid_batches += 1

        avg_loss = total_loss_epoch / num_valid_batches if num_valid_batches > 0 else float('nan')
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    return model


# =========================
# Evaluation Functions
# =========================
def evaluate_generic_model(model, test_loader, pred_steps=5, model_name="Generic Model", device_eval=None):
    """
    Standard evaluation for Leader LNN.
    """
    model.eval()
    all_predicted, all_true = [], []
    if not test_loader or len(test_loader.dataset) == 0: return

    with torch.no_grad():
        for b_data, b_target in test_loader:
            b_data, b_target = b_data.to(device_eval), b_target.to(device_eval)
            pred = model.predict_speed(b_data) if hasattr(model, 'predict_speed') else model(b_data)
            all_predicted.append(pred.cpu())
            all_true.append(b_target.cpu())

    y_pred = torch.cat(all_predicted, dim=0).numpy()
    y_true = torch.cat(all_true, dim=0).numpy()

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"\n{model_name} Evaluation: RMSE: {rmse:.4f}")


def get_idm_multistep_predictions(idm_model, leader_model, initial_idm_input, raw_data, pred_steps, dt, hist_len, dev):
    """
    Inference loop for LSTM-IDM multi-step prediction.
    """
    idm_model.eval();
    leader_model.eval()
    (_, initial_f_pos, initial_s_safe, pred_l_speeds, pred_l_pos, d1_offset) = \
        precompute_leader_trajectories_for_idm_training(leader_model, raw_data, pred_steps, dt, dev, hist_len)

    cur_idm_input = initial_idm_input.to(dev)
    cur_f_speed = cur_idm_input[:, -1, 0].clone()
    cur_f_pos = initial_f_pos.clone()
    cur_s_actual = initial_s_safe.clone()
    all_preds = []
    dt_val = dt if dt > 1e-6 else 1e-6

    with torch.no_grad():
        for k in range(pred_steps):
            v_f_pred_unsq, _ = idm_model.predict_speed(cur_idm_input, cur_s_actual)
            v_f_pred = v_f_pred_unsq.squeeze(1)
            all_preds.append(v_f_pred.unsqueeze(1))

            if k < pred_steps - 1:
                v_l_next = pred_l_speeds[:, k]
                a_f_next = torch.clamp((v_f_pred - cur_f_speed) / dt_val, -10.0, 10.0)
                disp_f = cur_f_speed * dt_val + 0.5 * a_f_next * dt_val ** 2
                pos_f_next = cur_f_pos + disp_f
                s_next = torch.clamp(pred_l_pos[:, k] - pos_f_next - d1_offset, min=0.1)
                new_slice = torch.stack([v_f_pred, s_next, v_l_next - v_f_pred, a_f_next, v_l_next], dim=1)
                cur_idm_input = torch.cat([cur_idm_input[:, 1:, :], new_slice.unsqueeze(1)], dim=1)
                cur_f_speed, cur_f_pos, cur_s_actual = v_f_pred, pos_f_next, s_next

    return torch.cat(all_preds, dim=1)


def evaluate_final_lstm_idm_model(idm_model, leader_model, raw_test, label_test, dt, pred_steps, hist_len, dev,
                                  output_excel_filepath, excel_sheet_name):
    """
    Evaluates the final Hybrid LSTM-IDM model on the test set and saves results to Excel.
    """
    idm_model.eval();
    leader_model.eval()
    N_test = raw_test.shape[0]
    if N_test == 0: return None, None, None, None, None, None

    idm_hist_test = raw_test[:, -hist_len:, [0, 1, 2, 3, 5]].clone() * 0.3048
    y_pred_speeds = get_idm_multistep_predictions(idm_model, leader_model, idm_hist_test, raw_test, pred_steps, dt,
                                                  hist_len, dev).cpu().numpy()
    y_true_speeds = label_test[:, :pred_steps, 0].cpu().numpy() * 0.3048

    # Calculate Speed Metrics
    rmse_speed = np.sqrt(np.mean((y_pred_speeds - y_true_speeds) ** 2))
    mae_speed = np.mean(np.abs(y_pred_speeds - y_true_speeds))
    print(f"Final LSTM-IDM Speed Metrics: RMSE: {rmse_speed:.4f}, MAE: {mae_speed:.4f}")

    # Position Inference
    initial_f_v = raw_test[:, -1, 0].cpu().numpy() * 0.3048
    initial_f_p = raw_test[:, -1, 4].cpu().numpy() * 0.3048
    y_pred_pos = np.zeros_like(y_pred_speeds)

    for i in range(N_test):
        cv, cp = initial_f_v[i], initial_f_p[i]
        for k in range(pred_steps):
            accel = (y_pred_speeds[i, k] - cv) / dt
            disp = cv * dt + 0.5 * accel * dt ** 2
            cp += disp
            y_pred_pos[i, k] = cp
            cv = y_pred_speeds[i, k]

    y_true_pos = label_test[:, :pred_steps, 3].cpu().numpy() * 0.3048
    rmse_pos = np.sqrt(np.mean((y_pred_pos - y_true_pos) ** 2))

    # Save to Excel
    df = pd.DataFrame({"True_Speed_Step1": y_true_speeds[:, 0], "Pred_Speed_Step1": y_pred_speeds[:, 0]})
    df.to_excel(output_excel_filepath, sheet_name=excel_sheet_name, index=False)

    return 0, rmse_speed, mae_speed, 0, rmse_pos, 0


# =========================
# Main Execution Block
# =========================
if __name__ == "__main__":
    torch.manual_seed(42);
    np.random.seed(42)
    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))

    all_datasets_metrics_summary = []

    for data_file in data_files:
        dataset_name = os.path.basename(data_file).replace(".mat", "")
        print(f"\nProcessing Dataset: {dataset_name}")

        data = sio.loadmat(data_file)
        DT, HIST_LEN = 0.1, 50
        raw_full = torch.tensor(data['train_data'], dtype=torch.float32)
        label_full = torch.tensor(data['lable_data'] if 'lable_data' in data else data['label_data'],
                                  dtype=torch.float32)

        num_use = int(raw_full.shape[0] * 0.2)
        raw_train = raw_full[:int(num_use * 0.8)];
        raw_test = raw_full[int(num_use * 0.8):num_use]
        label_train = label_full[:int(num_use * 0.8)];
        label_test = label_full[int(num_use * 0.8):num_use]

        # 1. Leader LNN Training
        PRED_K = label_train.shape[1]
        leader_model = LiquidNeuralNetworkMultiStep(2, 64, PRED_K, 1, HIST_LEN, DT).to(device)
        initialize_weights(leader_model)
        opt_l = optim.Adam(leader_model.parameters(), lr=1e-3)

        train_l_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(raw_train[:, -HIST_LEN:, [5, 6]] * 0.3048, label_train[:, :, 4] * 0.3048),
            batch_size=32, shuffle=True)
        train_generic_model(leader_model, train_l_loader, opt_l, nn.MSELoss(), 50, "Leader LNN")

        # 2. LSTM-IDM Training
        idm_model = HybridIDMModel(5, 64, 1, DT).to(device)
        initialize_weights(idm_model)
        opt_idm = optim.Adam(idm_model.parameters(), lr=2e-4)

        (init_idm_seq, init_f_p, init_s, p_l_v, p_l_p, d1) = precompute_leader_trajectories_for_idm_training(
            leader_model, raw_train, PRED_K, DT, device, HIST_LEN)

        train_idm_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(init_idm_seq, label_train[:, :PRED_K, 0] * 0.3048, init_f_p, init_s, p_l_v,
                                           p_l_p, d1,
                                           torch.stack([label_train[:, :PRED_K, 0], label_train[:, :PRED_K, 1],
                                                        label_train[:, :PRED_K, 4] - label_train[:, :PRED_K, 0],
                                                        label_train[:, :PRED_K, 2]], dim=2) * 0.3048,
                                           label_train[:, :PRED_K, 3] * 0.3048),
            batch_size=32, shuffle=True)

        train_idm_model_multistep(idm_model, train_idm_loader, opt_idm, 50, PRED_K, DT)

        # 3. Final Evaluation
        excel_path = os.path.join(RESULTS_DIR, f"results_{dataset_name}.xlsx")
        evaluate_final_lstm_idm_model(idm_model, leader_model, raw_test, label_test, DT, PRED_K, HIST_LEN, device,
                                      excel_path, dataset_name)

    print("\n--- Processing Complete ---")
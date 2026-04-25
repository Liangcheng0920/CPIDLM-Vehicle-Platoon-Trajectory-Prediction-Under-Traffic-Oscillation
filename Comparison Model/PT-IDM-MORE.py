import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import glob

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Global Path Definitions ---
DATA_DIR = "E:\\pythonProject1\\data_ngsim"
RESULTS_DIR = "E:\\pythonProject1\\results_ngsim_transformer_idm_only"  # Changed results directory name

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Automatically select computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")


# =========================
# Data Check and Initialization Functions
# =========================
def check_data(data, name="data"):
    """
    Checks for NaN or Inf values in the input data.
    """
    print(f"Checking {name} for NaN or Inf values...")
    has_nan = torch.isnan(data).any().item()
    has_inf = torch.isinf(data).any().item()
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    if has_nan or has_inf:
        print(f"Warning: {name} contains NaN or Inf values!")


def initialize_weights(model):
    """
    Initializes weights using Xavier uniform distribution and biases to 0.
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            if param.data.dim() > 1:
                nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)


# =========================
# 1. Hybrid IDM Model with Transformer (Replaces LSTM)
# =========================
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, dim_feedforward, dt=0.1, dropout=0.1):
        """
        Initialization for Hybrid IDM model using a Transformer Encoder.
        Args:
            input_dim (int): Transformer input feature dimension.
            model_dim (int): The number of expected features in the encoder inputs (d_model).
            nhead (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dt (float): Simulation time step (seconds).
            dropout (float): The dropout value.
        """
        super(HybridIDMModel, self).__init__()
        self.model_dim = model_dim
        # Linear layer to project input_dim to model_dim if they are different
        self.input_proj = nn.Linear(input_dim, model_dim) if input_dim != model_dim else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc = nn.Linear(model_dim, 6)  # Output IDM's 6 parameters
        self.softplus = nn.Softplus()  # Softplus ensures parameters are positive
        self.delta_t = dt

    def forward(self, x):
        """
        Forward pass for the Transformer-based IDM parameter prediction.
        Args:
            x (torch.Tensor): Input historical sequence data, shape (batch_size, seq_len, input_dim).
        Returns:
            torch.Tensor: IDM parameters after Softplus activation, shape (batch_size, 6).
        """
        # Project input features to model_dim if necessary
        x = self.input_proj(x)

        # Transformer Encoder expects (batch_size, seq_len, d_model)
        transformer_output = self.transformer_encoder(x)

        # Take the output of the last time step for IDM parameter prediction
        # Alternatively, one could use pooling (e.g., mean, max) or an attention mechanism
        params_raw = self.fc(transformer_output[:, -1, :])

        # Apply Softplus to ensure parameters are physically meaningful (e.g., positive)
        params_activated = self.softplus(params_raw)
        return params_activated

    def predict_speed(self, x, s_actual):
        """
        Predicts the speed of the following vehicle at the next time step using the IDM formula.
        Args:
            x (torch.Tensor): Historical input sequence data for the following vehicle (batch, seq_len, input_dim).
                              Features: [Ego Speed, Actual Spacing (historical), Speed Difference with leader, Ego Accel, Leader Speed]
            s_actual (torch.Tensor): Current actual observed spacing (batch,).
        Returns:
            torch.Tensor: Predicted speed of the following vehicle at the next time step (batch, 1).
            torch.Tensor: IDM model parameters (batch, 6), obtained from the network and clamped.
        """
        params = self.forward(x)  # Get IDM parameters predicted by the neural network
        v_n = x[:, -1, 0]  # Current ego speed (last time step of sequence)
        delta_v_hist = x[:, -1, 2]  # Current speed difference (v_leader - v_follower)

        # Parse and clamp IDM parameters to a reasonable range for stability
        v_des_raw, T_raw, a_max_raw, b_safe_raw, delta_idm_raw, s0_raw = \
            params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]

        v_des = torch.clamp(v_des_raw, min=0.1, max=50.0)  # Desired speed (m/s)
        T = torch.clamp(T_raw, min=0.1, max=5.0)  # Safe time headway (s)
        a_max = torch.clamp(a_max_raw, min=0.1, max=5.0)  # Max acceleration (m/s^2)
        b_safe = torch.clamp(b_safe_raw, min=0.1, max=9.0)  # Comfortable deceleration (m/s^2)
        delta_idm = torch.clamp(delta_idm_raw, min=1.0, max=10.0)  # Acceleration exponent
        s0 = torch.clamp(s0_raw, min=0.0, max=10.0)  # Minimum standstill distance (m)
        s_actual_clamped = torch.clamp(s_actual, min=0.5)  # Clamp actual spacing to prevent too small values

        # IDM formula calculation
        # Desired dynamic spacing s_star
        sqrt_ab_clamped = torch.clamp(torch.sqrt(a_max * b_safe), min=1e-6)  # Prevent square root of zero
        # Speed difference in IDM interaction term (v_n - v_l), corresponding to -delta_v_hist
        interaction_term = (v_n * (-delta_v_hist)) / (
                    2 * sqrt_ab_clamped + 1e-9)  # Add small epsilon to prevent division by zero
        s_star = s0 + torch.clamp(v_n * T, min=0.0) + interaction_term  # v_n * T should be non-negative
        s_star = torch.clamp(s_star, min=s0)  # s_star must be at least s0

        # Acceleration calculation
        v_n_clamped = torch.clamp(v_n, min=0.0)  # Current speed non-negative
        speed_ratio = (v_n_clamped + 1e-6) / (v_des + 1e-6)  # Add small epsilon to prevent division by zero
        term_speed_ratio = speed_ratio.pow(delta_idm)
        spacing_ratio = s_star / (s_actual_clamped + 1e-6)  # Add small epsilon to prevent division by zero
        term_spacing_ratio = spacing_ratio.pow(2)

        accel_component = 1.0 - term_speed_ratio - term_spacing_ratio
        a_idm_val = a_max * accel_component  # IDM calculated acceleration

        # Speed update: v(t+dt) = v(t) + a(t)*dt
        v_follow = v_n + a_idm_val * self.delta_t
        v_follow = torch.clamp(v_follow, min=0.0, max=60.0)  # Clamped predicted speed to reasonable range (0 ~ 216km/h)

        # NaN/Inf check for debugging
        if torch.isnan(v_follow).any() or torch.isinf(v_follow).any():
            print("Warning: NaN/Inf output detected in HybridIDMModel.predict_speed.")
            # Can add more detailed parameter printing here to help pinpoint the issue
        return v_follow.unsqueeze(1), params


# =========================
# 2. LNN Model (Liquid Neural Network) - Base class and implementation for leader prediction
# =========================
class LiquidCellMulti(nn.Module):  # Single LNN neuron cell
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        super(LiquidCellMulti, self).__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt  # ODE solver time step
        # Linear transformation layers, without bias (bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Hidden state to hidden state
        self.W_u = nn.Linear(input_dim, hidden_dim, bias=False)  # Input to hidden state
        self.bias = nn.Parameter(torch.zeros(hidden_dim))  # Learnable bias
        self.act = nn.Tanh()  # Activation function

    def forward(self, u, h):
        # h_dot = -h + act(W_h*h + W_u*u + bias)
        # h_new = h_old + dt * h_dot (Euler method)
        if h.shape[-1] != self.hidden_dim:  # Initialize hidden state (usually at the start of sequence)
            h = torch.zeros(u.shape[0], self.hidden_dim, device=u.device)
        dh = -h + self.act(self.W_h(h) + self.W_u(u) + self.bias)
        return h + self.dt * dh


class LiquidNeuralNetworkMultiStep(nn.Module):  # For Leader prediction
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1, num_steps=50, dt=0.1):
        super(LiquidNeuralNetworkMultiStep, self).__init__()
        self.input_dim = input_dim
        self.cells = nn.ModuleList()  # Store LNN layers
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(LiquidCellMulti(current_input_dim, hidden_dim, dt=dt))
        self.fc = nn.Linear(hidden_dim, prediction_steps)  # Output layer
        self.num_steps = num_steps  # Number of internal LNN simulation steps, usually equal to input sequence length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch, seq, features = x.shape
        h_states = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        effective_seq_len = min(seq, self.num_steps)  # Actual sequence length processed by LNN

        for t in range(effective_seq_len):  # Iterate along the time sequence
            u_t_layer = x[:, t, :]  # Input at current time step
            for i in range(self.num_layers):  # Iterate through each LNN layer
                input_signal_for_cell = h_states[
                    i - 1] if i > 0 else u_t_layer  # First layer takes raw signal, subsequent layers take previous layer's hidden state
                h_states[i] = self.cells[i](input_signal_for_cell, h_states[i])
        return self.fc(h_states[-1])  # Use the final hidden state of the last LNN layer for prediction

    def predict_speed(self, x):  # Convenience method
        return self.forward(x)


# =========================
# Training Function Definition
# =========================
def train_generic_model(model, loader, optimizer, criterion, epochs=30, model_name="Generic Model", clip_value=1.0):
    """
    General training function for models like the Leader LNN.
    """
    model.train()  # Set model to training mode
    for ep in range(epochs):  # Iterate through training epochs
        tot_loss = 0  # Total loss for current epoch
        num_batches_processed = 0  # Number of batches processed in current epoch

        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move data to device
            optimizer.zero_grad()
            pred = model.predict_speed(x_batch) if hasattr(model, 'predict_speed') else model(x_batch)
            loss = criterion(pred, y_batch)  # Calculate loss

            # NaN/Inf loss check
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: {model_name} encountered NaN/Inf loss in epoch {ep + 1}. Skipping this batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            tot_loss += loss.item()
            num_batches_processed += 1

        # Calculate and print average loss
        avg_loss = tot_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
        print(f"[{model_name}] Epoch {ep + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        # If NaN average loss occurs after the first epoch, it might indicate a serious issue, stop early
        if np.isnan(avg_loss) and ep > 0:
            print(f"Warning: {model_name} average loss is NaN, training stopped early.")
            break
    return model


def precompute_leader_trajectories_for_idm_training(
        leader_model, raw_data_slice, pred_steps_K, dt, device, hist_len=50
):
    """
    Precomputes leader trajectories (speed and position) for IDM model training.
    This avoids redundant leader trajectory prediction in each IDM training iteration, improving efficiency.
    """
    leader_model.eval()  # Set leader model to evaluation mode
    num_samples = raw_data_slice.shape[0]

    # Handle empty input case, return tensors with correct shapes but empty
    if num_samples == 0:
        _IDM_INPUT_DIM_placeholder = 5  # Should be consistent with global definition
        empty_tensor_k_steps = torch.empty(0, pred_steps_K, dtype=torch.float32, device=device)
        empty_tensor_idm_input = torch.empty(0, hist_len, _IDM_INPUT_DIM_placeholder, dtype=torch.float32,
                                             device=device)
        empty_tensor_scalar_batch = torch.empty(0, dtype=torch.float32, device=device)
        return empty_tensor_idm_input, empty_tensor_scalar_batch, empty_tensor_scalar_batch, \
            empty_tensor_k_steps, empty_tensor_k_steps, empty_tensor_scalar_batch

    # Prepare initial input sequences for the IDM model (features: v_f, s, dv, a_f, v_l)
    initial_idm_input_seqs = raw_data_slice[:, -hist_len:, [0, 1, 2, 3, 5]].clone() * 0.3048  # Unit conversion
    initial_follower_poses = raw_data_slice[:, -1, 4].clone() * 0.3048  # Ego initial position
    initial_leader_poses_val = raw_data_slice[:, -1, -1].clone() * 0.3048  # Leader initial position
    initial_s_safes = initial_idm_input_seqs[:, -1, 1].clone()  # Initial actual spacing (already converted units)
    # d1 is initial leader_pos - follower_pos - s_safe, for subsequent spacing correction
    batch_d1 = initial_leader_poses_val - initial_follower_poses - initial_s_safes

    # Prepare input for leader LNN model (features: v_l, a_l)
    leader_hist_for_lnn = raw_data_slice[:, -hist_len:, [5, 6]].clone() * 0.3048  # Unit conversion

    pred_leader_speeds_K_list = []  # Store predicted leader speed sequences for each sample
    pred_leader_pos_K_list = []  # Store predicted leader position sequences for each sample
    current_dt = dt if dt > 1e-6 else 1e-6  # Ensure dt is valid

    with torch.no_grad():  # Do not calculate gradients
        # Batch predict leader's future K-step speeds
        all_pred_l_speeds_k_steps_tensor = leader_model.predict_speed(
            leader_hist_for_lnn.to(device)).cpu()  # Move to CPU for processing

        for i in range(num_samples):  # Iterate through each sample
            pred_l_speeds_k_steps_tensor_i = all_pred_l_speeds_k_steps_tensor[
                i]  # Predicted speed sequence for current sample
            pred_leader_speeds_K_list.append(pred_l_speeds_k_steps_tensor_i)

            # Iteratively calculate leader's future K-step positions based on predicted speeds
            current_l_pos = initial_leader_poses_val[i].item()  # Current leader position (t=0)
            prev_l_v = leader_hist_for_lnn[i, -1, 0].item()  # Leader's speed at the end of history (v at t=0)
            l_pos_k_steps = []  # Store future K-step positions for current sample

            for k_idx in range(pred_steps_K):  # Iterate through future K steps
                vp = pred_l_speeds_k_steps_tensor_i[k_idx].item()  # Predicted v(t+k+1)
                a_leader = (vp - prev_l_v) / current_dt  # Average acceleration
                displacement_leader = prev_l_v * current_dt + 0.5 * a_leader * current_dt * current_dt  # Displacement
                next_l_pos = current_l_pos + displacement_leader  # New position
                l_pos_k_steps.append(next_l_pos)

                prev_l_v = vp  # Update speed for next step calculation
                current_l_pos = next_l_pos  # Update position
            pred_leader_pos_K_list.append(torch.tensor(l_pos_k_steps, dtype=torch.float32))

    # Stack tensors from lists
    pred_leader_speeds_K = torch.stack(pred_leader_speeds_K_list) if num_samples > 0 else torch.empty(0, pred_steps_K,
                                                                                                      dtype=torch.float32)
    pred_leader_pos_K = torch.stack(pred_leader_pos_K_list) if num_samples > 0 else torch.empty(0, pred_steps_K,
                                                                                                dtype=torch.float32)

    # Return all calculated tensors, ensuring they are on the correct device (especially for model input)
    return initial_idm_input_seqs.to(device), initial_follower_poses.to(device), initial_s_safes.to(device), \
        pred_leader_speeds_K.to(device), pred_leader_pos_K.to(device), batch_d1.to(device)


def train_idm_model_multistep(
        model, train_loader, optimizer,
        num_epochs=30, pred_steps_K=5, dt=0.1, alpha_decay=0.0,  # alpha_decay controls loss weight
        teacher_forcing_initial_ratio=1.0,  # Initial Teacher Forcing ratio
        min_teacher_forcing_ratio=0.0,  # Minimum Teacher Forcing ratio
        teacher_forcing_decay_epochs_ratio=0.75,  # Epochs ratio for TF decay
        clip_value=1.0  # Gradient clipping value
):
    """ Trains the Transformer-IDM hybrid model with multi-step prediction and scheduled sampling/teacher forcing. """
    model.train()
    criterion_mse_elementwise = nn.MSELoss(reduction='none')  # Element-wise MSE for weighting
    # Loss weights: w_t = exp(-alpha_decay * t), weights decrease for farther prediction steps (if alpha_decay > 0)
    loss_weights = torch.exp(-alpha_decay * torch.arange(pred_steps_K, device=device).float())
    decay_epochs = int(num_epochs * teacher_forcing_decay_epochs_ratio)  # Total epochs for TF decay
    current_dt = dt if dt > 1e-6 else 1e-6  # Ensure dt is valid

    for epoch in range(num_epochs):
        total_loss_epoch = 0
        num_valid_batches = 0  # Count valid (non-NaN) batches

        # Calculate current epoch's Teacher Forcing ratio (linear decay)
        current_teacher_forcing_ratio = teacher_forcing_initial_ratio - \
                                        (teacher_forcing_initial_ratio - min_teacher_forcing_ratio) * \
                                        (float(epoch) / decay_epochs if decay_epochs > 0 else 0)
        current_teacher_forcing_ratio = max(min_teacher_forcing_ratio, current_teacher_forcing_ratio)
        print(
            f"[Transformer-IDM Multi-step Training] Epoch [{epoch + 1}/{num_epochs}], Teacher Forcing Ratio: {current_teacher_forcing_ratio:.4f}")

        # Training data loader provides precomputed batch data
        for batch_idx, (batch_initial_idm_input_seq,  # Initial IDM input history
                        batch_true_follower_speeds_K_steps_for_loss,  # True ego future K-step speeds (for loss)
                        batch_initial_follower_pos,  # Ego initial position
                        batch_initial_s_safe,  # Initial actual spacing
                        batch_pred_leader_speeds_K_steps,  # Predicted leader future K-step speeds
                        batch_pred_leader_pos_K_steps,  # Predicted leader future K-step positions
                        batch_d1_offset,  # Spacing correction d1
                        batch_true_follower_all_features_K_steps,
                        # True ego future K-step all relevant features (for TF)
                        batch_true_follower_pos_K_steps  # True ego future K-step positions (for TF)
                        ) in enumerate(train_loader):

            # Data should already be on device from precompute function or DataLoader, but confirm for labels
            batch_true_follower_speeds_K_steps_for_loss = batch_true_follower_speeds_K_steps_for_loss.to(device)
            batch_true_follower_all_features_K_steps = batch_true_follower_all_features_K_steps.to(device)
            batch_true_follower_pos_K_steps = batch_true_follower_pos_K_steps.to(device)

            optimizer.zero_grad()
            # Initialize state variables needed for recurrent prediction (clone to avoid in-place modification)
            batch_current_idm_input_torch = batch_initial_idm_input_seq.clone()
            batch_current_follower_speed_pred = batch_current_idm_input_torch[:, -1,
                                                0].clone()  # Current ego speed (prediction start)
            batch_current_follower_pos = batch_initial_follower_pos.clone()  # Current ego position
            batch_current_s_actual_for_idm = batch_initial_s_safe.clone()  # Current actual spacing input for IDM
            all_predicted_follower_speeds_batch_list = []  # Store K-step predictions
            skip_batch_update = False  # Flag to skip current batch update due to NaN/Inf

            # Loop prediction over pred_steps_K future time steps
            for k_step in range(pred_steps_K):
                # Input check
                if torch.isnan(batch_current_idm_input_torch).any() or torch.isinf(
                        batch_current_idm_input_torch).any() or \
                        torch.isnan(batch_current_s_actual_for_idm).any() or torch.isinf(
                    batch_current_s_actual_for_idm).any():
                    print(
                        f"Warning: IDM input at E:{epoch + 1}, B:{batch_idx}, K:{k_step} contains NaN/Inf. Skipping this batch.")
                    skip_batch_update = True
                    break

                # IDM model predicts one step (v_follower_t+k+1)
                v_follower_t_plus_k_plus_1_pred_batch_unsqueeze, _ = model.predict_speed(
                    batch_current_idm_input_torch, batch_current_s_actual_for_idm)
                v_follower_t_plus_k_plus_1_pred_batch = v_follower_t_plus_k_plus_1_pred_batch_unsqueeze.squeeze(1)

                # Predicted output check
                if torch.isnan(v_follower_t_plus_k_plus_1_pred_batch).any() or torch.isinf(
                        v_follower_t_plus_k_plus_1_pred_batch).any():
                    print(
                        f"Warning: IDM predicted speed at E:{epoch + 1}, B:{batch_idx}, K:{k_step} is NaN/Inf. Skipping this batch.")
                    skip_batch_update = True
                    break
                all_predicted_follower_speeds_batch_list.append(v_follower_t_plus_k_plus_1_pred_batch.unsqueeze(1))

                # Prepare IDM input for the next time step (k_step+1) (if not the last prediction step)
                if k_step < pred_steps_K - 1:
                    use_ground_truth = torch.rand(
                        1).item() < current_teacher_forcing_ratio  # Decide whether to use ground truth

                    # Get leader's predicted speed and position at t+k+1
                    v_leader_t_plus_k_plus_1_batch = batch_pred_leader_speeds_K_steps[:, k_step]
                    pos_leader_t_plus_k_plus_1_batch = batch_pred_leader_pos_K_steps[:, k_step]

                    if use_ground_truth:  # Teacher Forcing: Use true values to construct next step input
                        v_f_next_true = batch_true_follower_all_features_K_steps[:, k_step,
                                        0]  # True value for k_step, as input state for k_step+1
                        s_actual_next_true = batch_true_follower_all_features_K_steps[:, k_step, 1]
                        delta_v_next_true = batch_true_follower_all_features_K_steps[:, k_step, 2]
                        a_f_next_true = batch_true_follower_all_features_K_steps[:, k_step, 3]
                        pos_f_next_true = batch_true_follower_pos_K_steps[:, k_step]

                        # New feature slice: [v_f_true, s_true, dv_true, a_f_true, v_l_pred]
                        new_feature_slice_batch = torch.stack([
                            v_f_next_true, s_actual_next_true, delta_v_next_true,
                            a_f_next_true, v_leader_t_plus_k_plus_1_batch  # Leader info from LNN prediction
                        ], dim=1)
                        # Update states needed for next IDM prediction (based on true values)
                        batch_current_follower_speed_pred = v_f_next_true.clone()
                        batch_current_follower_pos = pos_f_next_true.clone()
                        batch_current_s_actual_for_idm = s_actual_next_true.clone()
                    else:  # Student Forcing: Use model's own predictions to construct next step input
                        # Calculate ego acceleration: a = (v_pred(t+1) - v_current_pred(t)) / dt
                        a_follower_t_plus_k_plus_1_batch = (
                                                                   v_follower_t_plus_k_plus_1_pred_batch - batch_current_follower_speed_pred) / current_dt
                        a_follower_t_plus_k_plus_1_batch = torch.clamp(a_follower_t_plus_k_plus_1_batch, -10.0,
                                                                       10.0)  # Clamp

                        # Calculate ego displacement and new position
                        disp_follower_batch = batch_current_follower_speed_pred * current_dt + 0.5 * a_follower_t_plus_k_plus_1_batch * current_dt ** 2
                        pos_follower_t_plus_k_plus_1_batch = batch_current_follower_pos + disp_follower_batch

                        # Calculate new spacing
                        spacing_raw_t_plus_k_plus_1 = pos_leader_t_plus_k_plus_1_batch - pos_follower_t_plus_k_plus_1_batch
                        spacing_adjusted_t_plus_k_plus_1 = spacing_raw_t_plus_k_plus_1 - batch_d1_offset  # Correct
                        spacing_adjusted_t_plus_k_plus_1 = torch.clamp(spacing_adjusted_t_plus_k_plus_1,
                                                                       min=0.1)  # Clamp

                        # Calculate new speed difference
                        delta_v_t_plus_k_plus_1_batch = v_leader_t_plus_k_plus_1_batch - v_follower_t_plus_k_plus_1_pred_batch

                        # New feature slice: [v_f_pred, s_pred, dv_pred, a_f_pred, v_l_pred]
                        new_feature_slice_batch = torch.stack([
                            v_follower_t_plus_k_plus_1_pred_batch, spacing_adjusted_t_plus_k_plus_1,
                            delta_v_t_plus_k_plus_1_batch, a_follower_t_plus_k_plus_1_batch,
                            v_leader_t_plus_k_plus_1_batch
                        ], dim=1)
                        # Update states needed for next IDM prediction (based on model's own prediction)
                        batch_current_follower_speed_pred = v_follower_t_plus_k_plus_1_pred_batch.clone()
                        batch_current_follower_pos = pos_follower_t_plus_k_plus_1_batch.clone()
                        batch_current_s_actual_for_idm = spacing_adjusted_t_plus_k_plus_1.clone()

                    # Check newly generated feature slice
                    if torch.isnan(new_feature_slice_batch).any() or torch.isinf(new_feature_slice_batch).any():
                        print(
                            f"Warning: new_feature_slice at E:{epoch + 1}, B:{batch_idx}, K:{k_step} contains NaN/Inf. Skipping this batch.")
                        skip_batch_update = True
                        break

                    # Update IDM input sequence: remove oldest, add newest
                    batch_current_idm_input_torch = torch.cat(
                        [batch_current_idm_input_torch[:, 1:, :], new_feature_slice_batch.unsqueeze(1)], dim=1)

            if skip_batch_update: optimizer.zero_grad(); continue  # If skipped due to NaN, zero gradients and skip batch

            # Calculate loss for current batch
            batch_predicted_multi_step_speeds = torch.cat(all_predicted_follower_speeds_batch_list, dim=1)
            if torch.isnan(batch_predicted_multi_step_speeds).any() or torch.isinf(
                    batch_predicted_multi_step_speeds).any():
                print(
                    f"Warning: Final predicted speed sequence at E:{epoch + 1}, B:{batch_idx} contains NaN/Inf. Skipping this batch.")
                optimizer.zero_grad()
                continue

            squared_errors = criterion_mse_elementwise(batch_predicted_multi_step_speeds,
                                                       batch_true_follower_speeds_K_steps_for_loss)
            loss = (squared_errors * loss_weights.unsqueeze(0)).sum(dim=1).mean()  # Weighted average loss

            # Backpropagation and parameter update
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Final loss at E:{epoch + 1}, B:{batch_idx} is NaN/Inf. Skipping parameter update.")
                optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                total_loss_epoch += loss.item()
                num_valid_batches += 1

        avg_loss_epoch = total_loss_epoch / num_valid_batches if num_valid_batches > 0 else float('nan')
        print(
            f"[Transformer-IDM Multi-step Training] Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss_epoch:.4f} (based on {num_valid_batches}/{len(train_loader)} valid batches)")
        if np.isnan(avg_loss_epoch) and epoch > 0: print("Warning: Average loss is NaN, training stopped early."); break
    return model


def evaluate_generic_model(model, test_loader, pred_steps=5, model_name="Generic Model", device_eval=None):
    """
    General evaluation function for models like the Leader LNN.
    """
    model.eval()  # Set model to evaluation mode
    all_predicted, all_true = [], []

    # Check if test loader is valid
    if not test_loader or not hasattr(test_loader, 'dataset') or len(test_loader.dataset) == 0:
        print(f"{model_name} Evaluation: Test dataset is empty. Skipping evaluation.")
        return

    with torch.no_grad():  # Do not calculate gradients
        for batch_data, batch_target_speed in test_loader:
            batch_data, batch_target_speed = batch_data.to(device_eval), batch_target_speed.to(device_eval)
            predicted_speed = model.predict_speed(batch_data) if hasattr(model, 'predict_speed') else model(batch_data)
            all_predicted.append(predicted_speed.cpu())  # Move to CPU for storage
            all_true.append(batch_target_speed.cpu())  # Move to CPU for storage

    if not all_predicted: print(f"{model_name} Evaluation: No predictions made."); return
    all_predicted_cat = torch.cat(all_predicted, dim=0).numpy()
    all_true_cat = torch.cat(all_true, dim=0).numpy()
    if all_true_cat.shape[0] == 0: print(f"{model_name} Evaluation: True data is empty."); return

    # Calculate evaluation metrics
    mse_val = np.mean((all_predicted_cat - all_true_cat) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted_cat - all_true_cat))
    # Step-wise metrics
    mse_per_step = np.mean((all_predicted_cat - all_true_cat) ** 2, axis=0)
    rmse_per_step = np.sqrt(mse_per_step)
    mae_per_step = np.mean(np.abs(all_predicted_cat - all_true_cat), axis=0)

    print(f"\n{model_name} Evaluation (Overall {pred_steps} steps):")
    print(
        f"  Mean Squared Error (MSE): {mse_val:.4f}, Root Mean Squared Error (RMSE): {rmse_val:.4f}, Mean Absolute Error (MAE): {mae_val:.4f}")
    # Ensure index does not go out of bounds when printing step-wise metrics
    for i in range(min(pred_steps, rmse_per_step.shape[0])):
        print(f"  Step {i + 1} Prediction: RMSE: {rmse_per_step[i]:.4f}, MAE: {mae_per_step[i]:.4f}")

    # Plotting: Concatenate predictions and true trajectories for a subset of samples
    num_plot_samples = min(30, all_true_cat.shape[0])  # Plot at most 30 full sequences
    # Calculate sampling interval to ensure uniform selection of samples for plotting
    plot_step_interval = max(1, all_true_cat.shape[0] // num_plot_samples if num_plot_samples > 0 else 1)
    true_concat_plot, pred_concat_plot = [], []

    # From all test samples, sample at intervals, flatten K-step predictions/true values for each sampled sample, and append to plotting lists
    for i in range(0, all_true_cat.shape[0], plot_step_interval):
        if len(true_concat_plot) / pred_steps >= num_plot_samples: break  # If reached max plotting samples
        true_concat_plot.extend(all_true_cat[i])  # Extend true values list
        pred_concat_plot.extend(all_predicted_cat[i])  # Extend predicted values list

    if true_concat_plot:  # If there is data to plot
        plt.figure(figsize=(12, 6))
        plt.plot(np.array(true_concat_plot), linestyle='--', marker='o', markersize=3, label=f'True {model_name} Speed')
        plt.plot(np.array(pred_concat_plot), linestyle='-', marker='x', markersize=3,
                 label=f'Predicted {model_name} Speed')
        plt.title(f'{model_name} Multi-step Speed Prediction (Concatenated Samples)')
        plt.xlabel('Time Step in Prediction Horizon (Concatenated)')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.grid(True)
    else:
        print(f"{model_name} test set has insufficient samples for plotting.")


# Helper function to get multi-step predictions for Transformer-IDM model during evaluation
def get_idm_multistep_predictions(
        idm_model, leader_model, initial_idm_input_seq_batch, raw_data_slice_batch,
        pred_steps_K, dt, hist_len, device_compute
):
    """
    Performs multi-step prediction using the Transformer-IDM model for evaluation.
    Args:
        idm_model (HybridIDMModel): Pre-trained Transformer-IDM model.
        leader_model (LiquidNeuralNetworkMultiStep): Pre-trained Leader LNN model.
        initial_idm_input_seq_batch (torch.Tensor): Initial IDM input history for the current batch (batch, hist_len, idm_input_dim).
        raw_data_slice_batch (torch.Tensor): Raw data slice for the current batch, used to get initial positions and leader history.
        pred_steps_K (int): Number of prediction steps.
        dt (float): Time step.
        hist_len (int): Historical sequence length.
        device_compute (torch.device): Computation device.
    Returns:
        torch.Tensor: Transformer-IDM predicted future K-step ego speeds (batch, pred_steps_K).
    """
    idm_model.eval()  # Set to evaluation mode
    leader_model.eval()  # Set to evaluation mode

    # Precompute leader trajectories for the current batch
    (_, initial_f_pos_batch, initial_s_safe_batch,
     pred_l_speeds_K_batch, pred_l_pos_K_batch, d1_offset_batch) = \
        precompute_leader_trajectories_for_idm_training(
            leader_model, raw_data_slice_batch.to(device_compute),
            pred_steps_K, dt, device_compute, hist_len
        )
    initial_idm_input_seq_batch = initial_idm_input_seq_batch.to(device_compute)
    initial_f_pos_batch = initial_f_pos_batch.to(device_compute)
    initial_s_safe_batch = initial_s_safe_batch.to(device_compute)
    pred_l_speeds_K_batch = pred_l_speeds_K_batch.to(device_compute)
    pred_l_pos_K_batch = pred_l_pos_K_batch.to(device_compute)
    d1_offset_batch = d1_offset_batch.to(device_compute)

    # Initialize state variables for recurrent prediction
    batch_current_idm_input_torch = initial_idm_input_seq_batch.clone()
    batch_current_follower_speed_pred = batch_current_idm_input_torch[:, -1, 0].clone()
    batch_current_follower_pos = initial_f_pos_batch.clone()
    batch_current_s_actual_for_idm = initial_s_safe_batch.clone()

    all_predicted_follower_speeds_batch_list = []
    current_dt_val = dt if dt > 1e-6 else 1e-6

    with torch.no_grad():
        for k_step in range(pred_steps_K):
            if torch.isnan(batch_current_idm_input_torch).any() or torch.isnan(batch_current_s_actual_for_idm).any():
                v_follower_pred = batch_current_follower_speed_pred.clone()
                if k_step == 0 and torch.isnan(v_follower_pred).any():
                    v_follower_pred = torch.zeros_like(v_follower_pred)
                print(
                    f"Warning: IDM input in get_idm_multistep_predictions contains NaN (step {k_step}). Using fallback speed.")
            else:
                v_follower_pred_unsqueeze, _ = idm_model.predict_speed(
                    batch_current_idm_input_torch, batch_current_s_actual_for_idm
                )
                v_follower_pred = v_follower_pred_unsqueeze.squeeze(1)

            if torch.isnan(v_follower_pred).any() or torch.isinf(v_follower_pred).any():
                nan_inf_mask = torch.isnan(v_follower_pred) | torch.isinf(v_follower_pred)
                v_follower_pred[nan_inf_mask] = batch_current_follower_speed_pred[nan_inf_mask]
                if torch.isnan(v_follower_pred).any():
                    v_follower_pred[torch.isnan(v_follower_pred)] = 0.0
                print(
                    f"Warning: IDM predicted speed in get_idm_multistep_predictions contains NaN/Inf (step {k_step}). Using fallback speed.")

            all_predicted_follower_speeds_batch_list.append(v_follower_pred.unsqueeze(1))

            if k_step < pred_steps_K - 1:
                v_leader_next = pred_l_speeds_K_batch[:, k_step]
                pos_leader_next = pred_l_pos_K_batch[:, k_step]
                a_follower_next = (v_follower_pred - batch_current_follower_speed_pred) / current_dt_val
                a_follower_next = torch.clamp(a_follower_next, -10.0, 10.0)
                disp_follower_batch = batch_current_follower_speed_pred * current_dt_val + \
                                      0.5 * a_follower_next * current_dt_val ** 2
                pos_follower_next = batch_current_follower_pos + disp_follower_batch
                spacing_raw_next = pos_leader_next - pos_follower_next
                spacing_adjusted_next = spacing_raw_next - d1_offset_batch
                spacing_adjusted_next = torch.clamp(spacing_adjusted_next, min=0.1)
                delta_v_next = v_leader_next - v_follower_pred
                new_feature_slice = torch.stack([
                    v_follower_pred, spacing_adjusted_next, delta_v_next,
                    a_follower_next, v_leader_next
                ], dim=1)

                if torch.isnan(new_feature_slice).any():
                    print(f"Warning: new_feature_slice in get_idm_multistep_predictions contains NaN (step {k_step}).")
                else:
                    batch_current_idm_input_torch = torch.cat(
                        [batch_current_idm_input_torch[:, 1:, :], new_feature_slice.unsqueeze(1)], dim=1
                    )

                batch_current_follower_speed_pred = v_follower_pred.clone()
                batch_current_follower_pos = pos_follower_next.clone()
                batch_current_s_actual_for_idm = spacing_adjusted_next.clone()

    return torch.cat(all_predicted_follower_speeds_batch_list, dim=1)


def evaluate_final_lstm_idm_model(  # Function name kept for consistency, but it's now Transformer-IDM
        idm_model, leader_model_for_idm,
        raw_data_test_slice, label_data_test_slice,
        dt, pred_steps, hist_len_idm, device_comp,
        output_excel_filepath="transformer_idm_model_test_predictions.xlsx",  # Updated default filename
        excel_sheet_name="all_data"
):
    """
    Performs final Transformer-IDM multi-step prediction, evaluates its performance, and saves detailed predictions to Excel.
    """
    idm_model.eval()
    leader_model_for_idm.eval()

    N_test = raw_data_test_slice.shape[0]
    if N_test == 0: print(
        "Transformer-IDM Evaluation: Test data is empty. Skipping evaluation and saving."); return None, None, None, None, None, None

    # Get Transformer-IDM model's predictions
    idm_input_hist_test = raw_data_test_slice[:, -hist_len_idm:, [0, 1, 2, 3, 5]].clone() * 0.3048
    with torch.no_grad():
        y_transformer_idm_pred_speeds = get_idm_multistep_predictions(
            idm_model, leader_model_for_idm, idm_input_hist_test.to(device_comp), raw_data_test_slice,
            pred_steps, dt, hist_len_idm, device_comp
        )
    y_transformer_idm_pred_speeds_np = y_transformer_idm_pred_speeds.cpu().numpy()  # (N_test, pred_steps)

    # Load true ego future K-step speeds for evaluation
    true_f_speeds_np = label_data_test_slice[:, :pred_steps, 0].clone().cpu().numpy() * 0.3048

    # Infer ego future multi-step positions based on predicted ego speeds
    initial_ego_speeds_m_s = raw_data_test_slice[:, -1, 0].clone().cpu().numpy() * 0.3048
    initial_ego_positions_m = raw_data_test_slice[:, -1, 4].clone().cpu().numpy() * 0.3048
    pred_ego_positions_m_np = np.zeros_like(y_transformer_idm_pred_speeds_np)

    for i in range(N_test):
        current_speed_for_pos_calc = initial_ego_speeds_m_s[i]
        current_pos_for_pos_calc = initial_ego_positions_m[i]
        for k in range(pred_steps):
            predicted_speed_step_k = y_transformer_idm_pred_speeds_np[i, k]
            acceleration = (predicted_speed_step_k - current_speed_for_pos_calc) / dt
            displacement = current_speed_for_pos_calc * dt + 0.5 * acceleration * dt * dt
            new_position = current_pos_for_pos_calc + displacement
            pred_ego_positions_m_np[i, k] = new_position
            current_speed_for_pos_calc = predicted_speed_step_k
            current_pos_for_pos_calc = new_position

    # Load true ego future K-step positions for comparison
    true_ego_positions_m_np = label_data_test_slice[:, :pred_steps, 3].clone().cpu().numpy() * 0.3048

    # Calculate speed evaluation metrics
    true_f_speeds_mape_denom = np.where(np.abs(true_f_speeds_np) < 1e-5, 1e-5,
                                        true_f_speeds_np)  # Avoid division by zero
    mse_speed = np.mean((y_transformer_idm_pred_speeds_np - true_f_speeds_np) ** 2)
    rmse_speed = np.sqrt(mse_speed)
    mae_speed = np.mean(np.abs(y_transformer_idm_pred_speeds_np - true_f_speeds_np))
    mape_speed = np.mean(np.abs((y_transformer_idm_pred_speeds_np - true_f_speeds_np) / true_f_speeds_mape_denom)) * 100
    rmse_per_step_speed = np.sqrt(np.mean((y_transformer_idm_pred_speeds_np - true_f_speeds_np) ** 2, axis=0))
    mae_per_step_speed = np.mean(np.abs(y_transformer_idm_pred_speeds_np - true_f_speeds_np), axis=0)

    print(f"\nFinal Transformer-IDM Model Prediction Results ({pred_steps} steps) - Ego Speed:")
    print(f"  Mean Squared Error (MSE): {mse_speed:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse_speed:.4f} m/s")
    print(f"  Mean Absolute Error (MAE): {mae_speed:.4f} m/s")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape_speed:.2f}%")
    for i in range(pred_steps):
        print(
            f"  Step {i + 1} Speed Prediction (Transformer-IDM): RMSE: {rmse_per_step_speed[i]:.4f}, MAE: {mae_per_step_speed[i]:.4f}")

    # Calculate position evaluation metrics
    pos_rmse, pos_mape = np.nan, np.nan  # Initialize
    if true_ego_positions_m_np.shape == pred_ego_positions_m_np.shape and N_test > 0:
        true_f_pos_mape_denom = np.where(np.abs(true_ego_positions_m_np) < 1e-5, 1e-5, true_ego_positions_m_np)
        mse_pos = np.mean((pred_ego_positions_m_np - true_ego_positions_m_np) ** 2)
        pos_rmse = np.sqrt(mse_pos)
        mae_pos = np.mean(np.abs(pred_ego_positions_m_np - true_ego_positions_m_np))
        pos_mape = np.mean(np.abs((pred_ego_positions_m_np - true_ego_positions_m_np) / true_f_pos_mape_denom)) * 100
        rmse_per_step_pos = np.sqrt(np.mean((pred_ego_positions_m_np - true_ego_positions_m_np) ** 2, axis=0))
        mae_per_step_pos = np.mean(np.abs(pred_ego_positions_m_np - true_ego_positions_m_np), axis=0)

        print(f"\nFinal Transformer-IDM Model Inference Results ({pred_steps} steps) - Ego Position:")
        print(f"  Mean Squared Error (MSE): {mse_pos:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {pos_rmse:.4f} m")
        print(f"  Mean Absolute Error (MAE): {mae_pos:.4f} m")
        print(f"  Mean Absolute Percentage Error (MAPE): {pos_mape:.2f}%")
        for i in range(pred_steps):
            print(
                f"  Step {i + 1} Position Inference (Transformer-IDM): RMSE: {rmse_per_step_pos[i]:.4f}, MAE: {mae_per_step_pos[i]:.4f}")
    else:
        print(
            "\nWarning: Cannot calculate position evaluation metrics, true position and predicted position data mismatch or are empty.")

    # Save predicted speed, inferred position, and corresponding true values to Excel
    data_to_save_dict = {}
    for k_step_idx in range(pred_steps):
        data_to_save_dict[f"Pred_Ego_Speed_step{k_step_idx + 1}(m/s)"] = y_transformer_idm_pred_speeds_np[:, k_step_idx]
        data_to_save_dict[f"Pred_Ego_Pos_step{k_step_idx + 1}(m)"] = pred_ego_positions_m_np[:, k_step_idx]
        data_to_save_dict[f"True_Ego_Speed_step{k_step_idx + 1}(m/s)"] = true_f_speeds_np[:, k_step_idx]
        if true_ego_positions_m_np.shape[1] > k_step_idx:
            data_to_save_dict[f"True_Ego_Pos_step{k_step_idx + 1}(m)"] = true_ego_positions_m_np[:, k_step_idx]
        else:
            data_to_save_dict[f"True_Ego_Pos_step{k_step_idx + 1}(m)"] = np.full(N_test, np.nan)

    df_predictions = pd.DataFrame(data_to_save_dict)
    try:
        mode_for_writer = 'a' if os.path.exists(output_excel_filepath) else 'w'
        with pd.ExcelWriter(output_excel_filepath, engine="openpyxl", mode=mode_for_writer,
                            if_sheet_exists='replace') as writer:
            df_predictions.to_excel(writer, sheet_name=excel_sheet_name, index=False)
        print(
            f"\nTransformer-IDM model's detailed predictions (speed and inferred position) on test set saved to '{output_excel_filepath}' sheet '{excel_sheet_name}'.")
    except Exception as e:
        try:
            print(f"Failed to write to Excel in append mode ({e}), trying overwrite mode...")
            with pd.ExcelWriter(output_excel_filepath, engine="openpyxl", mode="w") as writer:
                df_predictions.to_excel(writer, sheet_name=excel_sheet_name, index=False)
            print(
                f"\nTransformer-IDM model's detailed predictions (speed and inferred position) on test set saved to '{output_excel_filepath}' sheet '{excel_sheet_name}' in overwrite mode.")
        except Exception as e2:
            print(f"\nError: Could not save predictions to Excel file '{output_excel_filepath}'. Error: {e2}")

    # Plotting: Ego speed
    num_plot_samples = min(30, N_test)
    plot_interval = max(1, N_test // num_plot_samples if num_plot_samples > 0 else 1)
    true_concat_plot_speed, pred_concat_plot_speed = [], []
    for i in range(0, N_test, plot_interval):
        if len(true_concat_plot_speed) / pred_steps >= num_plot_samples: break
        true_concat_plot_speed.extend(true_f_speeds_np[i, :])
        pred_concat_plot_speed.extend(y_transformer_idm_pred_speeds_np[i, :])

    if true_concat_plot_speed:
        plt.figure(figsize=(12, 6))
        plt.plot(np.array(true_concat_plot_speed), linestyle='--', marker='o', markersize=3,
                 label='True Ego Speed (Transformer-IDM Eval)')
        plt.plot(np.array(pred_concat_plot_speed), linestyle='-', marker='x', markersize=3,
                 label='Predicted Transformer-IDM Speed (Transformer-IDM Eval)')
        plt.title(f'Final Transformer-IDM Model Ego Speed Multi-step Prediction (Concatenated Samples)')
        plt.xlabel('Time Step in Prediction Horizon (Concatenated)')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.grid(True)
    else:
        print("Transformer-IDM evaluation has insufficient test data for plotting speed curve.")

    # Plotting: Ego position
    if N_test > 0 and pred_ego_positions_m_np.shape == true_ego_positions_m_np.shape:
        true_concat_plot_pos, pred_concat_plot_pos = [], []
        for i in range(0, N_test, plot_interval):
            if len(true_concat_plot_pos) / pred_steps >= num_plot_samples: break
            true_concat_plot_pos.extend(true_ego_positions_m_np[i, :])
            pred_concat_plot_pos.extend(pred_ego_positions_m_np[i, :])

        if true_concat_plot_pos:
            plt.figure(figsize=(12, 6))
            plt.plot(np.array(true_concat_plot_pos), linestyle='--', marker='o', markersize=3,
                     label='True Ego Position (Transformer-IDM Eval)')
            plt.plot(np.array(pred_concat_plot_pos), linestyle='-', marker='x', markersize=3,
                     label='Inferred Transformer-IDM Position (Transformer-IDM Eval)')
            plt.title(f'Final Transformer-IDM Model Ego Position Multi-step Inference (Concatenated Samples)')
            plt.xlabel('Time Step in Prediction Horizon (Concatenated)')
            plt.ylabel('Position (m)')
            plt.legend()
            plt.grid(True)
        else:
            print("Transformer-IDM evaluation has insufficient test data for plotting position curve.")

    return mse_speed, rmse_speed, mae_speed, mape_speed, pos_rmse, pos_mape


all_datasets_metrics_summary = []  # List to store dictionaries of evaluation metrics for all datasets


def store_dataset_metrics(dataset_name, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse, pos_mape):
    # Stores evaluation metrics for a single dataset
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


def save_all_metrics_to_csv(filepath="evaluation_summary_transformer_idm.csv"):  # Updated default filename
    # Saves summarized evaluation metrics for all datasets to a CSV file
    if not all_datasets_metrics_summary:
        print("No evaluation metrics to save.")
        return
    df_metrics = pd.DataFrame(all_datasets_metrics_summary)
    df_metrics.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"Summary of evaluation metrics for all datasets saved to {filepath}")


# =========================
# Main Function (Script Execution Entry)
# =========================
if __name__ == "__main__":
    torch.manual_seed(42)  # Set random seed for reproducibility
    np.random.seed(42)
    # torch.autograd.set_detect_anomaly(True) # Enable for debugging

    data_files = glob.glob(os.path.join(DATA_DIR, "*.mat"))
    print(f"Found .mat files: {data_files}")  # Print found file list
    if not data_files:
        print(f"No .mat files found in directory {DATA_DIR}.")
        exit()

    print(f"Found the following dataset files: {data_files}")

    # Iterate through each found data file
    for data_file_path in data_files:
        dataset_filename = os.path.basename(data_file_path)
        dataset_name_clean = dataset_filename.replace(".mat", "")

        print(f"\n==================== Starting processing dataset: {dataset_filename} ====================")

        data = sio.loadmat(data_file_path)
        # --- Define Hyperparameters and Configuration ---
        DT = 0.1  # Time step (seconds)
        HIST_LEN = 50  # Transformer-IDM and LNN input history sequence length

        IDM_INPUT_DIM = 5  # Transformer-IDM input feature dimension (v_f, s, dv, a_f, v_l)
        LEADER_LNN_INPUT_DIM = 2  # Leader LNN model input feature dimension (v_l, a_l)

        # Transformer Specific Parameters
        MODEL_DIM_TRANSFORMER = 64  # d_model in Transformer
        NHEAD_TRANSFORMER = 8  # Number of attention heads
        NUM_ENCODER_LAYERS_TRANSFORMER = 2  # Number of encoder layers
        DIM_FEEDFORWARD_TRANSFORMER = 128  # Dimension of the feedforward network

        # Model Hidden Dimensions (for LNN)
        HIDDEN_DIM_LNN_LEADER = 64

        # Number of layers (for LNN)
        NUM_LAYERS_LNN_LEADER = 1

        # Training Epochs
        LEADER_LNN_EPOCHS = 100
        IDM_MULTISTEP_EPOCHS = 100

        BATCH_SIZE = 32

        # --- Load Data ---
        if 'train_data' not in data or ('lable_data' not in data and 'label_data' not in data):
            print("Error: 'train_data' or 'lable_data'/'label_data' not found in .mat file. Please check data file.")
            continue  # Skip current file, process next

        label_key = 'lable_data' if 'lable_data' in data else 'label_data'
        raw_data_full = torch.tensor(data['train_data'], dtype=torch.float32)
        label_data_full = torch.tensor(data[label_key], dtype=torch.float32)

        # --- Data Subset Selection and Splitting ---
        total_samples_full = raw_data_full.shape[0]
        num_samples_to_use = int(total_samples_full * 1.0)  # Use 100% data

        raw_data_all = raw_data_full[:num_samples_to_use]
        label_data_all = label_data_full[:num_samples_to_use]
        print(f"Using {num_samples_to_use} samples for processing (Total samples: {total_samples_full})")
        if num_samples_to_use == 0: print("Error: No samples available for use."); continue  # Skip current file

        train_ratio_main = 0.8
        num_total_main = raw_data_all.shape[0]
        num_train_main = int(num_total_main * train_ratio_main)
        if num_train_main == 0 and num_total_main > 0: num_train_main = max(1,
                                                                            num_total_main - 1 if num_total_main > 1 else 1)
        if num_train_main == num_total_main and num_total_main > 1: num_train_main = num_total_main - 1
        num_test_main = num_total_main - num_train_main

        print(
            f"Main Split: Total samples {num_total_main}, Training samples {num_train_main}, Test samples {num_test_main}")
        if num_train_main == 0 or num_test_main == 0:
            print(
                f"Warning: Dataset {dataset_filename} has empty training or test set after splitting, skipping this dataset.")
            continue

        raw_train_data = raw_data_all[:num_train_main]
        label_train_data = label_data_all[:num_train_main]
        raw_test_data = raw_data_all[num_train_main:]
        label_test_data = label_data_all[num_train_main:]

        # --- 1. Train Leader LNN Model ---
        print("\n--- 1. Training Leader LNN Model ---")
        leader_lnn_input_hist_train = raw_train_data[:, -HIST_LEN:, [5, 6]].clone() * 0.3048
        leader_lnn_target_speeds_train = label_train_data[:, :, 4].clone() * 0.3048
        PRED_STEPS_K = leader_lnn_target_speeds_train.shape[1]  # Dynamically get prediction steps from label data

        leader_lnn_input_hist_test = raw_test_data[:, -HIST_LEN:, [5, 6]].clone() * 0.3048
        leader_lnn_target_speeds_test = label_test_data[:, :PRED_STEPS_K, 4].clone() * 0.3048

        leader_lnn_train_loader, leader_lnn_test_loader = None, None
        if leader_lnn_input_hist_train.shape[0] > 0:
            leader_lnn_train_dataset = torch.utils.data.TensorDataset(leader_lnn_input_hist_train,
                                                                      leader_lnn_target_speeds_train)
            leader_lnn_train_loader = torch.utils.data.DataLoader(leader_lnn_train_dataset, batch_size=BATCH_SIZE,
                                                                  shuffle=True)
        if leader_lnn_input_hist_test.shape[0] > 0:
            leader_lnn_test_dataset = torch.utils.data.TensorDataset(leader_lnn_input_hist_test,
                                                                     leader_lnn_target_speeds_test)
            leader_lnn_test_loader = torch.utils.data.DataLoader(leader_lnn_test_dataset, batch_size=BATCH_SIZE,
                                                                 shuffle=False)

        leader_model = LiquidNeuralNetworkMultiStep(
            LEADER_LNN_INPUT_DIM, HIDDEN_DIM_LNN_LEADER, PRED_STEPS_K, NUM_LAYERS_LNN_LEADER, HIST_LEN, DT
        ).to(device)
        initialize_weights(leader_model)
        optimizer_lead = optim.Adam(leader_model.parameters(), lr=1e-3)
        criterion_mse = nn.MSELoss()

        if leader_lnn_train_loader:
            train_generic_model(leader_model, leader_lnn_train_loader, optimizer_lead, criterion_mse,
                                LEADER_LNN_EPOCHS, "Leader LNN", clip_value=1.0)
        if leader_lnn_test_loader:
            evaluate_generic_model(leader_model, leader_lnn_test_loader, PRED_STEPS_K, "Leader LNN",
                                   device_eval=device)

        # --- 2. Prepare and Train Transformer-IDM Model ---
        print("\n--- 2. Training Transformer-IDM Model ---")
        idm_multistep_train_loader = None
        if raw_train_data.shape[0] > 0:
            (initial_idm_seq_train, initial_f_pos_train, initial_s_safe_train,
             pred_l_speeds_K_train, pred_l_pos_K_train, d1_train) = \
                precompute_leader_trajectories_for_idm_training(
                    leader_model, raw_train_data, PRED_STEPS_K, DT, device, HIST_LEN
                )
            true_f_speeds_K_train_for_loss = label_train_data[:, :PRED_STEPS_K, 0].clone() * 0.3048
            true_v_f_K = label_train_data[:, :PRED_STEPS_K, 0].clone()
            true_s_K = label_train_data[:, :PRED_STEPS_K, 1].clone()
            true_a_f_K = label_train_data[:, :PRED_STEPS_K, 2].clone()
            true_v_l_K = label_train_data[:, :PRED_STEPS_K, 4].clone()
            true_dv_K = true_v_l_K - true_v_f_K
            true_f_all_features_K_train = torch.stack([true_v_f_K, true_s_K, true_dv_K, true_a_f_K], dim=2) * 0.3048
            true_f_pos_K_train = label_train_data[:, :PRED_STEPS_K, 3].clone() * 0.3048

            if initial_idm_seq_train.shape[0] > 0:
                idm_multistep_train_dataset = torch.utils.data.TensorDataset(
                    initial_idm_seq_train, true_f_speeds_K_train_for_loss.to(device),
                    initial_f_pos_train, initial_s_safe_train,
                    pred_l_speeds_K_train, pred_l_pos_K_train, d1_train,
                    true_f_all_features_K_train.to(device), true_f_pos_K_train.to(device)
                )
                idm_multistep_train_loader = torch.utils.data.DataLoader(idm_multistep_train_dataset,
                                                                         batch_size=BATCH_SIZE, shuffle=True)
                print(f"Transformer-IDM: Training data samples {initial_idm_seq_train.shape[0]}")
            else:
                print("Transformer-IDM: Precomputed training data is empty.")

        idm_model = HybridIDMModel(IDM_INPUT_DIM, MODEL_DIM_TRANSFORMER, NHEAD_TRANSFORMER,
                                   NUM_ENCODER_LAYERS_TRANSFORMER, DIM_FEEDFORWARD_TRANSFORMER, DT).to(device)
        initialize_weights(idm_model)
        optimizer_idm = optim.Adam(idm_model.parameters(), lr=2e-4, weight_decay=1e-5)
        if idm_multistep_train_loader:
            train_idm_model_multistep(
                idm_model, idm_multistep_train_loader, optimizer_idm,
                IDM_MULTISTEP_EPOCHS, PRED_STEPS_K, DT, alpha_decay=0.05,
                teacher_forcing_initial_ratio=1.0, min_teacher_forcing_ratio=0.0,
                teacher_forcing_decay_epochs_ratio=0.75, clip_value=1.0
            )
        else:
            print("Transformer-IDM: Skipping training due to empty training data loader.")

        # --- 3. Final Transformer-IDM Model Evaluation and Results Saving ---
        print("\n--- 3. Final Transformer-IDM Model Evaluation and Results Saving ---")

        output_excel_filename = os.path.join(RESULTS_DIR, f"transformer_idm_predictions_{dataset_name_clean}.xlsx")
        dataset_basename_for_sheet = dataset_name_clean  # Use cleaned dataset name as sheet name

        if raw_test_data.shape[0] > 0 and label_test_data.shape[0] > 0:
            speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse, pos_mape = evaluate_final_lstm_idm_model(
                idm_model, leader_model,  # Pass trained models
                raw_test_data, label_test_data,  # Test data
                DT, PRED_STEPS_K, HIST_LEN, device,  # Relevant parameters
                output_excel_filepath=output_excel_filename,
                excel_sheet_name=dataset_basename_for_sheet
            )
            if speed_mse is not None:  # Check if valid metrics were returned
                store_dataset_metrics(dataset_name_clean, speed_mse, speed_rmse, speed_mae, speed_mape, pos_rmse,
                                      pos_mape)
        else:
            print(
                f"Dataset {dataset_name_clean} has insufficient test data for final Transformer-IDM model evaluation and saving.")

        print(f"\n==================== Dataset: {dataset_filename} processing complete ====================")

    # After all datasets are processed, save the summarized evaluation metrics
    summary_metrics_csv_path = os.path.join(RESULTS_DIR, "evaluation_summary_all_datasets_transformer_idm.csv")
    save_all_metrics_to_csv(summary_metrics_csv_path)

    # plt.show() # Uncomment to show all plots at once after script completion
    print("\n--- All processes completed ---")

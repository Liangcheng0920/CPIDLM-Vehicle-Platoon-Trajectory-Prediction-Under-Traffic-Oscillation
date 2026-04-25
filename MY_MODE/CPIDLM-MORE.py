import math
import os  # Import the os module for OS interactions, e.g., file path operations
import torch  # Import the PyTorch library for deep learning
import torch.nn as nn  # Import the PyTorch neural network module
import torch.optim as optim  # Import the PyTorch optimizer module
import matplotlib.pyplot as plt  # Import the matplotlib library for plotting
import scipy.io as sio  # Import scipy.io for loading .mat data files (MATLAB format)
import pandas as pd  # Import pandas for data processing and analysis
import numpy as np  # Import numpy for numerical computations

# Set the environment variable KMP_DUPLICATE_LIB_OK to TRUE.
# This is usually to resolve potential conflicts where the Intel MKL library
# (often used to accelerate math operations) might conflict with PyTorch's built-in libraries,
# allowing the loading of duplicate dynamic link libraries.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Automatically select the computing device: use cuda if a CUDA GPU is available, otherwise use CPU.
# The device object will be used later to move tensors and models to the selected device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device in use: {device}") # Print the currently used device

# =========================
# Data Inspection and Initialization Functions
# =========================
def check_data(data, name="data"):
    """
    Check if the input data contains NaN (Not a Number) or Inf (Infinity) values.
    These values typically indicate calculation errors or data issues.

    Args:
        data (torch.Tensor): The PyTorch tensor to be checked.
        name (str): The name of the data, used to distinguish different data sources in print statements.
    """
    print(f"Checking if {name} contains NaN or Inf values...")
    has_nan = torch.isnan(data).any().item()  # Check for any NaN values in the tensor
    has_inf = torch.isinf(data).any().item()  # Check for any Inf values in the tensor
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    if has_nan or has_inf:
        print(f"Warning: {name} contains NaN or Inf values!")


def initialize_weights(model):
    """
    Apply Xavier uniform initialization to the neural network model weights, and initialize biases to 0.
    Good weight initialization helps with model training and convergence. Xavier initialization is
    commonly used for networks with tanh or sigmoid activation functions.

    Args:
        model (nn.Module): The PyTorch model whose weights need to be initialized.
    """
    for name, param in model.named_parameters():  # Iterate over all named parameters of the model (including weights and biases)
        if "beta" in name:
            nn.init.constant_(param, 0.05)
        elif "weight" in name:  # If the parameter name contains "weight"
            if param.data.dim() > 1:  # Check the dimensions of the parameter. Weights are usually multi-dimensional
                nn.init.xavier_uniform_(param)  # Initialize weights using a Xavier uniform distribution
        elif "bias" in name:  # If the parameter name contains "bias"
            nn.init.constant_(param, 0)  # Initialize the bias term to 0


# =========================
# 1. SAM + CausalGAT-LSTM-IDM Model
# =========================
class CNormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(CNormalizedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        norm_weight = self.weight / self.weight.pow(2).sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)
        return nn.functional.linear(x, norm_weight, self.bias)


class SAMDiscriminator(nn.Module):
    def __init__(self, sizes, **kwargs):
        super(SAMDiscriminator, self).__init__()
        activation_function = kwargs.get("activation_function", nn.ReLU)
        activation_argument = kwargs.get("activation_argument", None)
        batch_norm = kwargs.get("batch_norm", False)
        dropout = kwargs.get("dropout", 0.0)
        layers = []
        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, j))
            if batch_norm:
                layers.append(nn.BatchNorm1d(j))
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SAMBlock(nn.Module):
    def __init__(self, sizes, zero_components=None, **kwargs):
        super(SAMBlock, self).__init__()
        zero_components = zero_components or []
        activation_function = kwargs.get("activation_function", nn.Tanh)
        activation_argument = kwargs.get("activation_argument", None)
        batch_norm = kwargs.get("batch_norm", False)
        layers = []
        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(CNormalizedLinear(i, j))
            if batch_norm:
                layers.append(nn.BatchNorm1d(j))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.layers = nn.Sequential(*layers)
        self.register_buffer("_filter", torch.ones(1, sizes[0]))
        for idx in zero_components:
            self._filter[:, idx] = 0.0
        self.fs_filter = nn.Parameter(self._filter.clone())

    def forward(self, x):
        filtered_x = x * (self._filter * self.fs_filter).expand_as(x)
        return self.layers(filtered_x)


class SAMGenerators(nn.Module):
    def __init__(self, data_shape, zero_components, nh=200, batch_size=-1, **kwargs):
        super(SAMGenerators, self).__init__()
        rows, self.cols = data_shape
        if batch_size == -1:
            batch_size = rows
        self.noise = [torch.randn(batch_size, 1) for _ in range(self.cols)]
        self.blocks = nn.ModuleList([
            SAMBlock([self.cols + 1, nh, 1], zero_components[i], **kwargs)
            for i in range(self.cols)
        ])

    def forward(self, x):
        batch_size = x.size(0)
        self.noise = [torch.randn(batch_size, 1, device=x.device) for _ in range(self.cols)]
        return [self.blocks[i](torch.cat([x, self.noise[i]], dim=1)) for i in range(self.cols)]


def run_sam(df_data, skeleton=None, **kwargs):
    del skeleton
    train_epochs = kwargs.get("train_epochs", 10)
    test_epochs = kwargs.get("test_epochs", 10)
    batch_size = kwargs.get("batch_size", -1)
    lr_gen = kwargs.get("lr_gen", 0.1)
    lr_disc = kwargs.get("lr_disc", lr_gen)
    verbose = kwargs.get("verbose", True)
    regul_param = kwargs.get("regul_param", 0.1)
    dnh = kwargs.get("dnh", 100)
    nh = kwargs.get("nh", 100)

    if hasattr(df_data, "columns"):
        data_np = df_data.values.astype("float32")
    else:
        data_np = np.asarray(df_data, dtype="float32")

    data_tensor = torch.from_numpy(data_np)
    rows, cols = data_tensor.size()
    if batch_size == -1:
        batch_size = rows

    zero_components = [[i] for i in range(cols)]
    sam = SAMGenerators((rows, cols), zero_components, nh=nh, batch_size=batch_size, batch_norm=True)
    discriminator = SAMDiscriminator(
        [cols, dnh, dnh, 1],
        batch_norm=True,
        activation_function=nn.LeakyReLU,
        activation_argument=0.2,
    )

    if device.type == "cuda":
        sam = sam.to(device)
        discriminator = discriminator.to(device)
        data_tensor = data_tensor.to(device)

    criterion = nn.BCEWithLogitsLoss()
    g_optimizer = optim.Adam(sam.parameters(), lr=lr_gen)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr_disc)
    causal_filters = torch.zeros(cols, cols, device=data_tensor.device)

    dataset = torch.utils.data.TensorDataset(data_tensor)
    drop_last = batch_size > 1 and rows % batch_size == 1
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    total_epochs = train_epochs + test_epochs
    for epoch in range(total_epochs):
        epoch_adv_loss = 0.0
        epoch_gen_loss = 0.0
        batch_count = 0
        for (batch,) in data_loader:
            batch_count += 1
            batch_vectors = [batch[:, i:i + 1] for i in range(cols)]
            current_batch_size = batch.size(0)
            true_variable = torch.ones(current_batch_size, 1, device=data_tensor.device)
            false_variable = torch.zeros(current_batch_size, 1, device=data_tensor.device)

            d_optimizer.zero_grad()
            generated_variables = sam(batch)
            disc_losses = []
            for i in range(cols):
                generator_output = torch.cat(
                    batch_vectors[:i] + [generated_variables[i]] + batch_vectors[i + 1:],
                    dim=1,
                )
                disc_output_detached = discriminator(generator_output.detach())
                disc_losses.append(criterion(disc_output_detached, false_variable))
            true_output = discriminator(batch)
            disc_loss_real = criterion(true_output, true_variable)
            adv_loss = (sum(disc_losses) / cols) + disc_loss_real
            adv_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            generated_variables = sam(batch)
            gen_losses = []
            for i in range(cols):
                generator_output = torch.cat(
                    batch_vectors[:i] + [generated_variables[i]] + batch_vectors[i + 1:],
                    dim=1,
                )
                gen_losses.append(criterion(discriminator(generator_output), true_variable))
            gen_loss = sum(gen_losses)
            filters = torch.stack([abs(block.fs_filter[0, :-1]) for block in sam.blocks], dim=1)
            loss = gen_loss + regul_param * filters.sum()
            loss.backward()
            if epoch >= train_epochs:
                causal_filters += filters.detach()
            g_optimizer.step()

            epoch_adv_loss += adv_loss.item()
            epoch_gen_loss += gen_loss.item()

        if verbose and batch_count > 0 and (epoch + 1) % 50 == 0:
            print(
                f"[SAM Epoch {epoch + 1}/{total_epochs}] "
                f"Discriminator Loss: {epoch_adv_loss / batch_count:.4f}, "
                f"Generator Loss: {epoch_gen_loss / batch_count:.4f}"
            )

    causal_filters = causal_filters / max(test_epochs, 1)
    return causal_filters.cpu().numpy()


class SAM(object):
    def __init__(
            self,
            lr=0.1,
            dlr=0.1,
            l1=0.1,
            nh=200,
            dnh=200,
            train_epochs=100,
            test_epochs=100,
            batchsize=-1,
    ):
        self.lr = lr
        self.dlr = dlr
        self.l1 = l1
        self.nh = nh
        self.dnh = dnh
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.batchsize = batchsize

    def predict(self, data, skeleton=None, nruns=1, verbose=True):
        results = []
        for run_idx in range(nruns):
            results.append(
                run_sam(
                    data,
                    skeleton=skeleton,
                    lr_gen=self.lr,
                    lr_disc=self.dlr,
                    regul_param=self.l1,
                    nh=self.nh,
                    dnh=self.dnh,
                    train_epochs=self.train_epochs,
                    test_epochs=self.test_epochs,
                    batch_size=self.batchsize,
                    verbose=verbose and nruns == 1,
                )
            )
        W = results[0]
        for result in results[1:]:
            W += result
        return W / nruns


class CausalGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(CausalGATLayer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)
        batch_size, num_nodes, _ = Wh.size()
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)

        e_base = self.leakyrelu(torch.matmul(torch.cat([Wh_i, Wh_j], dim=-1), self.a).squeeze(-1))
        e_guided = e_base + self.beta * adj

        adj_with_self_loop = adj + torch.eye(num_nodes, device=adj.device)
        zero_vec = -9e15 * torch.ones_like(e_guided)
        attention = torch.where(adj_with_self_loop > 0, e_guided, zero_vec)
        attention = torch.softmax(attention, dim=-1)
        attention = nn.functional.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return nn.functional.elu(h_prime) if self.concat else h_prime


class GAT(nn.Module):
    def __init__(self, gat_hidden_dim, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.gat1 = CausalGATLayer(in_features=1, out_features=gat_hidden_dim, dropout=dropout, alpha=alpha, concat=True)

    def forward(self, x, adj):
        return self.gat1(x, adj)


def build_sam_causal_matrix(idm_history_sequences, sam_train_epochs=10, sam_test_epochs=10, sam_batch_size=64):
    num_samples = idm_history_sequences.shape[0]
    num_features = idm_history_sequences.shape[-1]
    fallback = torch.zeros(num_features, num_features, dtype=torch.float32)
    if num_samples < 2:
        print("Insufficient SAM input samples, falling back to a zero causal matrix.")
        return fallback

    sam_input = idm_history_sequences[:, -1, :].detach().cpu().numpy()
    try:
        sam_model = SAM(
            lr=0.05,
            dlr=0.05,
            l1=0.1,
            nh=100,
            dnh=100,
            train_epochs=sam_train_epochs,
            test_epochs=sam_test_epochs,
            batchsize=min(sam_batch_size, num_samples),
        )
        causal_matrix_np = sam_model.predict(sam_input, nruns=1, verbose=False)
        causal_matrix = torch.tensor(causal_matrix_np, dtype=torch.float32)
    except Exception as exc:
        print(f"SAM causal discovery failed, falling back to a zero causal matrix: {exc}")
        return fallback

    min_val, max_val = torch.min(causal_matrix), torch.max(causal_matrix)
    if max_val > min_val:
        return (causal_matrix - min_val) / (max_val - min_val)
    return causal_matrix


class SAMCausalGATLSTMIDMModel(nn.Module):
    def __init__(self, input_dim, gat_hidden_dim, hidden_dim, num_layers=2, dt=0.1, causal_matrix=None):
        """
        SAM-CausalGAT-LSTM-IDM Model.
        Args:
            input_dim (int): Input feature dimension.
            gat_hidden_dim (int): GAT hidden dimension.
            hidden_dim (int): LSTM hidden layer dimension.
            num_layers (int): Number of LSTM layers.
            dt (float): Simulation time step (unit: seconds).
            causal_matrix (torch.Tensor): Causal prior matrix learned by SAM, shape (input_dim, input_dim).
        """
        super(SAMCausalGATLSTMIDMModel, self).__init__()
        self.gat = GAT(gat_hidden_dim)
        self.lstm_input_dim = input_dim + input_dim * gat_hidden_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 6)  # Output the 6 parameters of the IDM
        self.softplus = nn.Softplus()  # Softplus ensures parameters are positive
        self.delta_t = dt
        if causal_matrix is None:
            self.register_buffer("causal_matrix", torch.zeros(input_dim, input_dim))
        else:
            self.register_buffer("causal_matrix", causal_matrix)

    def forward(self, x):
        """
        The forward propagation process of the model. First aggregates features using CausalGAT, then outputs IDM parameters via LSTM.
        Args:
            x (torch.Tensor): Input historical sequence data, shape (batch_size, seq_len, input_dim).
        Returns:
            torch.Tensor: IDM parameters after Softplus activation, shape (batch_size, 6).
        """
        batch_size, seq_len, num_features = x.size()
        x_gat = x.view(batch_size * seq_len, num_features, 1)
        gat_out = self.gat(x_gat, self.causal_matrix)
        gat_out = gat_out.view(batch_size, seq_len, -1)
        lstm_in = torch.cat([x, gat_out], dim=-1)
        out, _ = self.lstm(lstm_in)
        params_raw = self.fc(out[:, -1, :])
        # Apply Softplus to ensure physical plausibility of parameters (e.g., positive values)
        params_activated = self.softplus(params_raw)
        return params_activated

    def predict_speed(self, x, s_actual):
        """
        Predicts the following vehicle's speed at the next time step using the IDM (Intelligent Driver Model) car-following formula.
        Args:
            x (torch.Tensor): Historical input sequence data of the following vehicle (batch, seq_len, input_dim).
                              Features: [Ego vehicle speed, Actual spacing (history), Speed difference with leading vehicle, Ego vehicle acceleration, Leading vehicle speed]
                              Note: The spacing in the IDM formula is the current actual spacing s_actual.
            s_actual (torch.Tensor): The current actual observed spacing (batch,).
        Returns:
            torch.Tensor: Predicted next-step speed of the following vehicle (batch, 1).
            torch.Tensor: IDM model parameters (batch, 6), after network output and clamping.
        """
        params = self.forward(x)  # Get the IDM parameters predicted by the neural network
        v_n = x[:, -1, 0]  # Current ego vehicle speed (last time point in the sequence)
        delta_v_hist = x[:, -1, 2]  # Current speed difference (v_leader - v_follower)

        # Parse and clamp IDM parameters to a reasonable range to enhance model stability
        v_des_raw, T_raw, a_max_raw, b_safe_raw, delta_idm_raw, s0_raw = \
            params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]

        v_des = torch.clamp(v_des_raw, min=0.1, max=50.0)  # Desired speed (m/s)
        T = torch.clamp(T_raw, min=0.1, max=5.0)  # Safe time headway (s)
        a_max = torch.clamp(a_max_raw, min=0.1, max=5.0)  # Maximum acceleration (m/s^2)
        b_safe = torch.clamp(b_safe_raw, min=0.1, max=9.0)  # Comfortable deceleration (m/s^2)
        delta_idm = torch.clamp(delta_idm_raw, min=1.0, max=10.0)  # Acceleration exponent
        s0 = torch.clamp(s0_raw, min=0.0, max=10.0)  # Minimum standstill spacing (m)
        s_actual_clamped = torch.clamp(s_actual, min=0.5)  # Clamp actual spacing to prevent it from being too small

        # IDM formula calculations
        # Desired dynamic spacing s_star
        sqrt_ab_clamped = torch.clamp(torch.sqrt(a_max * b_safe), min=1e-6)  # Prevent square root result from being 0
        # Speed difference in the IDM interaction term (v_n - v_l), corresponds to -delta_v_hist
        interaction_term = (v_n * (-delta_v_hist)) / (2 * sqrt_ab_clamped + 1e-9)  # Add a tiny amount to prevent division by zero
        s_star = s0 + torch.clamp(v_n * T, min=0.0) + interaction_term  # v_n * T should be non-negative
        s_star = torch.clamp(s_star, min=s0)  # s_star is at least s0

        # Acceleration calculation
        v_n_clamped = torch.clamp(v_n, min=0.0)  # Current speed is non-negative
        speed_ratio = (v_n_clamped + 1e-6) / (v_des + 1e-6)  # Add a tiny amount to prevent division by zero
        term_speed_ratio = speed_ratio.pow(delta_idm)
        spacing_ratio = s_star / (s_actual_clamped + 1e-6)  # Add a tiny amount to prevent division by zero
        term_spacing_ratio = spacing_ratio.pow(2)

        accel_component = 1.0 - term_speed_ratio - term_spacing_ratio
        a_idm_val = a_max * accel_component  # Acceleration calculated by the IDM

        # Speed update: v(t+dt) = v(t) + a(t)*dt
        v_follow = v_n + a_idm_val * self.delta_t
        v_follow = torch.clamp(v_follow, min=0.0, max=60.0)  # Clamp predicted speed to a reasonable range (0 ~ 216km/h)

        # NaN/Inf check, used for debugging
        if torch.isnan(v_follow).any() or torch.isinf(v_follow).any():
            print("Warning: NaN/Inf output detected in SAMCausalGATLSTMIDMModel.predict_speed.")
            # Detailed parameter printing can be added here to help locate the source of the problem
        return v_follow.unsqueeze(1), params


# =========================
# 2. LNN Model (Liquid Neural Network) - Base class and implementation for leading vehicle prediction
# =========================
class LiquidCellMulti(nn.Module):  # Single LNN neuron cell
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        super(LiquidCellMulti, self).__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt  # Time step for ODE solving
        # Linear transformation layer, no bias (bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Hidden state to hidden state
        self.W_u = nn.Linear(input_dim, hidden_dim, bias=False)  # Input to hidden state
        self.bias = nn.Parameter(torch.zeros(hidden_dim))  # Learnable bias
        self.act = nn.Tanh()  # Activation function

    def forward(self, u, h):
        # h_dot = -h + act(W_h*h + W_u*u + bias)
        # h_new = h_old + dt * h_dot (Euler method)
        if h.shape[-1] != self.hidden_dim:  # Initialize hidden state (usually at the beginning of a sequence)
            h = torch.zeros(u.shape[0], self.hidden_dim, device=u.device)
        dh = -h + self.act(self.W_h(h) + self.W_u(u) + self.bias)
        return h + self.dt * dh


class LiquidNeuralNetworkMultiStep(nn.Module):  # Used for leading vehicle (Leader) prediction
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1, num_steps=50, dt=0.1):
        super(LiquidNeuralNetworkMultiStep, self).__init__()
        self.input_dim = input_dim
        self.cells = nn.ModuleList()  # Store LNN layers
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(LiquidCellMulti(current_input_dim, hidden_dim, dt=dt))
        self.fc = nn.Linear(hidden_dim, prediction_steps)  # Output layer
        self.num_steps = num_steps  # Internal simulation steps of LNN, usually equals to input sequence length
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch, seq, features = x.shape
        h_states = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        effective_seq_len = min(seq, self.num_steps)  # The actual sequence length processed by LNN

        for t in range(effective_seq_len):  # Iterate along the time sequence
            u_t_layer = x[:, t, :]  # Input at the current time step
            for i in range(self.num_layers):  # Iterate through each layer of the LNN
                input_signal_for_cell = h_states[i - 1] if i > 0 else u_t_layer  # First layer gets the raw signal, subsequent layers get previous hidden state
                h_states[i] = self.cells[i](input_signal_for_cell, h_states[i])
        return self.fc(h_states[-1])  # Use the final hidden state of the last LNN layer for prediction

    def predict_speed(self, x):  # Convenience method
        return self.forward(x)


# =========================
# 2.1 NEW: LNN Model - For direct multi-step speed prediction of the ego vehicle (LNN-Ego)
# =========================
class LiquidNeuralNetworkMultiStepEgo(nn.Module):  # Used for direct prediction of the ego/follower vehicle
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1, num_steps=50, dt=0.1):
        """
        LNN model for direct multi-step speed prediction of the ego vehicle.
        Args:
            input_dim (int): Feature dimension at each time step in the input sequence.
                             (Should be consistent with LSTM-IDM input features)
            hidden_dim (int): Hidden state dimension of the LNN cells.
            prediction_steps (int): Number of future time steps to predict (ego vehicle speed).
            num_layers (int): Number of LNN layers.
            num_steps (int): Number of steps for ODE simulation inside the LNN (usually equal to the input sequence length).
            dt (float): Time step for ODE solving inside the LNN cells.
        """
        super(LiquidNeuralNetworkMultiStepEgo, self).__init__()
        self.input_dim = input_dim
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(LiquidCellMulti(current_input_dim, hidden_dim, dt=dt))
        # Fully connected layer mapping the final hidden state of the last LNN layer to the predicted ego vehicle speed for multiple future time steps
        self.fc = nn.Linear(hidden_dim, prediction_steps)
        self.num_steps = num_steps  # Internal simulation steps (usually the input sequence length)
        self.num_layers = num_layers  # Number of LNN layers
        self.hidden_dim = hidden_dim  # LNN hidden dimension

    def forward(self, x):
        """
        Forward pass of the LNN-Ego model.
        Args:
            x (torch.Tensor): Input historical sequence data, dimension (batch, seq_len, input_dim).
                              Features are consistent with LSTM-IDM input: [Ego speed, Spacing, Speed difference, Ego acceleration, Leader speed]
        Returns:
            torch.Tensor: Predicted ego vehicle speeds for multiple future time steps, dimension (batch, prediction_steps).
        """
        batch, seq, features = x.shape  # Get the dimensions of the input tensor
        h_states = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        effective_seq_len = min(seq, self.num_steps)

        for t in range(effective_seq_len):
            u_t_layer = x[:, t, :]  # Input at the current time step (batch, input_dim)
            for i in range(self.num_layers):
                # For the first layer, the cell input is u_t_layer
                # For subsequent layers, the cell input is the hidden state of the previous layer h_states[i-1]
                input_signal_for_cell = h_states[i - 1] if i > 0 else u_t_layer
                h_states[i] = self.cells[i](input_signal_for_cell, h_states[i])
        return self.fc(h_states[-1])  # Output the predicted K-step ego vehicle speed

    def predict_speed(self, x):
        """ A convenience method to directly call forward for speed prediction. """
        return self.forward(x)


# =========================
# 2.2 NEW: Fusion LSTM Model (FusionLSTM) - Used to output the fusion gating value alpha
# =========================
class FusionLSTMModel(nn.Module):
    def __init__(self, fusion_input_dim, fusion_hidden_dim, fusion_output_steps, fusion_num_layers=1):
        """
        Initialization function for the Fusion LSTM model.
        Args:
            fusion_input_dim (int): Input feature dimension of the LSTM (e.g., ego historical speed + leader historical speed = 2).
            fusion_hidden_dim (int): Hidden layer dimension of the LSTM.
            fusion_output_steps (int): Number of steps for the output alpha gating values (should equal PRED_STEPS_K).
            fusion_num_layers (int): Number of LSTM layers.
        """
        super(FusionLSTMModel, self).__init__()
        self.lstm = nn.LSTM(fusion_input_dim, fusion_hidden_dim, fusion_num_layers, batch_first=True)
        # Fully connected layer mapping the LSTM output to K alpha values
        self.fc = nn.Linear(fusion_hidden_dim, fusion_output_steps)
        # Sigmoid activation function to ensure alpha values are between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_ego_hist_speed, x_leader_hist_speed):
        """
        Forward pass of the Fusion LSTM model.
        Args:
            x_ego_hist_speed (torch.Tensor): Ego historical speed sequence (batch, FUSION_HIST_LEN, 1).
            x_leader_hist_speed (torch.Tensor): Leader historical speed sequence (batch, FUSION_HIST_LEN, 1).
        Returns:
            torch.Tensor: Predicted alpha fusion gating values for multiple future time steps (batch, fusion_output_steps).
        """
        # Concatenate ego and leader historical speeds as input
        # fusion_input dimension: (batch, FUSION_HIST_LEN, 2)
        fusion_input = torch.cat((x_ego_hist_speed, x_leader_hist_speed), dim=2)

        lstm_out, _ = self.lstm(fusion_input)
        # Get the output of the LSTM at the last time step
        alpha_raw = self.fc(lstm_out[:, -1, :])
        # Get the alpha values through the Sigmoid function
        alpha = self.sigmoid(alpha_raw)  # Dimension (batch, PRED_STEPS_K)
        return alpha


# =========================
# Training Function Definitions
# =========================
def train_generic_model(model, loader, optimizer, criterion, epochs=30, model_name="Generic Model", clip_value=1.0,
                        is_fusion_model=False, idm_model_frozen=None, lnn_ego_model_frozen=None,
                        leader_model_frozen=None, raw_data_loader_for_fusion=None, FUSION_HIST_LEN=None,
                        PRED_STEPS_K_fusion=None, DT_fusion=None, HIST_LEN_idm_ego_fusion=None, device_fusion=None):
    """
    Generic training function that can be used to train the leader LNN, ego LNN-Ego, and Fusion LSTM models.
    Args:
        model (nn.Module): The model to be trained.
        loader (torch.utils.data.DataLoader): Training data loader.
            - For LNN/LNN-Ego: loader yields (x_batch, y_batch)
            - For FusionLSTM: loader yields (raw_data_batch, label_data_batch) to extract required inputs
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        epochs (int): Total number of training epochs.
        model_name (str): Model name, used for logging.
        clip_value (float): Gradient clipping threshold.
        is_fusion_model (bool): Flag indicating if the fusion model is being trained.
        idm_model_frozen (HybridIDMModel): (Fusion only) Pre-trained, frozen LSTM-IDM model.
        lnn_ego_model_frozen (LiquidNeuralNetworkMultiStepEgo): (Fusion only) Pre-trained, frozen LNN-Ego model.
        leader_model_frozen (LiquidNeuralNetworkMultiStep): (Fusion only) Pre-trained, frozen leader LNN model (used for internal prediction in LSTM-IDM).
        raw_data_loader_for_fusion (torch.utils.data.DataLoader): (Deprecated parameter, now directly obtained from loader)
        FUSION_HIST_LEN, PRED_STEPS_K_fusion, DT_fusion, HIST_LEN_idm_ego_fusion, device_fusion: (Fusion only) Parameters required for fusion model training.

    Returns:
        nn.Module: The trained model.
    """
    model.train()  # Set the model to training mode
    if is_fusion_model:  # If training the fusion model, ensure its dependent sub-models are in evaluation mode (frozen parameters)
        if idm_model_frozen: idm_model_frozen.eval()
        if lnn_ego_model_frozen: lnn_ego_model_frozen.eval()
        if leader_model_frozen: leader_model_frozen.eval()

    for ep in range(epochs):  # Iterate over the number of training epochs
        tot_loss = 0  # Total loss for the current epoch
        num_batches_processed = 0  # Number of batches processed in the current epoch

        # Batch processing logic for the fusion model is more complex
        if is_fusion_model:
            # The loader for the fusion model should yield (raw data batch, label data batch)
            for batch_idx, (raw_batch, label_batch) in enumerate(loader):
                raw_batch, label_batch = raw_batch.to(device_fusion), label_batch.to(device_fusion)
                optimizer.zero_grad()  # Zero the gradients

                # 1. Get y_lstm_idm_pred (predicted output of the LSTM-IDM model)
                #    Requires LSTM-IDM input history and leader trajectory (predicted via leader_model_frozen)
                #    LSTM-IDM input features: [Ego speed, Spacing, Speed diff, Ego accel, Leader speed]
                idm_input_hist_batch = raw_batch[:, -HIST_LEN_idm_ego_fusion:, [0, 1, 2, 3, 5]].clone() * 0.3048  # Unit conversion
                with torch.no_grad():  # Ensure no gradients are calculated for frozen models
                    y_lstm_idm_pred_batch = predict_multi_step_idm_for_fusion_training(
                        idm_model_frozen, leader_model_frozen,
                        idm_input_hist_batch, raw_batch,  # raw_batch contains the info needed to generate the leader trajectory
                        PRED_STEPS_K_fusion, DT_fusion, HIST_LEN_idm_ego_fusion, device_fusion
                    )  # Shape: (batch, PRED_STEPS_K_fusion)

                # 2. Get y_lnn_ego_pred (predicted output of the LNN-Ego model)
                #    LNN-Ego input is the same as LSTM-IDM
                lnn_ego_input_batch = raw_batch[:, -HIST_LEN_idm_ego_fusion:, [0, 1, 2, 3, 5]].clone() * 0.3048  # Unit conversion
                with torch.no_grad():
                    y_lnn_ego_pred_batch = lnn_ego_model_frozen.predict_speed(
                        lnn_ego_input_batch)  # Shape: (batch, PRED_STEPS_K_fusion)

                # 3. Prepare input for FusionLSTM
                #    Ego historical speed: (batch, FUSION_HIST_LEN, 1)
                ego_speed_hist_fusion = raw_batch[:, -FUSION_HIST_LEN:, 0].unsqueeze(-1).clone() * 0.3048
                #    Leader historical speed: (batch, FUSION_HIST_LEN, 1)
                leader_speed_hist_fusion = raw_batch[:, -FUSION_HIST_LEN:, 5].unsqueeze(-1).clone() * 0.3048

                # 4. Get the fusion gating value alpha through FusionLSTM
                #    model refers to the currently training fusion_model
                alpha_batch = model(ego_speed_hist_fusion, leader_speed_hist_fusion)  # Shape: (batch, PRED_STEPS_K_fusion)

                # 5. Calculate fusion prediction y_fusion
                #    Use .detach() to ensure predictions from frozen models do not participate in gradient calculation for fusion_model
                y_fusion_batch = alpha_batch * y_lstm_idm_pred_batch.detach() + \
                                 (1 - alpha_batch) * y_lnn_ego_pred_batch.detach()

                # 6. Calculate loss
                #    Ground truth ego future K-step speed, used for comparison with y_fusion_batch
                true_follower_speeds_K_batch = label_batch[:, :PRED_STEPS_K_fusion, 0].clone() * 0.3048  # Unit conversion
                loss = criterion(y_fusion_batch, true_follower_speeds_K_batch)

                # NaN/Inf loss check
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: {model_name} encountered NaN/Inf loss at epoch {ep + 1}, batch {batch_idx}. Skipping this batch.")
                    optimizer.zero_grad()  # Clear any NaN gradients that might have resulted from this
                    continue  # Skip backward pass and parameter update

                loss.backward()  # Backward pass
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Gradient clipping
                optimizer.step()  # Update fusion model parameters
                tot_loss += loss.item()
                num_batches_processed += 1
        else:  # Train LNN-Leader or LNN-Ego model
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device_fusion if device_fusion else x_batch.device), y_batch.to(device_fusion if device_fusion else y_batch.device)  # Move data to device
                optimizer.zero_grad()
                # For LNN models, predict_speed is equivalent to forward
                pred = model.predict_speed(x_batch) if hasattr(model, 'predict_speed') else model(x_batch)
                loss = criterion(pred, y_batch)  # Calculate loss

                # NaN/Inf loss check
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: {model_name} encountered NaN/Inf loss at epoch {ep + 1}. Skipping this batch.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                tot_loss += loss.item()
                num_batches_processed += 1

        # Calculate and print average loss
        avg_loss = tot_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
        print(f"[{model_name}] Epoch {ep + 1}/{epochs}, Average loss: {avg_loss:.4f}")
        # If average loss is NaN and it's not the first epoch, there might be a severe issue; stop early
        if np.isnan(avg_loss) and ep > 0:
            print(f"Warning: {model_name} average loss is NaN, training stopped early.")
            break
    return model


def precompute_leader_trajectories_for_idm_training(
        leader_model, raw_data_slice, pred_steps_K, dt, device, hist_len=50
):
    """
    Precompute leader trajectories (speed and position) for IDM model training.
    This avoids repeatedly predicting leader trajectories in every iteration of IDM training, improving efficiency.
    Args:
        leader_model (nn.Module): Pre-trained leader LNN model.
        raw_data_slice (torch.Tensor): Raw data slice (num_samples, seq_len, features).
        pred_steps_K (int): Number of future prediction steps.
        dt (float): Time step.
        device (torch.device): Computing device.
        hist_len (int): Leader LNN model input history length.
    Returns:
        tuple: Contains various tensors needed for IDM training.
    """
    leader_model.eval()  # Set leader model to evaluation mode
    num_samples = raw_data_slice.shape[0]

    # Handle empty inputs by returning correctly shaped but empty tensors
    if num_samples == 0:
        # Assume IDM_INPUT_DIM is known or obtainable elsewhere, using a placeholder value here
        # If IDM_INPUT_DIM is not defined, this line will error; ensure it's defined before calling this function
        # In the main program, IDM_INPUT_DIM = 5
        _IDM_INPUT_DIM_placeholder = 5  # Should be consistent with the global definition
        empty_tensor_k_steps = torch.empty(0, pred_steps_K, dtype=torch.float32, device=device)
        empty_tensor_idm_input = torch.empty(0, hist_len, _IDM_INPUT_DIM_placeholder, dtype=torch.float32,
                                             device=device)
        empty_tensor_scalar_batch = torch.empty(0, dtype=torch.float32, device=device)
        return empty_tensor_idm_input, empty_tensor_scalar_batch, empty_tensor_scalar_batch, \
            empty_tensor_k_steps, empty_tensor_k_steps, empty_tensor_scalar_batch

    # Prepare initial input sequence for the IDM model (features: v_f, s, dv, a_f, v_l)
    initial_idm_input_seqs = raw_data_slice[:, -hist_len:, [0, 1, 2, 3, 5]].clone() * 0.3048  # Unit conversion
    initial_follower_poses = raw_data_slice[:, -1, 4].clone() * 0.3048  # Ego initial position
    initial_leader_poses_val = raw_data_slice[:, -1, -1].clone() * 0.3048  # Leader initial position
    initial_s_safes = initial_idm_input_seqs[:, -1, 1].clone()  # Initial actual spacing (units already converted)
    # d1 is the initial leader_pos - follower_pos - s_safe, used for subsequent spacing correction
    batch_d1 = initial_leader_poses_val - initial_follower_poses - initial_s_safes

    # Prepare input for the leader LNN model (features: v_l, a_l)
    leader_hist_for_lnn = raw_data_slice[:, -hist_len:, [5, 6]].clone() * 0.3048  # Unit conversion

    pred_leader_speeds_K_list = []  # Store the predicted leader speed sequence for each sample
    pred_leader_pos_K_list = []  # Store the predicted leader position sequence for each sample
    current_dt = dt if dt > 1e-6 else 1e-6  # Ensure dt is valid

    with torch.no_grad():  # No gradient calculation
        # Batch predict leader future K-step speeds
        all_pred_l_speeds_k_steps_tensor = leader_model.predict_speed(leader_hist_for_lnn.to(device)).cpu()  # Move to CPU for processing

        for i in range(num_samples):  # Iterate through each sample
            pred_l_speeds_k_steps_tensor_i = all_pred_l_speeds_k_steps_tensor[i]  # Predicted speed sequence for the current sample
            pred_leader_speeds_K_list.append(pred_l_speeds_k_steps_tensor_i)

            # Iteratively calculate leader's future K-step positions based on predicted speeds
            current_l_pos = initial_leader_poses_val[i].item()  # Current leader position (t=0)
            prev_l_v = leader_hist_for_lnn[i, -1, 0].item()  # Leader historical end speed (v at t=0)
            l_pos_k_steps = []  # Store the future K-step positions for the current sample

            for k_idx in range(pred_steps_K):  # Iterate over future K steps
                vp = pred_l_speeds_k_steps_tensor_i[k_idx].item()  # Predicted v(t+k+1)
                a_leader = (vp - prev_l_v) / current_dt  # Average acceleration
                displacement_leader = prev_l_v * current_dt + 0.5 * a_leader * current_dt * current_dt  # Displacement
                next_l_pos = current_l_pos + displacement_leader  # New position
                l_pos_k_steps.append(next_l_pos)

                prev_l_v = vp  # Update speed for the next calculation step
                current_l_pos = next_l_pos  # Update position
            pred_leader_pos_K_list.append(torch.tensor(l_pos_k_steps, dtype=torch.float32))

    # Stack tensors from lists
    pred_leader_speeds_K = torch.stack(pred_leader_speeds_K_list) if num_samples > 0 else torch.empty(0, pred_steps_K,
                                                                                                      dtype=torch.float32)
    pred_leader_pos_K = torch.stack(pred_leader_pos_K_list) if num_samples > 0 else torch.empty(0, pred_steps_K,
                                                                                                dtype=torch.float32)

    # Return all computed tensors, ensuring they are on the correct device (especially those to be input to models)
    return initial_idm_input_seqs.to(device), initial_follower_poses.to(device), initial_s_safes.to(device), \
        pred_leader_speeds_K.to(device), pred_leader_pos_K.to(device), batch_d1.to(device)


def train_idm_model_multistep(
        model, train_loader, optimizer,
        num_epochs=30, pred_steps_K=5, dt=0.1, alpha_decay=0.0,  # alpha_decay controls loss weight
        teacher_forcing_initial_ratio=1.0,  # Initial Teacher Forcing ratio
        min_teacher_forcing_ratio=0.0,  # Minimum Teacher Forcing ratio
        teacher_forcing_decay_epochs_ratio=0.75,  # Proportion of epochs for TF ratio decay
        clip_value=1.0  # Gradient clipping value
):
    """ Train the SAM-CausalGAT-LSTM-IDM model using multi-step prediction and scheduled sampling (Teacher Forcing) """
    model.train()
    criterion_mse_elementwise = nn.MSELoss(reduction='none')  # Element-wise MSE, used for weighting
    # Loss weight: w_t = exp(-alpha_decay * t), further prediction steps have smaller weights (if alpha_decay > 0)
    loss_weights = torch.exp(-alpha_decay * torch.arange(pred_steps_K, device=device).float())
    decay_epochs = int(num_epochs * teacher_forcing_decay_epochs_ratio)  # Total number of epochs for TF decay
    current_dt = dt if dt > 1e-6 else 1e-6  # Ensure dt is valid

    for epoch in range(num_epochs):
        total_loss_epoch = 0
        num_valid_batches = 0  # Record the number of valid batches (without NaN)

        # Calculate the Teacher Forcing ratio for the current epoch (linear decay)
        current_teacher_forcing_ratio = teacher_forcing_initial_ratio - \
                                        (teacher_forcing_initial_ratio - min_teacher_forcing_ratio) * \
                                        (float(epoch) / decay_epochs if decay_epochs > 0 else 0)
        current_teacher_forcing_ratio = max(min_teacher_forcing_ratio, current_teacher_forcing_ratio)
        print(
            f"[SAM-CausalGAT-LSTM-IDM Multi-step Training] Epoch [{epoch + 1}/{num_epochs}], "
            f"Teacher Forcing Ratio: {current_teacher_forcing_ratio:.4f}"
        )

        # The training data loader provides precomputed batch data
        for batch_idx, (batch_initial_idm_input_seq,  # Initial IDM input history
                        batch_true_follower_speeds_K_steps_for_loss,  # Ground truth ego vehicle future K-step speed (for loss)
                        batch_initial_follower_pos,  # Ego initial position
                        batch_initial_s_safe,  # Initial actual spacing
                        batch_pred_leader_speeds_K_steps,  # Predicted leader vehicle future K-step speed
                        batch_pred_leader_pos_K_steps,  # Predicted leader vehicle future K-step position
                        batch_d1_offset,  # Spacing correction offset d1
                        batch_true_follower_all_features_K_steps,  # Ground truth ego future K-step all relevant features (for TF)
                        batch_true_follower_pos_K_steps  # Ground truth ego future K-step position (for TF)
                        ) in enumerate(train_loader):

            # Data is already moved to device by precompute function or DataLoader, but labels might need confirmation
            batch_true_follower_speeds_K_steps_for_loss = batch_true_follower_speeds_K_steps_for_loss.to(device)
            batch_true_follower_all_features_K_steps = batch_true_follower_all_features_K_steps.to(device)
            batch_true_follower_pos_K_steps = batch_true_follower_pos_K_steps.to(device)

            optimizer.zero_grad()
            # Initialize state variables needed for loop prediction (copy to avoid in-place modification)
            batch_current_idm_input_torch = batch_initial_idm_input_seq.clone()
            batch_current_follower_speed_pred = batch_current_idm_input_torch[:, -1, 0].clone()  # Current ego vehicle speed (prediction starting point)
            batch_current_follower_pos = batch_initial_follower_pos.clone()  # Current ego vehicle position
            batch_current_s_actual_for_idm = batch_initial_s_safe.clone()  # Current actual spacing input for IDM
            all_predicted_follower_speeds_batch_list = []  # Store K-step predictions
            skip_batch_update = False  # Flag whether to skip current batch due to NaN/Inf

            # Loop prediction over pred_steps_K future time steps
            for k_step in range(pred_steps_K):
                # Input check
                if torch.isnan(batch_current_idm_input_torch).any() or torch.isinf(
                        batch_current_idm_input_torch).any() or \
                        torch.isnan(batch_current_s_actual_for_idm).any() or torch.isinf(
                    batch_current_s_actual_for_idm).any():
                    print(f"Warning: IDM input contains NaN/Inf at E:{epoch + 1}, B:{batch_idx}, K:{k_step}. Skipping this batch.")
                    skip_batch_update = True;
                    break

                # IDM model predicts one step (v_follower_t+k+1)
                v_follower_t_plus_k_plus_1_pred_batch_unsqueeze, _ = model.predict_speed(
                    batch_current_idm_input_torch, batch_current_s_actual_for_idm)
                v_follower_t_plus_k_plus_1_pred_batch = v_follower_t_plus_k_plus_1_pred_batch_unsqueeze.squeeze(1)

                # Prediction output check
                if torch.isnan(v_follower_t_plus_k_plus_1_pred_batch).any() or torch.isinf(
                        v_follower_t_plus_k_plus_1_pred_batch).any():
                    print(f"Warning: IDM predicted speed is NaN/Inf at E:{epoch + 1}, B:{batch_idx}, K:{k_step}. Skipping this batch.")
                    skip_batch_update = True;
                    break
                all_predicted_follower_speeds_batch_list.append(v_follower_t_plus_k_plus_1_pred_batch.unsqueeze(1))

                # Prepare IDM input for the next time step (k_step+1) (if not the last prediction step)
                if k_step < pred_steps_K - 1:
                    use_ground_truth = torch.rand(1).item() < current_teacher_forcing_ratio  # Decide whether to use ground truth values

                    # Get predicted speed and position of the leader vehicle at t+k+1
                    v_leader_t_plus_k_plus_1_batch = batch_pred_leader_speeds_K_steps[:, k_step]
                    pos_leader_t_plus_k_plus_1_batch = batch_pred_leader_pos_K_steps[:, k_step]

                    if use_ground_truth:  # Teacher Forcing: use ground truth values to construct next step input
                        v_f_next_true = batch_true_follower_all_features_K_steps[:, k_step,
                                        0]  # Ground truth value corresponding to k_step, used as input state for k_step+1
                        s_actual_next_true = batch_true_follower_all_features_K_steps[:, k_step, 1]
                        delta_v_next_true = batch_true_follower_all_features_K_steps[:, k_step, 2]
                        a_f_next_true = batch_true_follower_all_features_K_steps[:, k_step, 3]
                        pos_f_next_true = batch_true_follower_pos_K_steps[:, k_step]

                        # New feature slice: [v_f_true, s_true, dv_true, a_f_true, v_l_pred]
                        new_feature_slice_batch = torch.stack([
                            v_f_next_true, s_actual_next_true, delta_v_next_true,
                            a_f_next_true, v_leader_t_plus_k_plus_1_batch  # Leader vehicle info comes from LNN prediction
                        ], dim=1)
                        # Update states needed for next IDM prediction round (based on ground truth)
                        batch_current_follower_speed_pred = v_f_next_true.clone()
                        batch_current_follower_pos = pos_f_next_true.clone()
                        batch_current_s_actual_for_idm = s_actual_next_true.clone()
                    else:  # Student Forcing: use model's own prediction to construct next step input
                        # Calculate ego acceleration: a = (v_pred(t+1) - v_current_pred(t)) / dt
                        a_follower_t_plus_k_plus_1_batch = (
                                                                       v_follower_t_plus_k_plus_1_pred_batch - batch_current_follower_speed_pred) / current_dt
                        a_follower_t_plus_k_plus_1_batch = torch.clamp(a_follower_t_plus_k_plus_1_batch, -10.0,
                                                                       10.0)  # Clamp

                        # Calculate ego vehicle displacement and new position
                        disp_follower_batch = batch_current_follower_speed_pred * current_dt + 0.5 * a_follower_t_plus_k_plus_1_batch * current_dt ** 2
                        pos_follower_t_plus_k_plus_1_batch = batch_current_follower_pos + disp_follower_batch

                        # Calculate new spacing
                        spacing_raw_t_plus_k_plus_1 = pos_leader_t_plus_k_plus_1_batch - pos_follower_t_plus_k_plus_1_batch
                        spacing_adjusted_t_plus_k_plus_1 = spacing_raw_t_plus_k_plus_1 - batch_d1_offset  # Correction
                        spacing_adjusted_t_plus_k_plus_1 = torch.clamp(spacing_adjusted_t_plus_k_plus_1, min=0.1)  # Clamp

                        # Calculate new speed difference
                        delta_v_t_plus_k_plus_1_batch = v_leader_t_plus_k_plus_1_batch - v_follower_t_plus_k_plus_1_pred_batch

                        # New feature slice: [v_f_pred, s_pred, dv_pred, a_f_pred, v_l_pred]
                        new_feature_slice_batch = torch.stack([
                            v_follower_t_plus_k_plus_1_pred_batch, spacing_adjusted_t_plus_k_plus_1,
                            delta_v_t_plus_k_plus_1_batch, a_follower_t_plus_k_plus_1_batch,
                            v_leader_t_plus_k_plus_1_batch
                        ], dim=1)
                        # Update states needed for next IDM prediction round (based on model's own prediction)
                        batch_current_follower_speed_pred = v_follower_t_plus_k_plus_1_pred_batch.clone()
                        batch_current_follower_pos = pos_follower_t_plus_k_plus_1_batch.clone()
                        batch_current_s_actual_for_idm = spacing_adjusted_t_plus_k_plus_1.clone()

                    # Check newly generated feature slice
                    if torch.isnan(new_feature_slice_batch).any() or torch.isinf(new_feature_slice_batch).any():
                        print(
                            f"Warning: new_feature_slice contains NaN/Inf at E:{epoch + 1}, B:{batch_idx}, K:{k_step}. Skipping this batch.")
                        skip_batch_update = True;
                        break

                    # Update IDM input sequence: remove the oldest, append the newest
                    batch_current_idm_input_torch = torch.cat(
                        [batch_current_idm_input_torch[:, 1:, :], new_feature_slice_batch.unsqueeze(1)], dim=1)

            if skip_batch_update: optimizer.zero_grad(); continue  # If escaped due to NaN, zero gradients and skip batch

            # Calculate loss for the current batch
            batch_predicted_multi_step_speeds = torch.cat(all_predicted_follower_speeds_batch_list, dim=1)
            if torch.isnan(batch_predicted_multi_step_speeds).any() or torch.isinf(
                    batch_predicted_multi_step_speeds).any():
                print(f"Warning: Final predicted speed sequence contains NaN/Inf at E:{epoch + 1}, B:{batch_idx}. Skipping this batch.")
                optimizer.zero_grad();
                continue

            squared_errors = criterion_mse_elementwise(batch_predicted_multi_step_speeds,
                                                       batch_true_follower_speeds_K_steps_for_loss)
            loss = (squared_errors * loss_weights.unsqueeze(0)).sum(dim=1).mean()  # Weighted average loss

            # Backpropagation and parameter update
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Final loss is NaN/Inf at E:{epoch + 1}, B:{batch_idx}. Skipping parameter update.")
                optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                total_loss_epoch += loss.item()
                num_valid_batches += 1

        avg_loss_epoch = total_loss_epoch / num_valid_batches if num_valid_batches > 0 else float('nan')
        print(
            f"[SAM-CausalGAT-LSTM-IDM Multi-step Training] Epoch [{epoch + 1}/{num_epochs}], "
            f"Average Loss: {avg_loss_epoch:.4f} (based on {num_valid_batches}/{len(train_loader)} valid batches)"
        )
        if np.isnan(avg_loss_epoch) and epoch > 0: print("Warning: Average loss is NaN, training stopped early."); break
    return model


def evaluate_generic_model(model, test_loader, pred_steps=5, model_name="Generic Model", is_fusion_model=False,
                           idm_model_frozen=None, lnn_ego_model_frozen=None, leader_model_frozen=None,
                           FUSION_HIST_LEN=None, HIST_LEN_idm_ego=None, DT_eval=None, device_eval=None):
    """
    Generic evaluation function, can be used to evaluate Leader LNN, Ego LNN-Ego.
    Evaluation for the fusion model will be done in the dedicated function `compute_multi_step_fusion_predictions`.
    """
    model.eval()  # Set model to evaluation mode
    all_predicted, all_true = [], []

    # Check if test loader is valid
    if not test_loader or not hasattr(test_loader, 'dataset') or len(test_loader.dataset) == 0:
        print(f"{model_name} Evaluation: Test dataset is empty. Skipping evaluation.")
        return

    with torch.no_grad():  # No gradient calculation
        if is_fusion_model:
            # This generic evaluation function is not directly used for the final fusion model evaluation, fusion evaluation has a dedicated function
            print(f"{model_name} (Fusion model) evaluation is handled by a dedicated function, skipping generic evaluation here.")
            return
        else:  # Used for LNN-Leader and LNN-Ego
            for batch_data, batch_target_speed in test_loader:
                batch_data, batch_target_speed = batch_data.to(device_eval), batch_target_speed.to(
                    device_eval)  # Use device_eval
                predicted_speed = model.predict_speed(batch_data) if hasattr(model, 'predict_speed') else model(
                    batch_data)
                all_predicted.append(predicted_speed.cpu())  # Move to CPU for storage
                all_true.append(batch_target_speed.cpu())  # Move to CPU for storage

    if not all_predicted: print(f"{model_name} Evaluation: No predictions made."); return
    all_predicted_cat = torch.cat(all_predicted, dim=0).numpy()
    all_true_cat = torch.cat(all_true, dim=0).numpy()
    if all_true_cat.shape[0] == 0: print(f"{model_name} Evaluation: Ground truth data is empty."); return

    # Calculate evaluation metrics
    mse_val = np.mean((all_predicted_cat - all_true_cat) ** 2)
    rmse_val = np.sqrt(mse_val)
    mae_val = np.mean(np.abs(all_predicted_cat - all_true_cat))
    # Step-by-step metrics
    mse_per_step = np.mean((all_predicted_cat - all_true_cat) ** 2, axis=0)
    rmse_per_step = np.sqrt(mse_per_step)
    mae_per_step = np.mean(np.abs(all_predicted_cat - all_true_cat), axis=0)

    print(f"\n{model_name} Evaluation (Overall {pred_steps} steps):")
    print(f"  Mean Squared Error (MSE): {mse_val:.4f}, Root Mean Squared Error (RMSE): {rmse_val:.4f}, Mean Absolute Error (MAE): {mae_val:.4f}")
    # Ensure index does not go out of bounds when printing step-by-step metrics
    for i in range(min(pred_steps, rmse_per_step.shape[0])):
        print(f"  Step {i + 1} prediction: RMSE: {rmse_per_step[i]:.4f}, MAE: {mae_per_step[i]:.4f}")

    # Plotting: Concatenate predicted and true trajectories for a subset of samples
    num_plot_samples = min(30, all_true_cat.shape[0])  # Plot at most 30 complete sequences
    # Calculate sampling interval to ensure uniform sample selection for plotting
    plot_step_interval = max(1, all_true_cat.shape[0] // num_plot_samples if num_plot_samples > 0 else 1)
    true_concat_plot, pred_concat_plot = [], []

    # Sample at intervals from all test samples, flatten K-step predicted/true values for each sampled item, and concatenate to plotting lists
    for i in range(0, all_true_cat.shape[0], plot_step_interval):
        if len(true_concat_plot) / pred_steps >= num_plot_samples: break  # If the upper limit of plotted samples is reached
        true_concat_plot.extend(all_true_cat[i])  # Extend ground truth list
        pred_concat_plot.extend(all_predicted_cat[i])  # Extend predicted value list

    if true_concat_plot:  # If there is data available for plotting
        plt.figure(figsize=(12, 6))
        plt.plot(np.array(true_concat_plot), linestyle='--', marker='o', markersize=3, label=f'True {model_name} Speed')
        plt.plot(np.array(pred_concat_plot), linestyle='-', marker='x', markersize=3, label=f'Predicted {model_name} Speed')
        plt.title(f'{model_name} Multi-step Speed Prediction (Sample Concatenation)')
        plt.xlabel('Time steps in prediction horizon (concatenated)')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.grid(True)
    else:
        print(f"Not enough samples in the {model_name} test set for plotting.")


# NEW: Helper function to get multi-step predictions from LSTM-IDM during fusion model training/evaluation
def predict_multi_step_idm_for_fusion_training(
        idm_model, leader_model, initial_idm_input_seq_batch, raw_data_slice_batch,
        pred_steps_K, dt, hist_len, device_compute
):
    """
    Helper function for fusion model training/evaluation to make multi-step predictions using the LSTM-IDM model.
    Args:
        idm_model (HybridIDMModel): Pre-trained LSTM-IDM model.
        leader_model (LiquidNeuralNetworkMultiStep): Pre-trained leader LNN model.
        initial_idm_input_seq_batch (torch.Tensor): Initial IDM input history for current batch (batch, hist_len, idm_input_dim).
        raw_data_slice_batch (torch.Tensor): Raw data slice for current batch, used to get initial position and leader history.
        pred_steps_K (int): Number of prediction steps.
        dt (float): Time step length.
        hist_len (int): Length of historical sequence.
        device_compute (torch.device): Computing device.
    Returns:
        torch.Tensor: Future K-step ego speed predicted by LSTM-IDM (batch, pred_steps_K).
    """
    idm_model.eval()  # Set to evaluation mode
    leader_model.eval()  # Set to evaluation mode

    # Precompute leader trajectory for current batch
    # Note: Tensors returned by precompute_leader_trajectories_for_idm_training should be moved to device_compute internally
    (_, initial_f_pos_batch, initial_s_safe_batch,
     pred_l_speeds_K_batch, pred_l_pos_K_batch, d1_offset_batch) = \
        precompute_leader_trajectories_for_idm_training(
            leader_model, raw_data_slice_batch.to(device_compute),  # Ensure raw_data_slice_batch is on correct device
            pred_steps_K, dt, device_compute, hist_len
        )
    # Double-check that all tensors involved in computation are on the same device
    initial_idm_input_seq_batch = initial_idm_input_seq_batch.to(device_compute)
    initial_f_pos_batch = initial_f_pos_batch.to(device_compute)
    initial_s_safe_batch = initial_s_safe_batch.to(device_compute)
    pred_l_speeds_K_batch = pred_l_speeds_K_batch.to(device_compute)
    pred_l_pos_K_batch = pred_l_pos_K_batch.to(device_compute)
    d1_offset_batch = d1_offset_batch.to(device_compute)

    # Initialize state variables for loop prediction
    batch_current_idm_input_torch = initial_idm_input_seq_batch.clone()
    batch_current_follower_speed_pred = batch_current_idm_input_torch[:, -1, 0].clone()  # Current ego vehicle speed
    batch_current_follower_pos = initial_f_pos_batch.clone()  # Current ego vehicle position
    batch_current_s_actual_for_idm = initial_s_safe_batch.clone()  # Current actual spacing input for IDM

    all_predicted_follower_speeds_batch_list = []  # Store K-step predictions
    current_dt_val = dt if dt > 1e-6 else 1e-6  # Ensure dt is valid

    with torch.no_grad():  # No gradient calculation
        for k_step in range(pred_steps_K):
            # Input check: if IDM input or actual spacing contains NaN, subsequent predictions may be meaningless
            if torch.isnan(batch_current_idm_input_torch).any() or torch.isnan(batch_current_s_actual_for_idm).any():
                # Simple handling: use predicted speed from previous time step or fill with 0
                v_follower_pred = batch_current_follower_speed_pred.clone()
                if k_step == 0 and torch.isnan(v_follower_pred).any():  # If the first step input is already NaN
                    v_follower_pred = torch.zeros_like(v_follower_pred)  # Fill with 0
                print(f"Warning: IDM input contains NaN in predict_multi_step_idm_for_fusion_training (step {k_step}). Using fallback speed.")
            else:  # Input is normal, proceed with IDM prediction
                v_follower_pred_unsqueeze, _ = idm_model.predict_speed(
                    batch_current_idm_input_torch, batch_current_s_actual_for_idm
                )
                v_follower_pred = v_follower_pred_unsqueeze.squeeze(1)

            # IDM prediction output check: handle NaN/Inf
            if torch.isnan(v_follower_pred).any() or torch.isinf(v_follower_pred).any():
                nan_inf_mask = torch.isnan(v_follower_pred) | torch.isinf(v_follower_pred)
                # Attempt to fill NaN/Inf parts with valid predicted values from previous time step
                v_follower_pred[nan_inf_mask] = batch_current_follower_speed_pred[nan_inf_mask]
                if torch.isnan(v_follower_pred).any():  # If previous time step is also NaN, fill with 0
                    v_follower_pred[torch.isnan(v_follower_pred)] = 0.0
                print(
                    f"Warning: IDM prediction contains NaN/Inf in predict_multi_step_idm_for_fusion_training (step {k_step}). Using fallback speed.")

            all_predicted_follower_speeds_batch_list.append(v_follower_pred.unsqueeze(1))  # Store prediction

            # Prepare IDM input for the next time step (if not the last prediction step)
            # Note: Teacher Forcing is not used here, as the goal is to get pure predictions from the model itself
            if k_step < pred_steps_K - 1:
                v_leader_next = pred_l_speeds_K_batch[:, k_step]  # Predicted leader speed for the next time step
                pos_leader_next = pred_l_pos_K_batch[:, k_step]  # Predicted leader position for the next time step

                # Calculate ego vehicle acceleration, displacement, new position
                a_follower_next = (v_follower_pred - batch_current_follower_speed_pred) / current_dt_val
                a_follower_next = torch.clamp(a_follower_next, -10.0, 10.0)  # Clamp

                disp_follower_batch = batch_current_follower_speed_pred * current_dt_val + \
                                      0.5 * a_follower_next * current_dt_val ** 2
                pos_follower_next = batch_current_follower_pos + disp_follower_batch

                # Calculate new spacing and new speed difference
                spacing_raw_next = pos_leader_next - pos_follower_next
                spacing_adjusted_next = spacing_raw_next - d1_offset_batch  # Correction
                spacing_adjusted_next = torch.clamp(spacing_adjusted_next, min=0.1)  # Clamp
                delta_v_next = v_leader_next - v_follower_pred

                # Construct new feature slice: [v_f_pred, s_pred, dv_pred, a_f_pred, v_l_pred]
                new_feature_slice = torch.stack([
                    v_follower_pred, spacing_adjusted_next, delta_v_next,
                    a_follower_next, v_leader_next
                ], dim=1)

                # New feature slice check: if it contains NaN, special handling might be needed
                # For simplicity, assume if there are no NaNs upstream, they are unlikely to generate here (unless extreme cases like very small dt)
                # If they do occur, one strategy is to keep IDM input unchanged or fill with old features
                if torch.isnan(new_feature_slice).any():
                    print(
                        f"Warning: new_feature_slice contains NaN in predict_multi_step_idm_for_fusion_training (step {k_step}). IDM input may not be correctly updated.")
                    # Simple handling: do not update IDM input, let it use old input in the next step. More complex strategies might involve filling with 0 or the previous valid value.
                    # Keep batch_current_idm_input_torch unchanged, relying on the previous step's v_follower_pred as current_follower_speed_pred
                else:  # Normally update IDM input sequence
                    batch_current_idm_input_torch = torch.cat(
                        [batch_current_idm_input_torch[:, 1:, :], new_feature_slice.unsqueeze(1)], dim=1
                    )

                # Update states for the next loop round
                batch_current_follower_speed_pred = v_follower_pred.clone()
                batch_current_follower_pos = pos_follower_next.clone()
                batch_current_s_actual_for_idm = spacing_adjusted_next.clone()

    return torch.cat(all_predicted_follower_speeds_batch_list, dim=1)  # Return (batch, pred_steps_K)


def compute_multi_step_fusion_predictions(
        idm_model, leader_model_for_idm, lnn_ego_model, fusion_model,
        raw_data_test_slice, label_data_test_slice,  # Raw data and label data for the test set
        dt, pred_steps, hist_len_idm_ego, hist_len_fusion, device_comp
):
    """
    Perform final fusion multi-step predictions and evaluate performance.
    This is the final evaluation step of the entire system.
    """
    # Set all models to evaluation mode
    idm_model.eval();
    leader_model_for_idm.eval();
    lnn_ego_model.eval();
    fusion_model.eval()

    N_test = raw_data_test_slice.shape[0]  # Number of test samples
    if N_test == 0: print("Fusion Evaluation: Test data is empty. Skipping evaluation."); return

    # --- 1. Get y_lstm_idm_pred (predicted output of the LSTM-IDM model) ---
    # LSTM-IDM input history features: [Ego speed, Spacing, Speed diff, Ego accel, Leader speed]
    idm_input_hist_test = raw_data_test_slice[:, -hist_len_idm_ego:, [0, 1, 2, 3, 5]].clone() * 0.3048  # Unit conversion
    # Use helper function for multi-step prediction
    with torch.no_grad():
        y_lstm_idm_pred = predict_multi_step_idm_for_fusion_training(
            idm_model, leader_model_for_idm, idm_input_hist_test, raw_data_test_slice,
            pred_steps, dt, hist_len_idm_ego, device_comp
        ).to(device_comp)  # Ensure on correct device

    # --- 2. Get y_lnn_ego_pred (predicted output of the LNN-Ego model) ---
    # LNN-Ego input history (same structure as IDM input)
    lnn_ego_input_test = raw_data_test_slice[:, -hist_len_idm_ego:, [0, 1, 2, 3, 5]].clone() * 0.3048  # Unit conversion
    with torch.no_grad():
        y_lnn_ego_pred = lnn_ego_model.predict_speed(lnn_ego_input_test.to(device_comp)).to(device_comp)  # Ensure on correct device

    # --- 3. Prepare input for FusionLSTM ---
    # Ego historical speed: (batch, hist_len_fusion, 1)
    ego_speed_hist_fusion_test = raw_data_test_slice[:, -hist_len_fusion:, 0].unsqueeze(-1).clone() * 0.3048
    # Leader historical speed: (batch, hist_len_fusion, 1)
    leader_speed_hist_fusion_test = raw_data_test_slice[:, -hist_len_fusion:, 5].unsqueeze(-1).clone() * 0.3048

    # --- 4. Get the fusion gating value alpha through FusionLSTM ---
    with torch.no_grad():
        alpha_test = fusion_model(ego_speed_hist_fusion_test.to(device_comp),
                                  leader_speed_hist_fusion_test.to(device_comp)).to(device_comp)  # Ensure on correct device

    # --- 5. Calculate final fusion prediction y_fusion ---
    # Ensure all tensors involved in computation are on the same device and use .detach() (although already inside no_grad block here)
    y_fusion_pred = alpha_test * y_lstm_idm_pred.detach() + (1 - alpha_test) * y_lnn_ego_pred.detach()
    y_fusion_pred_np = y_fusion_pred.cpu().numpy()  # Convert to NumPy array for evaluation

    # --- Load ground truth ego trajectory data for evaluation ---
    # The 0th column of label_data_test_slice is the ego future K-step speed
    true_f_speeds_np = label_data_test_slice[:, :pred_steps, 0].clone().cpu().numpy() * 0.3048  # Unit conversion

    # --- Calculate evaluation metrics ---
    # Replace near-zero ground truth values with a tiny value to avoid division by zero when calculating MAPE
    true_f_speeds_mape = np.where(np.abs(true_f_speeds_np) < 1e-5, 1e-5, true_f_speeds_np)

    mse_fusion = np.mean((y_fusion_pred_np - true_f_speeds_np) ** 2)
    rmse_fusion = np.sqrt(mse_fusion)
    mae_fusion = np.mean(np.abs(y_fusion_pred_np - true_f_speeds_np))
    mape_fusion = np.mean(np.abs((y_fusion_pred_np - true_f_speeds_np) / true_f_speeds_mape)) * 100

    # Step-by-step evaluation metrics
    rmse_per_step = np.sqrt(np.mean((y_fusion_pred_np - true_f_speeds_np) ** 2, axis=0))
    mae_per_step = np.mean(np.abs(y_fusion_pred_np - true_f_speeds_np), axis=0)

    print(f"\nFinal Fusion Model Prediction Results ({pred_steps} steps):")
    print(f"  Mean Squared Error (MSE): {mse_fusion:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse_fusion:.4f} m/s")
    print(f"  Mean Absolute Error (MAE): {mae_fusion:.4f} m/s")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape_fusion:.2f}%")
    for i in range(pred_steps):
        print(f"  Step {i + 1} prediction (Fusion): RMSE: {rmse_per_step[i]:.4f}, MAE: {mae_per_step[i]:.4f}")

    # --- Plotting: Concatenate predicted and true trajectories for a subset of samples ---
    num_plot_samples = min(30, N_test)
    plot_interval = max(1, N_test // num_plot_samples if num_plot_samples > 0 else 1)
    true_concat_plot, pred_concat_plot = [], []
    for i in range(0, N_test, plot_interval):
        if len(true_concat_plot) / pred_steps >= num_plot_samples: break
        true_concat_plot.extend(true_f_speeds_np[i, :])
        pred_concat_plot.extend(y_fusion_pred_np[i, :])

    if true_concat_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(np.array(true_concat_plot), linestyle='--', marker='o', markersize=3, label='True ego speed (Fusion Eval)')
        plt.plot(np.array(pred_concat_plot), linestyle='-', marker='x', markersize=3, label='Predicted fusion speed (Fusion Eval)')
        plt.title(f'Final Fusion Model Ego Speed Multi-step Prediction (Sample Concatenation)')
        plt.xlabel('Time steps in prediction horizon (concatenated)')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.grid(True)
    else:
        print("Not enough test data in fusion evaluation to plot speed curves.")


# =========================
# Main function (Script execution entry point)
# =========================
if __name__ == "__main__":
    torch.manual_seed(42)  # Set random seed to ensure reproducible results
    np.random.seed(42)
    # torch.autograd.set_detect_anomaly(True) # Enable during debugging to help locate NaN/Inf issues, but it slows down execution

    # --- Define hyperparameters and configuration ---
    PRED_STEPS_K = 10  # Number of future prediction steps
    DT = 0.1  # Time step (seconds)
    HIST_LEN = 50  # Length of historical sequence for LSTM-IDM and LNN-Ego input
    FUSION_HIST_LEN = 20  # Length of historical sequence for FusionLSTM input (ego/leader historical speed)

    IDM_INPUT_DIM = 5  # Input feature dimension of LSTM-IDM (v_f, s, dv, a_f, v_l)
    LEADER_LNN_INPUT_DIM = 2  # Input feature dimension of Leader LNN model (v_l, a_l)
    LNN_EGO_INPUT_DIM = IDM_INPUT_DIM  # LNN-Ego input dimension is consistent with IDM
    FUSION_LSTM_INPUT_DIM = 2  # FusionLSTM input dimension (v_f_hist_for_fusion, v_l_hist_for_fusion)

    # Model hidden layer dimensions
    HIDDEN_DIM_IDM = 64
    GAT_HIDDEN_DIM_IDM = 16
    HIDDEN_DIM_LNN_LEADER = 64
    HIDDEN_DIM_LNN_EGO = 64  # LNN-Ego hidden dimension
    HIDDEN_DIM_FUSION_LSTM = 32  # FusionLSTM hidden dimension (adjustable)

    # Number of model layers
    NUM_LAYERS_IDM = 1
    NUM_LAYERS_LNN_LEADER = 1
    NUM_LAYERS_LNN_EGO = 1  # Number of LNN-Ego layers
    NUM_LAYERS_FUSION_LSTM = 1  # Number of FusionLSTM layers

    # Number of training epochs (adjustable as needed, more epochs usually yield better results but take longer)
    LEADER_LNN_EPOCHS = 100 # Originally 50, can be appropriately increased, e.g., 100
    IDM_MULTISTEP_EPOCHS = 100  # Originally 50, can be appropriately increased, e.g., 100
    LNN_EGO_EPOCHS = 100  # Number of LNN-Ego training epochs
    FUSION_LSTM_EPOCHS = 30  # Number of FusionLSTM training epochs
    SAM_TRAIN_EPOCHS = 10
    SAM_TEST_EPOCHS = 10

    BATCH_SIZE = 32  # Batch size (Originally 32, trying to increase)

    # --- Load Data ---
    try:
        # Please replace this path with the actual path to your .mat data file
        data_path = 'E:\\pythonProject1\\data_ngsim\\data_10.mat'
        # Example: data_path = 'E:\\pythonProject1\\data_ngsim\\data_5.mat'
        if not os.path.exists(data_path):
            print(f"Error: Data file '{data_path}' not found. Please check the path.");
            exit()
        data = sio.loadmat(data_path)
        print(f"Data loaded from '{data_path}' contains keys: {list(data.keys())}")
    except Exception as e:
        print(f"Error loading data: {e}");
        exit()

    # Ensure '.mat' file contains 'train_data' and 'lable_data' (or 'label_data')
    # 'train_data': Historical observation data [Current speed, Following distance, Speed difference, Ego acceleration, Ego position, Leader speed, Leader acceleration, Leader position]
    # 'lable_data': Future target data [Speed, Leader distance, Ego acceleration, Ego position, Leader speed, Leader future position]
    if 'train_data' not in data or ('lable_data' not in data and 'label_data' not in data):
        print("Error: 'train_data' or 'lable_data'/'label_data' not found in .mat file. Please check the data file.")
        exit()

    # Use 'lable_data' or 'label_data' (if it exists)
    label_key = 'lable_data' if 'lable_data' in data else 'label_data'

    raw_data_full = torch.tensor(data['train_data'], dtype=torch.float32)
    label_data_full = torch.tensor(data[label_key], dtype=torch.float32)

    # --- Data Subset Selection and Splitting ---
    total_samples_full = raw_data_full.shape[0]
    # Use a subset of data for quick testing and development, e.g., 10% or a fixed amount
    # num_samples_to_use = int(total_samples_full * 0.1)
    num_samples_to_use = int(total_samples_full * 0.1)  # Use 30% of data for processing (originally 0.2)
    # num_samples_to_use = total_samples_full # Use all data

    raw_data_all = raw_data_full[:num_samples_to_use]
    label_data_all = label_data_full[:num_samples_to_use]
    print(f"Using {num_samples_to_use} samples for processing (Total samples: {total_samples_full})")
    if num_samples_to_use == 0: print("Error: No samples available for use."); exit()

    # Main train/test split (used for final testing of LNN-Ego, FusionLSTM)
    train_ratio_main = 0.8  # Main training set ratio
    num_total_main = raw_data_all.shape[0]
    num_train_main = int(num_total_main * train_ratio_main)
    # Ensure both train and test sets have data even with few samples (if total > 1)
    if num_train_main == 0 and num_total_main > 0: num_train_main = max(1,
                                                                        num_total_main - 1 if num_total_main > 1 else 1)
    if num_train_main == num_total_main and num_total_main > 1: num_train_main = num_total_main - 1  # Ensure test set has at least 1

    num_test_main = num_total_main - num_train_main

    print(f"Main split: Total samples {num_total_main}, Training samples {num_train_main}, Testing samples {num_test_main}")

    # Split data
    raw_train_data = raw_data_all[:num_train_main]  # Used to train LNN-Leader, LSTM-IDM, LNN-Ego, FusionLSTM
    label_train_data = label_data_all[:num_train_main]  # Corresponding label data
    raw_test_data = raw_data_all[num_train_main:]  # Used for final fusion model evaluation, can also be used for independent sub-model evaluation
    label_test_data = label_data_all[num_train_main:]  # Corresponding label data

    # --- 1. Train Leader LNN Model (Leader LNN) ---
    print("\n--- 1. Train Leader LNN Model (Leader LNN) ---")
    # Leader LNN Input: Leader historical speed and acceleration (columns [5, 6])
    leader_lnn_input_hist_train = raw_train_data[:, -HIST_LEN:, [5, 6]].clone() * 0.3048
    # Leader LNN Target: Leader future K-step speed (4th column of label_data)
    leader_lnn_target_speeds_train = label_train_data[:, :PRED_STEPS_K, 4].clone() * 0.3048

    # Leader LNN Test Data (Using part or all of the main test set)
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
    optimizer_lead = optim.Adam(leader_model.parameters(), lr=1e-3)  # Adam optimizer
    criterion_mse = nn.MSELoss()  # Mean Squared Error loss

    if leader_lnn_train_loader:
        train_generic_model(leader_model, leader_lnn_train_loader, optimizer_lead, criterion_mse,
                            LEADER_LNN_EPOCHS, "Leader LNN", clip_value=1.0)
    if leader_lnn_test_loader:  # Evaluate Leader LNN
        evaluate_generic_model(leader_model, leader_lnn_test_loader, PRED_STEPS_K, "Leader LNN",
                               device_eval=device)

    # --- 2. Prepare and train SAM-CausalGAT-LSTM-IDM model ---
    # This model uses the main training set (raw_train_data, label_train_data) and learns the causal prior matrix via SAM.
    # Internally, it still needs leader_model to predict the leader's trajectory, keeping the interface consistent with the original LSTM-IDM.
    print("\n--- 2. Train SAM-CausalGAT-LSTM-IDM model ---")
    idm_multistep_train_loader = None
    if raw_train_data.shape[0] > 0:
        # Precompute leader trajectories and other data required for IDM training
        (initial_idm_seq_train, initial_f_pos_train, initial_s_safe_train,
         pred_l_speeds_K_train, pred_l_pos_K_train, d1_train) = \
            precompute_leader_trajectories_for_idm_training(
                leader_model, raw_train_data, PRED_STEPS_K, DT, device, HIST_LEN  # Use the trained leader_model
            )
        # Ground truth ego future K-step speed (used for loss calculation)
        true_f_speeds_K_train_for_loss = label_train_data[:, :PRED_STEPS_K, 0].clone() * 0.3048

        # Prepare all relevant ground truth ego future K-step features required for Teacher Forcing
        # Feature order: [Speed(v_f), Spacing(s), Speed diff(dv=v_l-v_f), Acceleration(a_f)]
        true_v_f_K = label_train_data[:, :PRED_STEPS_K, 0].clone()
        true_s_K = label_train_data[:, :PRED_STEPS_K, 1].clone()
        true_a_f_K = label_train_data[:, :PRED_STEPS_K, 2].clone()
        true_v_l_K = label_train_data[:, :PRED_STEPS_K, 4].clone()  # Ground truth leader future speed (from labels)
        # Note: During IDM training, future leader speed is predicted by LNN, but delta_v for TF can be calculated using label data
        true_dv_K = true_v_l_K - true_v_f_K
        # Unified unit conversion
        true_f_all_features_K_train = torch.stack([true_v_f_K, true_s_K, true_dv_K, true_a_f_K], dim=2) * 0.3048
        # Ground truth ego future K-step position
        true_f_pos_K_train = label_train_data[:, :PRED_STEPS_K, 3].clone() * 0.3048

        if initial_idm_seq_train.shape[0] > 0:  # Ensure data exists after precomputation
            idm_multistep_train_dataset = torch.utils.data.TensorDataset(
                initial_idm_seq_train, true_f_speeds_K_train_for_loss.to(device),  # Ensure labels are on the same device
                initial_f_pos_train, initial_s_safe_train,
                pred_l_speeds_K_train, pred_l_pos_K_train, d1_train,
                true_f_all_features_K_train.to(device), true_f_pos_K_train.to(device)
            )
            idm_multistep_train_loader = torch.utils.data.DataLoader(idm_multistep_train_dataset, batch_size=BATCH_SIZE,
                                                                     shuffle=True)
            print(f"SAM-CausalGAT-LSTM-IDM: Number of training data samples {initial_idm_seq_train.shape[0]}")
        else:
            print("SAM-CausalGAT-LSTM-IDM: Training data is empty after precomputation.")

    A_causal_normalized = build_sam_causal_matrix(
        initial_idm_seq_train if raw_train_data.shape[0] > 0 else torch.empty(0, HIST_LEN, IDM_INPUT_DIM),
        sam_train_epochs=SAM_TRAIN_EPOCHS,
        sam_test_epochs=SAM_TEST_EPOCHS,
        sam_batch_size=BATCH_SIZE,
    )
    print("SAM normalized causal matrix (as CausalGAT prior input):")
    print(A_causal_normalized.numpy())

    idm_model = SAMCausalGATLSTMIDMModel(
        input_dim=IDM_INPUT_DIM,
        gat_hidden_dim=GAT_HIDDEN_DIM_IDM,
        hidden_dim=HIDDEN_DIM_IDM,
        num_layers=NUM_LAYERS_IDM,
        dt=DT,
        causal_matrix=A_causal_normalized.to(device),
    ).to(device)
    initialize_weights(idm_model)
    optimizer_idm = optim.Adam(idm_model.parameters(), lr=2e-4, weight_decay=1e-5)  # Adam, lr=0.0002, L2 regularization
    if idm_multistep_train_loader:
        train_idm_model_multistep(
            idm_model, idm_multistep_train_loader, optimizer_idm,
            IDM_MULTISTEP_EPOCHS, PRED_STEPS_K, DT, alpha_decay=0.05,  # Loss weight decay
            teacher_forcing_initial_ratio=1.0, min_teacher_forcing_ratio=0.0,  # Teacher Forcing parameters
            teacher_forcing_decay_epochs_ratio=0.75, clip_value=1.0  # Gradient clipping
        )
    else:
        print("SAM-CausalGAT-LSTM-IDM: Skipping training because training data loader is empty.")

    # --- 2.1. NEW: Evaluate SAM-CausalGAT-LSTM-IDM Model ---
    print("\n--- 2.1. Evaluate SAM-CausalGAT-LSTM-IDM Model ---")
    if raw_test_data.shape[0] > 0 and label_test_data.shape[0] > 0:
        idm_model.eval()  # Ensure model is in evaluation mode
        leader_model.eval()  # Leader model also needs to be in evaluation mode

        # Prepare input for LSTM-IDM prediction on the test set
        initial_idm_input_hist_test_for_eval = raw_test_data[:, -HIST_LEN:, [0, 1, 2, 3, 5]].clone().to(device) * 0.3048

        with torch.no_grad():
            y_lstm_idm_pred_test = predict_multi_step_idm_for_fusion_training(
                idm_model, leader_model,
                initial_idm_input_hist_test_for_eval,  # This is already unit-converted and moved to device
                raw_test_data,  # This function handles moving to device and unit conversion internally
                PRED_STEPS_K, DT, HIST_LEN, device
            )

        y_lstm_idm_pred_np_test = y_lstm_idm_pred_test.cpu().numpy()

        # Ground truth ego future K-step speed of the test set
        true_f_speeds_np_test_idm = label_test_data[:, :PRED_STEPS_K, 0].clone().cpu().numpy() * 0.3048

        if true_f_speeds_np_test_idm.shape[0] > 0 and y_lstm_idm_pred_np_test.shape[0] == \
                true_f_speeds_np_test_idm.shape[0]:
            mse_idm_eval = np.mean((y_lstm_idm_pred_np_test - true_f_speeds_np_test_idm) ** 2)
            rmse_idm_eval = np.sqrt(mse_idm_eval)
            mae_idm_eval = np.mean(np.abs(y_lstm_idm_pred_np_test - true_f_speeds_np_test_idm))

            rmse_per_step_idm_eval = np.sqrt(
                np.mean((y_lstm_idm_pred_np_test - true_f_speeds_np_test_idm) ** 2, axis=0))
            mae_per_step_idm_eval = np.mean(np.abs(y_lstm_idm_pred_np_test - true_f_speeds_np_test_idm), axis=0)

            print(f"SAM-CausalGAT-LSTM-IDM Model Evaluation ({PRED_STEPS_K} steps):")
            print(f"  Mean Squared Error (MSE): {mse_idm_eval:.4f}")
            print(f"  Root Mean Squared Error (RMSE): {rmse_idm_eval:.4f} m/s")
            print(f"  Mean Absolute Error (MAE): {mae_idm_eval:.4f} m/s")
            for i in range(PRED_STEPS_K):
                if i < rmse_per_step_idm_eval.shape[0]:  # Check bounds
                    print(
                        f"  Step {i + 1} prediction (SAM-CausalGAT-LSTM-IDM): "
                        f"RMSE: {rmse_per_step_idm_eval[i]:.4f}, MAE: {mae_per_step_idm_eval[i]:.4f}")
                else:
                    print(f"  Step {i + 1} prediction (SAM-CausalGAT-LSTM-IDM): Insufficient metric data")

            # Plotting for LSTM-IDM
            num_plot_samples_idm = min(30, true_f_speeds_np_test_idm.shape[0])
            plot_interval_idm = max(1, true_f_speeds_np_test_idm.shape[
                                           0] // num_plot_samples_idm if num_plot_samples_idm > 0 else 1)
            true_concat_plot_idm, pred_concat_plot_idm = [], []

            for i in range(0, true_f_speeds_np_test_idm.shape[0], plot_interval_idm):
                if (len(true_concat_plot_idm) / PRED_STEPS_K if PRED_STEPS_K > 0 else 0) >= num_plot_samples_idm: break
                true_concat_plot_idm.extend(true_f_speeds_np_test_idm[i, :])
                pred_concat_plot_idm.extend(y_lstm_idm_pred_np_test[i, :])

            if true_concat_plot_idm:
                plt.figure(figsize=(12, 6))
                plt.plot(np.array(true_concat_plot_idm), linestyle='--', marker='o', markersize=3,
                         label='True ego speed (IDM Eval)')
                plt.plot(np.array(pred_concat_plot_idm), linestyle='-', marker='x', markersize=3,
                         label='Predicted IDM speed (IDM Eval)')
                plt.title(f'SAM-CausalGAT-LSTM-IDM Model Ego Speed Multi-step Prediction (Sample Concatenation)')
                plt.xlabel('Time steps in prediction horizon (concatenated)')
                plt.ylabel('Speed (m/s)')
                plt.legend()
                plt.grid(True)
            else:
                print("SAM-CausalGAT-LSTM-IDM Eval: Not enough test data to plot speed curves.")
        else:
            print("SAM-CausalGAT-LSTM-IDM Eval: Test data or predicted data is empty or shape mismatch. Skipping metric calculation and plotting.")
    else:
        print("SAM-CausalGAT-LSTM-IDM Eval: Insufficient test data, skipping evaluation.")

    # --- 3. Prepare and train Ego LNN model (LNN-Ego) ---
    print("\n--- 3. Train Ego LNN-Ego model ---")
    # LNN-Ego input is identical to IDM: [v_f, s, dv, a_f, v_l] (columns [0,1,2,3,5] of raw_data)
    lnn_ego_input_hist_train = raw_train_data[:, -HIST_LEN:, [0, 1, 2, 3, 5]].clone() * 0.3048
    # LNN-Ego Target: Ego future K-step speed (0th column of label_data)
    lnn_ego_target_speeds_train = label_train_data[:, :PRED_STEPS_K, 0].clone() * 0.3048

    # LNN-Ego Test Data (Using main test set)
    lnn_ego_input_hist_test = raw_test_data[:, -HIST_LEN:, [0, 1, 2, 3, 5]].clone() * 0.3048
    lnn_ego_target_speeds_test = label_test_data[:, :PRED_STEPS_K, 0].clone() * 0.3048

    lnn_ego_train_loader, lnn_ego_test_loader = None, None
    if lnn_ego_input_hist_train.shape[0] > 0:
        lnn_ego_train_dataset = torch.utils.data.TensorDataset(lnn_ego_input_hist_train, lnn_ego_target_speeds_train)
        lnn_ego_train_loader = torch.utils.data.DataLoader(lnn_ego_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if lnn_ego_input_hist_test.shape[0] > 0:
        lnn_ego_test_dataset = torch.utils.data.TensorDataset(lnn_ego_input_hist_test, lnn_ego_target_speeds_test)
        lnn_ego_test_loader = torch.utils.data.DataLoader(lnn_ego_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    lnn_ego_model = LiquidNeuralNetworkMultiStepEgo(
        LNN_EGO_INPUT_DIM, HIDDEN_DIM_LNN_EGO, PRED_STEPS_K, NUM_LAYERS_LNN_EGO, HIST_LEN, DT
    ).to(device)
    initialize_weights(lnn_ego_model)
    optimizer_lnn_ego = optim.Adam(lnn_ego_model.parameters(), lr=1e-3)
    # criterion_mse (already defined earlier)

    if lnn_ego_train_loader:
        train_generic_model(lnn_ego_model, lnn_ego_train_loader, optimizer_lnn_ego, criterion_mse,
                            LNN_EGO_EPOCHS, "Ego LNN (LNN-Ego)", clip_value=1.0)
    if lnn_ego_test_loader:  # Evaluate LNN-Ego
        evaluate_generic_model(lnn_ego_model, lnn_ego_test_loader, PRED_STEPS_K, "Ego LNN (LNN-Ego)",
                               device_eval=device)

    # --- 4. Prepare and train Fusion LSTM model (FusionLSTM) ---
    # FusionLSTM training also uses the main training set (raw_train_data, label_train_data)
    print("\n--- 4. Train Fusion LSTM model (FusionLSTM) ---")
    # FusionLSTM Input: Ego historical speed (FUSION_HIST_LEN, 1), Leader historical speed (FUSION_HIST_LEN, 1) - extracted from raw_train_data
    # FusionLSTM Target: Ground truth ego future K-step speed (used to calculate fusion loss) - extracted from label_train_data
    fusion_train_loader = None
    if raw_train_data.shape[0] > 0:
        # DataLoader for fusion training yields (raw_batch, label_batch)
        # raw_batch contains all historical info, label_batch contains target future speed
        fusion_raw_label_train_dataset = torch.utils.data.TensorDataset(raw_train_data, label_train_data)
        fusion_train_loader = torch.utils.data.DataLoader(fusion_raw_label_train_dataset, batch_size=BATCH_SIZE,
                                                          shuffle=True)
        print(f"FusionLSTM: Number of training data samples {raw_train_data.shape[0]}")

    fusion_model = FusionLSTMModel(
        FUSION_LSTM_INPUT_DIM, HIDDEN_DIM_FUSION_LSTM, PRED_STEPS_K, NUM_LAYERS_FUSION_LSTM
    ).to(device)
    initialize_weights(fusion_model)
    optimizer_fusion = optim.Adam(fusion_model.parameters(), lr=1e-3)
    # criterion_mse (already defined earlier)

    if fusion_train_loader:
        # When training the fusion model, pre-trained, frozen sub-models must be passed in
        train_generic_model(
            model=fusion_model,
            loader=fusion_train_loader,
            optimizer=optimizer_fusion,
            criterion=criterion_mse,
            epochs=FUSION_LSTM_EPOCHS,
            model_name="Fusion LSTM",
            clip_value=1.0,
            is_fusion_model=True,  # Flag indicating the fusion model is being trained
            idm_model_frozen=idm_model,  # Pass in the trained LSTM-IDM model
            lnn_ego_model_frozen=lnn_ego_model,  # Pass in the trained LNN-Ego model
            leader_model_frozen=leader_model,  # Pass in the trained Leader LNN model (used inside IDM)
            # Specific parameters required for FusionLSTM training
            FUSION_HIST_LEN=FUSION_HIST_LEN,
            PRED_STEPS_K_fusion=PRED_STEPS_K,
            DT_fusion=DT,
            HIST_LEN_idm_ego_fusion=HIST_LEN,  # History length for IDM and LNN-Ego
            device_fusion=device  # Computing device
        )
    else:
        print("FusionLSTM: Skipping training because training data loader is empty.")

    # --- 5. Final Fusion Model Evaluation ---
    # Use the main test set (raw_test_data, label_test_data) for final evaluation
    print("\n--- 5. Final Fusion Model Evaluation ---")
    if raw_test_data.shape[0] > 0 and label_test_data.shape[0] > 0:
        compute_multi_step_fusion_predictions(
            idm_model, leader_model, lnn_ego_model, fusion_model,  # Pass in all trained models
            raw_test_data, label_test_data,  # Test data
            DT, PRED_STEPS_K, HIST_LEN, FUSION_HIST_LEN, device  # Relevant parameters
        )
    else:
        print("Not enough test data for final fusion model evaluation.")

    plt.show()  # Display all matplotlib images accumulated during the script execution
    print("\n--- All processes executed successfully ---")
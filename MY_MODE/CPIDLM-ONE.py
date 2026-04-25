import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, TensorDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################################
# 1. SAM Causal Discovery Model (Structural Agnostic Model)
#############################################
class CNormalized_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(CNormalized_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        norm_weight = self.weight / self.weight.pow(2).sum(dim=0, keepdim=True).sqrt()
        return nn.functional.linear(input, norm_weight, self.bias)


class SAM_discriminator(nn.Module):
    def __init__(self, sizes, zero_components=[], **kwargs):
        super(SAM_discriminator, self).__init__()
        activation_function = kwargs.get('activation_function', nn.ReLU)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        dropout = kwargs.get("dropout", 0.0)
        layers = []
        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(nn.Linear(i, j))
            if batch_norm:
                layers.append(nn.BatchNorm1d(j))
            if dropout != 0.0:
                layers.append(nn.Dropout(p=dropout))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SAM_block(nn.Module):
    def __init__(self, sizes, zero_components=[], **kwargs):
        super(SAM_block, self).__init__()
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        activation_function = kwargs.get('activation_function', nn.Tanh)
        activation_argument = kwargs.get('activation_argument', None)
        batch_norm = kwargs.get("batch_norm", False)
        layers = []
        for i, j in zip(sizes[:-2], sizes[1:-1]):
            layers.append(CNormalized_Linear(i, j))
            if batch_norm:
                layers.append(nn.BatchNorm1d(j))
            if activation_argument is None:
                layers.append(activation_function())
            else:
                layers.append(activation_function(activation_argument))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.layers = nn.Sequential(*layers)
        self.register_buffer('_filter', torch.ones(1, sizes[0]))
        for i in zero_components:
            self._filter[:, i] = 0.0
        self.fs_filter = nn.Parameter(self._filter.clone())
        if gpu:
            self._filter = self._filter.cuda(gpu_no)

    def forward(self, x):
        filtered_x = x * (self._filter * self.fs_filter).expand_as(x)
        return self.layers(filtered_x)


class SAM_generators(nn.Module):
    def __init__(self, data_shape, zero_components, nh=200, batch_size=-1, **kwargs):
        super(SAM_generators, self).__init__()
        rows, self.cols = data_shape
        if batch_size == -1:
            batch_size = rows
        gpu = kwargs.get('gpu', False)
        gpu_no = kwargs.get('gpu_no', 0)
        self.noise = [torch.randn(batch_size, 1) for _ in range(self.cols)]
        if gpu:
            self.noise = [n.cuda(gpu_no) for n in self.noise]
        self.blocks = nn.ModuleList()
        for i in range(self.cols):
            self.blocks.append(SAM_block([self.cols + 1, nh, 1], zero_components[i], **kwargs))

    def forward(self, x):
        batch_size = x.size(0)
        self.noise = [torch.randn(batch_size, 1, device=x.device) for _ in range(self.cols)]
        generated_variables = [self.blocks[i](torch.cat([x, self.noise[i]], dim=1)) for i in range(self.cols)]
        return generated_variables


def run_SAM(df_data, skeleton=None, **kwargs):
    gpu = kwargs.get('gpu', False)
    gpu_no = kwargs.get('gpu_no', 0)
    train_epochs = kwargs.get('train_epochs', 1000)
    test_epochs = kwargs.get('test_epochs', 1000)
    batch_size = kwargs.get('batch_size', -1)
    lr_gen = kwargs.get('lr_gen', 0.1)
    lr_disc = kwargs.get('lr_disc', lr_gen)
    verbose = kwargs.get('verbose', True)
    regul_param = kwargs.get('regul_param', 0.1)
    dnh = kwargs.get('dnh', None)

    if hasattr(df_data, 'columns'):
        list_nodes = list(df_data.columns)
        data_np = df_data[list_nodes].values.astype('float32')
    else:
        list_nodes = list(range(df_data.shape[1]))
        data_np = df_data.astype('float32')
    data_tensor = torch.from_numpy(data_np)
    if batch_size == -1:
        batch_size = data_tensor.size(0)
    rows, cols = data_tensor.size()

    zero_components = [[i] for i in range(cols)]
    kwargs_for_generators = kwargs.copy()
    kwargs_for_generators.pop("batch_size", None)

    sam = SAM_generators((rows, cols), zero_components, batch_size=batch_size, batch_norm=True, **kwargs_for_generators)
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("activation_function", None)
    discriminator_sam = SAM_discriminator([cols, dnh, dnh, 1], batch_norm=True, activation_function=nn.LeakyReLU,
                                          activation_argument=0.2, **kwargs_copy)

    if gpu:
        sam = sam.cuda(gpu_no)
        discriminator_sam = discriminator_sam.cuda(gpu_no)
        data_tensor = data_tensor.cuda(gpu_no)

    criterion = nn.BCEWithLogitsLoss()
    g_optimizer = optim.Adam(sam.parameters(), lr=lr_gen)
    d_optimizer = optim.Adam(discriminator_sam.parameters(), lr=lr_disc)

    causal_filters = torch.zeros(cols, cols, device=data_tensor.device)
    dataset = TensorDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_epochs = train_epochs + test_epochs
    for epoch in range(total_epochs):
        epoch_adv_loss = 0.0
        epoch_gen_loss = 0.0
        batch_count = 0
        for i_batch, (batch,) in enumerate(data_loader):
            batch_count += 1
            batch_vectors = [batch[:, i:i + 1] for i in range(cols)]
            current_batch_size = batch.size(0)
            true_variable = torch.ones(current_batch_size, 1, device=data_tensor.device)
            false_variable = torch.zeros(current_batch_size, 1, device=data_tensor.device)

            d_optimizer.zero_grad()
            generated_variables = sam(batch)
            disc_losses = []
            for i in range(cols):
                generator_output = torch.cat(batch_vectors[:i] + [generated_variables[i]] + batch_vectors[i + 1:],
                                             dim=1)
                disc_output_detached = discriminator_sam(generator_output.detach())
                disc_loss_fake = criterion(disc_output_detached, false_variable)
                disc_losses.append(disc_loss_fake)
            true_output = discriminator_sam(batch)
            disc_loss_real = criterion(true_output, true_variable)
            adv_loss = (sum(disc_losses) / cols) + disc_loss_real
            adv_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            generated_variables = sam(batch)
            gen_losses = []
            for i in range(cols):
                generator_output = torch.cat(batch_vectors[:i] + [generated_variables[i]] + batch_vectors[i + 1:],
                                             dim=1)
                disc_output = discriminator_sam(generator_output)
                gen_losses.append(criterion(disc_output, true_variable))
            gen_loss = sum(gen_losses)
            filters = torch.stack([abs(block.fs_filter[0, :-1]) for block in sam.blocks], dim=1)
            l1_reg = regul_param * filters.sum()
            loss = gen_loss + l1_reg
            loss.backward()

            if epoch >= train_epochs:
                causal_filters += filters.detach()
            g_optimizer.step()

            epoch_adv_loss += adv_loss.item()
            epoch_gen_loss += gen_loss.item()

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"[SAM Epoch {epoch + 1}/{total_epochs}] Discriminator Loss: {epoch_adv_loss / batch_count:.4f}, Generator Loss: {epoch_gen_loss / batch_count:.4f}")

    causal_filters = causal_filters / test_epochs
    return causal_filters.cpu().numpy()


class SAM(object):
    def __init__(self, lr=0.1, dlr=0.1, l1=0.1, nh=200, dnh=200, train_epochs=100, test_epochs=100, batchsize=-1):
        self.lr = lr
        self.dlr = dlr
        self.l1 = l1
        self.nh = nh
        self.dnh = dnh
        self.train_epochs = train_epochs
        self.test_epochs = test_epochs
        self.batchsize = batchsize

    def predict(self, data, skeleton=None, nruns=6, njobs=1, gpus=0, verbose=True):
        results = Parallel(n_jobs=njobs)(
            delayed(run_SAM)(data, skeleton=skeleton, lr_gen=self.lr, lr_disc=self.dlr,
                             regul_param=self.l1, nh=self.nh, dnh=self.dnh, gpu=bool(gpus),
                             train_epochs=self.train_epochs, test_epochs=self.test_epochs,
                             batch_size=self.batchsize, verbose=verbose, gpu_no=idx % max(gpus, 1))
            for idx in range(nruns)
        )
        W = results[0]
        for w in results[1:]:
            W += w
        W /= nruns
        return W


###########################
# 2. General Utility Functions
###########################
def check_data(data, name="data"):
    print(f"Checking {name} for NaN or Inf values...")
    print(f"Has NaN: {torch.isnan(data).any().item()}")
    print(f"Has Inf: {torch.isinf(data).any().item()}")


def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'beta' in name:
            # [Important Modification] Lower the prior weight initialization to prevent GAT
            # from becoming a static operator and blocking gradients in the early stage
            nn.init.constant_(param, 0.05)
        elif "weight" in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0)


###########################
# 3. CausalGAT-LSTM-IDM Model Definition
###########################
class CausalGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(CausalGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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
        batch_size, N, _ = Wh.size()
        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)

        e_base = self.leakyrelu(torch.matmul(torch.cat([Wh_i, Wh_j], dim=-1), self.a).squeeze(-1))
        e_guided = e_base + self.beta * adj

        # [Important Modification] Add a diagonal self-loop matrix to prevent the loss of self-information
        # during node feature aggregation
        adj_with_self_loop = adj + torch.eye(N, device=adj.device)
        zero_vec = -9e15 * torch.ones_like(e_guided)
        # Use >0 to prevent all-zero masks caused by floating-point precision errors
        attention = torch.where(adj_with_self_loop > 0, e_guided, zero_vec)

        attention = torch.softmax(attention, dim=-1)
        attention = nn.functional.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)
        return nn.functional.elu(h_prime) if self.concat else h_prime


class GAT(nn.Module):
    def __init__(self, num_features, gat_hidden_dim, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.gat1 = CausalGATLayer(in_features=1, out_features=gat_hidden_dim, dropout=dropout, alpha=alpha,
                                   concat=True)

    def forward(self, x, adj):
        out = self.gat1(x, adj)
        return out


class CausalGAT_LSTM_IDM_Model(nn.Module):
    def __init__(self, num_features, gat_hidden_dim, lstm_hidden_dim, num_layers=1, causal_matrix=None):
        super(CausalGAT_LSTM_IDM_Model, self).__init__()
        self.num_features = num_features
        self.gat_hidden_dim = gat_hidden_dim
        self.delta_t = 0.1

        self.gat = GAT(num_features, gat_hidden_dim)

        # [Important Modification] Residual connection: feed the original physical features directly into the LSTM!
        # This ensures the model maintains the exact same baseline performance as the original LSTM-IDM
        # before GAT is fully trained
        self.lstm_input_dim = num_features + (num_features * gat_hidden_dim)
        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(lstm_hidden_dim, 6)
        self.softplus = nn.Softplus()

        if causal_matrix is None:
            self.register_buffer('causal_matrix', torch.zeros(num_features, num_features))
        else:
            self.register_buffer('causal_matrix', causal_matrix)

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()

        x_gat = x.view(batch_size * seq_len, num_features, 1)
        gat_out = self.gat(x_gat, self.causal_matrix)
        gat_out = gat_out.view(batch_size, seq_len, -1)

        # Concatenate the original input x with the features extracted by GAT, then feed to LSTM
        lstm_in = torch.cat([x, gat_out], dim=-1)
        lstm_out, _ = self.lstm(lstm_in)

        last_hidden = lstm_out[:, -1, :]
        params = self.fc(last_hidden)
        params = self.softplus(params)
        return params

    def predict_speed(self, x, s_safe):
        params = self.forward(x)
        v_n = x[:, -1, 0]
        delta_v = x[:, -1, 2]
        v_desired, T, a_max, b_safe, delta, s0 = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:,
                                                                                                         4], params[:,
                                                                                                             5]

        s_star = s0 + v_n * T + (v_n * -delta_v) / (2 * torch.sqrt(a_max * b_safe) + 1e-6)
        s_star = torch.clamp(s_star, min=0)
        v_follow = v_n + self.delta_t * a_max * (
                    1 - (v_n / (v_desired + 1e-6)) ** delta - (s_star / (s_safe + 1e-6)) ** 2)
        predicted_speed = torch.clamp(v_follow, min=0)

        return predicted_speed.unsqueeze(1), params


###########################
# 4. Liquid Neural Network (LNN) and Fusion Module Definition
###########################
class LiquidCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        super(LiquidCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.activation = nn.Tanh()

    def forward(self, u, h):
        dh = -h + self.activation(self.W_h(h) + self.W_u(u) + self.bias)
        h_new = h + self.dt * dh
        return h_new


class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_steps=50, output_dim=1):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.liquid_cells = nn.ModuleList([
            LiquidCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        T = min(seq_len, self.num_steps)
        h = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        for t in range(T):
            input_t = x[:, t, :]
            for i, cell in enumerate(self.liquid_cells):
                if i == 0:
                    h[i] = cell(input_t, h[i])
                else:
                    h[i] = cell(h[i - 1], h[i])
        out = self.fc(h[-1])
        return out

    def predict_speed(self, x):
        return self.forward(x)


class FusionModule(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, num_layers=1):
        super(FusionModule, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        lambda_val = self.sigmoid(self.fc(h_last))
        return lambda_val


###########################
# 5. Training and Evaluation Functions
###########################
def train_idm_model(model, train_loader, optimizer, criterion, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed, batch_s_safe in train_loader:
            optimizer.zero_grad()
            predicted_speed, _ = model.predict_speed(batch_data.to(device), batch_s_safe.to(device))
            loss = criterion(predicted_speed, batch_speed.to(device).unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[SAM-CausalGAT-LSTM-IDM] Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    return model


def train_lnn_model(model, train_loader, optimizer, criterion, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_speed in train_loader:
            optimizer.zero_grad()
            predicted_speed = model.predict_speed(batch_data.to(device))
            loss = criterion(predicted_speed, batch_speed.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[LNN] Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
    return model


def train_fusion_module(fusion_module, idm_model, lnn_model, fusion_loader, optimizer, criterion, num_epochs=20):
    idm_model.eval()
    lnn_model.eval()
    fusion_module.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in fusion_loader:
            fusion_input, idm_input, lnn_input, ground_truth, s_safe = batch
            fusion_input = fusion_input.to(device)
            idm_input = idm_input.to(device)
            lnn_input = lnn_input.to(device)
            ground_truth = ground_truth.to(device).unsqueeze(1)
            s_safe = s_safe.to(device)

            with torch.no_grad():
                y_ph, _ = idm_model.predict_speed(idm_input, s_safe)
                y_da = lnn_model.predict_speed(lnn_input)

            lambda_val = fusion_module(fusion_input)
            fused_output = lambda_val * y_da + (1 - lambda_val) * y_ph

            loss = criterion(fused_output, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Fusion] Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(fusion_loader):.4f}")
    return fusion_module


def evaluate_model(model, test_loader, model_type="IDM", idm_model=None, lnn_model=None):
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    all_predicted = []
    all_true = []
    with torch.no_grad():
        for batch in test_loader:
            if model_type == "IDM":
                batch_data, batch_speed, batch_s_safe = batch
                predicted_speed, _ = model.predict_speed(batch_data.to(device), batch_s_safe.to(device))
                gt = batch_speed.to(device).unsqueeze(1)
            elif model_type == "LNN":
                batch_data, batch_speed = batch
                predicted_speed = model.predict_speed(batch_data.to(device))
                gt = batch_speed.to(device)
            elif model_type == "Fusion":
                fusion_input, idm_input, lnn_input, gt, s_safe = batch
                fusion_input, idm_input, lnn_input = fusion_input.to(device), idm_input.to(device), lnn_input.to(device)
                gt, s_safe = gt.to(device).unsqueeze(1), s_safe.to(device)

                lambda_val = model(fusion_input)
                y_ph, _ = idm_model.predict_speed(idm_input, s_safe)
                y_da = lnn_model.predict_speed(lnn_input)
                predicted_speed = lambda_val * y_da + (1 - lambda_val) * y_ph

            loss = mse_loss(predicted_speed, gt)
            total_mse += loss.item()
            all_predicted.append(predicted_speed.cpu())
            all_true.append(gt.cpu())

    mse = total_mse / len(test_loader)
    rmse = torch.sqrt(torch.tensor(mse))
    mae = torch.mean(torch.abs(torch.cat(all_predicted) - torch.cat(all_true))).item()
    print(f"Evaluation Metrics ({model_type}):\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")


def compute_position_and_spacing_and_save(
        fusion_module, idm_model, lnn_model, fusion_input, idm_input, lnn_input, s_safe,
        raw_data, label_data, train_size, dt=0.1, output_file="fusion_predictions.xlsx"):
    fusion_module.eval()
    idm_model.eval()
    lnn_model.eval()

    with torch.no_grad():
        y_ph, _ = idm_model.predict_speed(idm_input.to(device), s_safe.to(device))
        y_da = lnn_model.predict_speed(lnn_input.to(device))
        lambda_val = fusion_module(fusion_input.to(device))
        pred_speed = (lambda_val * y_da + (1 - lambda_val) * y_ph).squeeze().cpu().numpy()

    N_test = fusion_input.shape[0]
    idx = np.arange(train_size, train_size + N_test)

    current_Y_ft = raw_data[idx, -1, 4].numpy()
    current_speed_m = fusion_input[:, -1, 0].numpy()
    true_Y_ft = label_data[idx, -1, 3].numpy()
    true_spacing_m = label_data[idx, -1, 1].numpy() * 0.3048

    disp_m = current_speed_m * dt + 0.5 * ((pred_speed - current_speed_m) / dt) * dt ** 2
    disp_ft = disp_m / 0.3048
    pred_Y_ft = current_Y_ft + disp_ft

    pred_Y_m = pred_Y_ft * 0.3048
    true_Y_m = true_Y_ft * 0.3048
    pred_spacing_m = true_Y_m - pred_Y_m + true_spacing_m

    rmse_Y = np.sqrt(np.mean((pred_Y_m - true_Y_m) ** 2))
    mape_Y = np.mean(np.abs((pred_Y_m - true_Y_m) / true_Y_m)) * 100
    rmse_sp = np.sqrt(np.mean((pred_spacing_m - true_spacing_m) ** 2))
    mape_sp = np.mean(np.abs((pred_spacing_m - true_spacing_m) / true_spacing_m)) * 100

    print(f"[Fusion] Position Error -- RMSE: {rmse_Y:.4f} m, MAPE: {mape_Y:.2f}%")
    print(f"[Fusion] Spacing  Error -- RMSE: {rmse_sp:.4f} m, MAPE: {mape_sp:.2f}%")

    df = pd.DataFrame({"Pred Speed (m/s)": pred_speed, "Predicted Y (m)": pred_Y_m, "True Y (m)": true_Y_m,
                       "Predicted Spacing (m)": pred_spacing_m, "True Spacing (m)": true_spacing_m})
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="a" if os.path.exists(output_file) else "w") as writer:
        df.to_excel(writer, sheet_name="Fusion", index=False)


###########################
# 6. Main Function
###########################
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    ################### Data Loading and Preprocessing ###################
    try:
        data = sio.loadmat('E:\\pythonProject1\\data_fine_0.1.mat')
    except FileNotFoundError:
        print("Error: Data file 'E:\\pythonProject1\\data_fine_0.1.mat' not found.")
        exit()

    raw_data = torch.tensor(data['train_data'], dtype=torch.float32)
    lable_data = torch.tensor(data['lable_data'], dtype=torch.float32)

    data_idm = raw_data[:, -50:, [0, 1, 2, 3, -1]].clone()
    data_lnn = raw_data[:, -50:, [0, 1, 2, 3, -1]].clone()
    fusion_data = raw_data[:, -5:, :].clone()
    fusion_data = fusion_data[:, :, [0, -1]]

    for tensor in [data_idm, data_lnn, fusion_data]:
        tensor *= 0.3048

    ground_truth = lable_data[:, -1, 0].clone() * 0.3048
    s_safe = data_idm[:, -1, 1].clone()

    # Take the first 10% for quick demonstration; revert to the full dataset as needed
    total_samples = data_idm.shape[0]
    sample_size = int(total_samples * 1)

    data_idm = data_idm[:sample_size]
    data_lnn = data_lnn[:sample_size]
    fusion_data = fusion_data[:sample_size]
    ground_truth = ground_truth[:sample_size]
    s_safe = s_safe[:sample_size]

    dataset_size = data_idm.shape[0]
    train_size = int(dataset_size * 0.8)

    train_idm_data = data_idm[:train_size]
    test_idm_data = data_idm[train_size:]
    train_idm_gt = ground_truth[:train_size]
    test_idm_gt = ground_truth[train_size:]
    train_s_safe = s_safe[:train_size]
    test_s_safe = s_safe[train_size:]

    train_lnn_data = data_lnn[:train_size]
    test_lnn_data = data_lnn[train_size:]
    train_lnn_gt = ground_truth[:train_size].unsqueeze(1)
    test_lnn_gt = ground_truth[train_size:].unsqueeze(1)

    train_fusion_input = fusion_data[:train_size]
    test_fusion_input = fusion_data[train_size:]

    batch_size = 32
    idm_train_loader = DataLoader(TensorDataset(train_idm_data, train_idm_gt, train_s_safe), batch_size=batch_size,
                                  shuffle=True)
    idm_test_loader = DataLoader(TensorDataset(test_idm_data, test_idm_gt, test_s_safe), batch_size=batch_size,
                                 shuffle=False)
    lnn_train_loader = DataLoader(TensorDataset(train_lnn_data, train_lnn_gt), batch_size=batch_size, shuffle=True)
    lnn_test_loader = DataLoader(TensorDataset(test_lnn_data, test_lnn_gt), batch_size=batch_size, shuffle=False)
    fusion_train_loader = DataLoader(
        TensorDataset(train_fusion_input, train_idm_data, train_lnn_data, train_idm_gt, train_s_safe),
        batch_size=batch_size, shuffle=True)
    fusion_test_loader = DataLoader(
        TensorDataset(test_fusion_input, test_idm_data, test_lnn_data, test_idm_gt, test_s_safe), batch_size=batch_size,
        shuffle=False)

    #########################################################
    # SAM dynamically generates the true causal matrix
    #########################################################
    print("\n--- Starting SAM model for causal discovery ---")
    sam_input = train_idm_data[:, -1, :].cpu().numpy()

    sam_model = SAM(lr=0.05, dlr=0.05, l1=0.1, nh=100, dnh=100, train_epochs=100, test_epochs=100, batchsize=64)
    causal_matrix_np = sam_model.predict(sam_input, nruns=1, verbose=False)

    # Min-Max normalize the causal matrix
    A_causal = torch.tensor(causal_matrix_np, dtype=torch.float32)
    min_val, max_val = torch.min(A_causal), torch.max(A_causal)
    if max_val > min_val:
        A_causal_normalized = (A_causal - min_val) / (max_val - min_val)
    else:
        A_causal_normalized = A_causal

    print("Normalized causal relationship matrix (as prior input to CausalGAT-LSTM-IDM):")
    print(A_causal_normalized.numpy())

    #########################################################
    # Train the SAM-CausalGAT-LSTM-IDM model
    #########################################################
    input_dim_idm = train_idm_data.shape[2]
    gat_hidden_dim = 16
    hidden_dim_idm = 128
    num_layers_idm = 1

    idm_model = CausalGAT_LSTM_IDM_Model(
        num_features=input_dim_idm,
        gat_hidden_dim=gat_hidden_dim,
        lstm_hidden_dim=hidden_dim_idm,
        num_layers=num_layers_idm,
        causal_matrix=A_causal_normalized.to(device)
    ).to(device)

    initialize_weights(idm_model)
    criterion_idm = nn.MSELoss()
    optimizer_idm = optim.Adam(idm_model.parameters(), lr=0.0005)
    num_epochs_idm = 100

    print("\n--- Training SAM-CausalGAT-LSTM-IDM Model ---")
    idm_model = train_idm_model(idm_model, idm_train_loader, optimizer_idm, criterion_idm, num_epochs=num_epochs_idm)
    print("\n--- Evaluating SAM-CausalGAT-LSTM-IDM Model ---")
    evaluate_model(idm_model, idm_test_loader, model_type="IDM")

    #########################################################
    # Train the LNN model
    #########################################################
    input_dim_lnn = train_lnn_data.shape[2]
    hidden_dim_lnn = 128
    num_layers_lnn = 1
    num_steps = train_lnn_data.shape[1]

    lnn_model = LiquidNeuralNetwork(input_dim_lnn, hidden_dim_lnn, num_layers=num_layers_lnn, num_steps=num_steps,
                                    output_dim=1).to(device)
    initialize_weights(lnn_model)
    criterion_lnn = nn.MSELoss()
    optimizer_lnn = optim.Adam(lnn_model.parameters(), lr=0.0005)
    num_epochs_lnn = 150

    print("\n--- Training Liquid Neural Network (LNN) ---")
    lnn_model = train_lnn_model(lnn_model, lnn_train_loader, optimizer_lnn, criterion_lnn, num_epochs=num_epochs_lnn)
    print("\n--- Evaluating LNN Model ---")
    evaluate_model(lnn_model, lnn_test_loader, model_type="LNN")

    #########################################################
    # Train the Fusion module
    #########################################################
    fusion_module = FusionModule(input_dim=2, hidden_dim=32, num_layers=1).to(device)
    initialize_weights(fusion_module)
    criterion_fusion = nn.MSELoss()
    optimizer_fusion = optim.Adam(fusion_module.parameters(), lr=0.001)
    num_epochs_fusion = 30

    print("\n--- Training Fusion Module ---")
    fusion_module = train_fusion_module(fusion_module, idm_model, lnn_model, fusion_train_loader, optimizer_fusion,
                                        criterion_fusion, num_epochs=num_epochs_fusion)

    print("\n--- Evaluating Fusion Model ---")
    evaluate_model(fusion_module, fusion_test_loader, model_type="Fusion", idm_model=idm_model, lnn_model=lnn_model)

    # Calculate and save prediction results
    compute_position_and_spacing_and_save(
        fusion_module=fusion_module,
        idm_model=idm_model,
        lnn_model=lnn_model,
        fusion_input=test_fusion_input,
        idm_input=test_idm_data[train_size:],
        lnn_input=test_lnn_data[train_size:],
        s_safe=test_s_safe,
        raw_data=raw_data,
        label_data=lable_data,
        train_size=train_size,
        dt=0.1,
        output_file="SAMmymodel_1.xlsx"
    )

import torch
import torch.nn as nn

# =========================
# 1. Hybrid IDM 模型
# =========================
class HybridIDMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dt=0.1):
        super(HybridIDMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 6)
        self.softplus = nn.Softplus()
        self.delta_t = dt

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        params_raw = self.fc(out[:, -1, :])
        params_activated = self.softplus(params_raw)
        return params_activated

    def predict_speed(self, x, s_actual):
        params = self.forward(x)
        v_n = x[:, -1, 0]
        delta_v_hist = x[:, -1, 2]

        v_des_raw, T_raw, a_max_raw, b_safe_raw, delta_idm_raw, s0_raw = \
            params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4], params[:, 5]

        v_des = torch.clamp(v_des_raw, min=0.1, max=50.0)
        T = torch.clamp(T_raw, min=0.1, max=5.0)
        a_max = torch.clamp(a_max_raw, min=0.1, max=5.0)
        b_safe = torch.clamp(b_safe_raw, min=0.1, max=9.0)
        delta_idm = torch.clamp(delta_idm_raw, min=1.0, max=10.0)
        s0 = torch.clamp(s0_raw, min=0.0, max=10.0)
        s_actual_clamped = torch.clamp(s_actual, min=0.5)

        sqrt_ab_clamped = torch.clamp(torch.sqrt(a_max * b_safe), min=1e-6)
        interaction_term = (v_n * (-delta_v_hist)) / (2 * sqrt_ab_clamped + 1e-9)
        s_star = s0 + torch.clamp(v_n * T, min=0.0) + interaction_term
        s_star = torch.clamp(s_star, min=s0)

        v_n_clamped = torch.clamp(v_n, min=0.0)
        speed_ratio = (v_n_clamped + 1e-6) / (v_des + 1e-6)
        term_speed_ratio = speed_ratio.pow(delta_idm)
        spacing_ratio = s_star / (s_actual_clamped + 1e-6)
        term_spacing_ratio = spacing_ratio.pow(2)

        accel_component = 1.0 - term_speed_ratio - term_spacing_ratio
        a_idm_val = a_max * accel_component

        v_follow = v_n + a_idm_val * self.delta_t
        v_follow = torch.clamp(v_follow, min=0.0, max=60.0)

        if torch.isnan(v_follow).any() or torch.isinf(v_follow).any():
            print("警告: HybridIDMModel.predict_speed 中检测到 NaN/Inf 输出。")
        return v_follow.unsqueeze(1), params


# =========================
# 2. LNN 模型组件
# =========================
class LiquidCellMulti(nn.Module):
    def __init__(self, input_dim, hidden_dim, dt=0.1):
        super(LiquidCellMulti, self).__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.act = nn.Tanh()

    def forward(self, u, h):
        if h.shape[-1] != self.hidden_dim:
            h = torch.zeros(u.shape[0], self.hidden_dim, device=u.device)
        dh = -h + self.act(self.W_h(h) + self.W_u(u) + self.bias)
        return h + self.dt * dh


class LiquidNeuralNetworkMultiStep(nn.Module):  # 用于前车预测
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1, num_steps=50, dt=0.1):
        super(LiquidNeuralNetworkMultiStep, self).__init__()
        self.input_dim = input_dim
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(LiquidCellMulti(current_input_dim, hidden_dim, dt=dt))
        self.fc = nn.Linear(hidden_dim, prediction_steps)
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch, seq, features = x.shape
        h_states = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        effective_seq_len = min(seq, self.num_steps)

        for t in range(effective_seq_len):
            u_t_layer = x[:, t, :]
            for i in range(self.num_layers):
                input_signal_for_cell = h_states[i - 1] if i > 0 else u_t_layer
                h_states[i] = self.cells[i](input_signal_for_cell, h_states[i])
        return self.fc(h_states[-1])

    def predict_speed(self, x):
        return self.forward(x)


class LiquidNeuralNetworkMultiStepEgo(nn.Module):  # 用于自车预测
    def __init__(self, input_dim, hidden_dim, prediction_steps, num_layers=1, num_steps=50, dt=0.1):
        super(LiquidNeuralNetworkMultiStepEgo, self).__init__()
        self.input_dim = input_dim
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            current_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(LiquidCellMulti(current_input_dim, hidden_dim, dt=dt))
        self.fc = nn.Linear(hidden_dim, prediction_steps)
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch, seq, features = x.shape
        h_states = [torch.zeros(batch, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        effective_seq_len = min(seq, self.num_steps)

        for t in range(effective_seq_len):
            u_t_layer = x[:, t, :]
            for i in range(self.num_layers):
                input_signal_for_cell = h_states[i - 1] if i > 0 else u_t_layer
                h_states[i] = self.cells[i](input_signal_for_cell, h_states[i])
        return self.fc(h_states[-1])

    def predict_speed(self, x):
        return self.forward(x)


# =========================
# 3. 融合 LSTM 模型
# =========================
class FusionLSTMModel(nn.Module):
    def __init__(self, fusion_input_dim, fusion_hidden_dim, fusion_output_steps, fusion_num_layers=1):
        super(FusionLSTMModel, self).__init__()
        self.lstm = nn.LSTM(fusion_input_dim, fusion_hidden_dim, fusion_num_layers, batch_first=True)
        self.fc = nn.Linear(fusion_hidden_dim, fusion_output_steps)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_ego_hist_speed, x_leader_hist_speed):
        fusion_input = torch.cat((x_ego_hist_speed, x_leader_hist_speed), dim=2)
        lstm_out, _ = self.lstm(fusion_input)
        alpha_raw = self.fc(lstm_out[:, -1, :])
        alpha = self.sigmoid(alpha_raw)
        return alpha
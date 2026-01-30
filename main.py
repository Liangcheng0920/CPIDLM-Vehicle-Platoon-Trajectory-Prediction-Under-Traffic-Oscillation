import os
import torch
import torch.optim as optim
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from src.utils import device, initialize_weights
from src.models import HybridIDMModel, LiquidNeuralNetworkMultiStep, LiquidNeuralNetworkMultiStepEgo, FusionLSTMModel
from src.engine import (
    train_generic_model,
    evaluate_generic_model,
    precompute_leader_trajectories_for_idm_training,
    train_idm_model_multistep,
    compute_multi_step_fusion_predictions,
    predict_multi_step_idm_for_fusion_training
)

# =========================
# 配置参数
# =========================
DATA_PATH = r'data/data_10.mat'  # 请确保文件在此路径下
PRED_STEPS_K = 10
DT = 0.1
HIST_LEN = 50
FUSION_HIST_LEN = 20
BATCH_SIZE = 32

# 训练轮数设置
EPOCHS_DICT = {
    "leader_lnn": 50,  # 示例值，可调大
    "idm": 50,
    "lnn_ego": 50,
    "fusion": 30
}


def load_and_process_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件未找到: {path}")

    data = sio.loadmat(path)
    label_key = 'lable_data' if 'lable_data' in data else 'label_data'

    raw_full = torch.tensor(data['train_data'], dtype=torch.float32)
    label_full = torch.tensor(data[label_key], dtype=torch.float32)

    # 取30%数据
    limit = int(raw_full.shape[0] * 0.3)
    raw = raw_full[:limit]
    label = label_full[:limit]

    split = int(raw.shape[0] * 0.8)
    return raw[:split], label[:split], raw[split:], label[split:]


def main():
    torch.manual_seed(42)
    print(f"使用设备: {device}")

    # 1. 加载数据
    try:
        raw_train, label_train, raw_test, label_test = load_and_process_data(DATA_PATH)
        print(f"训练集样本: {len(raw_train)}, 测试集样本: {len(raw_test)}")
    except Exception as e:
        print(e)
        return

    # 2. 训练 Leader LNN
    print("\n--- Training Leader LNN ---")
    leader_in = raw_train[:, -HIST_LEN:, [5, 6]].clone() * 0.3048
    leader_tgt = label_train[:, :PRED_STEPS_K, 4].clone() * 0.3048
    leader_dataset = torch.utils.data.TensorDataset(leader_in, leader_tgt)
    leader_loader = torch.utils.data.DataLoader(leader_dataset, batch_size=BATCH_SIZE, shuffle=True)

    leader_model = LiquidNeuralNetworkMultiStep(2, 64, PRED_STEPS_K, 1, HIST_LEN, DT).to(device)
    initialize_weights(leader_model)
    train_generic_model(leader_model, leader_loader, optim.Adam(leader_model.parameters(), lr=1e-3),
                        nn.MSELoss(), EPOCHS_DICT["leader_lnn"], "Leader LNN")

    # 3. 训练 Hybrid IDM
    print("\n--- Training Hybrid IDM ---")
    (idm_in, f_pos, s_safe, l_speeds, l_pos, d1) = precompute_leader_trajectories_for_idm_training(
        leader_model, raw_train, PRED_STEPS_K, DT, device, HIST_LEN
    )
    # 构造 IDM 训练所需的所有特征
    true_v_f = label_train[:, :PRED_STEPS_K, 0] * 0.3048
    true_all_feats = torch.stack([
        label_train[:, :PRED_STEPS_K, 0],  # v_f
        label_train[:, :PRED_STEPS_K, 1],  # s
        label_train[:, :PRED_STEPS_K, 4] - label_train[:, :PRED_STEPS_K, 0],  # dv
        label_train[:, :PRED_STEPS_K, 2]  # a_f
    ], dim=2) * 0.3048
    true_f_pos = label_train[:, :PRED_STEPS_K, 3] * 0.3048

    idm_dataset = torch.utils.data.TensorDataset(idm_in, true_v_f.to(device), f_pos, s_safe,
                                                 l_speeds, l_pos, d1, true_all_feats.to(device), true_f_pos.to(device))
    idm_loader = torch.utils.data.DataLoader(idm_dataset, batch_size=BATCH_SIZE, shuffle=True)

    idm_model = HybridIDMModel(5, 64, 1, DT).to(device)
    initialize_weights(idm_model)
    train_idm_model_multistep(idm_model, idm_loader, optim.Adam(idm_model.parameters(), lr=2e-4),
                              EPOCHS_DICT["idm"], PRED_STEPS_K, DT)

    # 4. 训练 Ego LNN
    print("\n--- Training Ego LNN ---")
    ego_in = raw_train[:, -HIST_LEN:, [0, 1, 2, 3, 5]].clone() * 0.3048
    ego_tgt = label_train[:, :PRED_STEPS_K, 0].clone() * 0.3048
    ego_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(ego_in, ego_tgt), batch_size=BATCH_SIZE,
                                             shuffle=True)

    lnn_ego_model = LiquidNeuralNetworkMultiStepEgo(5, 64, PRED_STEPS_K, 1, HIST_LEN, DT).to(device)
    initialize_weights(lnn_ego_model)
    train_generic_model(lnn_ego_model, ego_loader, optim.Adam(lnn_ego_model.parameters(), lr=1e-3),
                        nn.MSELoss(), EPOCHS_DICT["lnn_ego"], "Ego LNN")

    # 5. 训练 Fusion Model
    print("\n--- Training Fusion Model ---")
    fusion_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(raw_train, label_train),
                                                batch_size=BATCH_SIZE, shuffle=True)
    fusion_model = FusionLSTMModel(2, 32, PRED_STEPS_K, 1).to(device)
    initialize_weights(fusion_model)

    train_generic_model(
        fusion_model, fusion_loader, optim.Adam(fusion_model.parameters(), lr=1e-3), nn.MSELoss(),
        EPOCHS_DICT["fusion"], "Fusion Model", is_fusion_model=True,
        idm_model_frozen=idm_model, lnn_ego_model_frozen=lnn_ego_model, leader_model_frozen=leader_model,
        FUSION_HIST_LEN=FUSION_HIST_LEN, PRED_STEPS_K_fusion=PRED_STEPS_K, DT_fusion=DT,
        HIST_LEN_idm_ego_fusion=HIST_LEN, device_fusion=device
    )

    # 6. 最终评估
    print("\n--- Final Evaluation ---")
    compute_multi_step_fusion_predictions(
        idm_model, leader_model, lnn_ego_model, fusion_model,
        raw_test, label_test, DT, PRED_STEPS_K, HIST_LEN, FUSION_HIST_LEN, device
    )

    plt.show()


if __name__ == "__main__":
    main()
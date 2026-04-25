import pandas as pd
import numpy as np
import os
from scipy.io import savemat

# Configuration
save_path = r'E:\matlab_mode_data\traffic_flow\NGSIM'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Observation window length
sequence_length = 50

# Loop through L (prediction window setup)
for L in [1]:
    T = L - 1  # Prediction horizon offset

    # Initialize lists to store training segments
    train_data_list = []
    label_data_list = []

    # Get unique vehicle IDs from the valid_data DataFrame
    unique_ids = valid_data['Vehicle_ID'].unique()

    for veh_id in unique_ids:
        # Filter data for the current vehicle
        current_data = valid_data[valid_data['Vehicle_ID'] == veh_id].copy()
        current_data = current_data.reset_index(drop=True)

        # Skip if the data segment is shorter than the required window
        if len(current_data) < (sequence_length + 1 + T):
            continue

        # Data cleaning: Remove segments with anomalous negative following distance
        if (current_data['distance_fellow'] < 0).any():
            continue

        # Sliding window to generate time-series sequences
        # In Python, the range is (total_length - sequence_length - T)
        for j in range(len(current_data) - sequence_length - T):
            # 1. Extract Input Sequence (Observation Window)
            # Slice from j to j+sequence_length (exclusive of the end index)
            sequence = current_data.iloc[j: j + sequence_length]

            # Construct feature matrix [Sequence_Length, 8]
            input_features = sequence[[
                'v_Vel', 'distance_fellow', 'diff_v', 'v_Acc',
                'Local_Y', 'Prec_speed', 'Prec_acc', 'Prec_Local_y'
            ]].values

            # 2. Extract Label Features (Prediction Window)
            # Slice from the end of sequence to sequence + prediction window
            pred_start = j + sequence_length
            pred_end = j + sequence_length + T + 1
            future_data = current_data.iloc[pred_start: pred_end]

            # Construct label matrix [1+T, 6]
            lab_features = future_data[[
                'v_Vel', 'distance_fellow', 'v_Acc',
                'Local_Y', 'Prec_speed', 'Prec_Local_y'
            ]].values

            # Append to lists (equivalent to cat in MATLAB)
            train_data_list.append(input_features)
            label_data_list.append(lab_features)

        # Check sample limit (approx. 40,000 samples) to prevent memory issues
        if len(train_data_list) > 40000:
            break

    # Convert lists to NumPy arrays
    # Resulting shapes:
    # train_data: (Samples, Sequence_Length, 8)
    # label_data: (Samples, 1+T, 6)
    train_data = np.array(train_data_list)
    label_data = np.array(label_data_list)

    # Define filename and full path
    filename = f'data_{T}.mat'
    full_filename = os.path.join(save_path, filename)

    # Save to .mat format (compatible with MATLAB)
    savemat(full_filename, {
        'train_data': train_data,
        'lable_data': label_data  # Kept 'lable' spelling to match your MATLAB code
    })

    print(f"Saved: {full_filename} with {train_data.shape[0]} samples.")

    # Cleanup local loop variables to free memory
    del train_data_list, label_data_list, train_data, label_data
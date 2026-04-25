import pandas as pd
import os

# --- 1. Data Loading Section ---
# Set the directory path where your file is located
file_path = r'F:\new_traffic\EA\Trajectories\trajectories05000515.csv'

# Check if the file exists before reading
if os.path.exists(file_path):
    # Read the file and assign it to 'data'
    # Use pd.read_excel(file_path) if it's an Excel file
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
else:
    print(f"Error: The file at {file_path} was not found.")
    # Exit or handle error
    data = pd.DataFrame()

# --- 2. Parameters ---
vehicle_ids = data['Vehicle_ID'].unique() if not data.empty else []
space_headway_threshold = 65
min_duration_ms = 10 * 1000

# --- 3. Processing Logic ---
valid_data_list = []

for current_vehicle_id in vehicle_ids:
    # Filter for the current vehicle with a valid preceding car
    current_vehicle_data = data[(data['Vehicle_ID'] == current_vehicle_id) &
                                (data['Preceding'] > 0)].sort_values('Global_Time')

    if current_vehicle_data.empty:
        continue

    continuous_segment = []

    for _, following_row in current_vehicle_data.iterrows():
        current_time = following_row['Global_Time']
        preceding_id = following_row['Preceding']
        space_headway = following_row['Space_Headway']

        # Check car-following condition
        if space_headway <= space_headway_threshold:
            # Find preceding vehicle's status at the same time
            prec_data = data[(data['Vehicle_ID'] == preceding_id) &
                             (data['Global_Time'] == current_time)]

            if not prec_data.empty:
                prec_row = prec_data.iloc[0]
                row_dict = following_row.to_dict()

                # Merge preceding vehicle info
                row_dict.update({
                    'Prec_speed': prec_row['v_Vel'],
                    'Prec_acc': prec_row['v_Acc'],
                    'Prec_Local_x': prec_row['Local_X'],
                    'Prec_Local_y': prec_row['Local_Y'],
                    'Prec_v_Length': prec_row['v_Length'],
                    'Prec_v_Width': prec_row['v_Width']
                })
                continuous_segment.append(row_dict)
            else:
                # Break segment if preceding data is missing
                if len(continuous_segment) > 0:
                    duration = continuous_segment[-1]['Global_Time'] - continuous_segment[0]['Global_Time']
                    if duration >= min_duration_ms:
                        valid_data_list.extend(continuous_segment)
                continuous_segment = []
        else:
            # Break segment if distance threshold exceeded
            if len(continuous_segment) > 0:
                duration = continuous_segment[-1]['Global_Time'] - continuous_segment[0]['Global_Time']
                if duration >= min_duration_ms:
                    valid_data_list.extend(continuous_segment)
            continuous_segment = []

    # Final check for the last segment of the current vehicle
    if len(continuous_segment) > 0:
        duration = continuous_segment[-1]['Global_Time'] - continuous_segment[0]['Global_Time']
        if duration >= min_duration_ms:
            valid_data_list.extend(continuous_segment)

# Convert list to final DataFrame
valid_data = pd.DataFrame(valid_data_list)
import pandas as pd
import numpy as np
import os
import glob

# Configuration and Paths
base_path = r'F:\new_traffic\EA\Trajectories'
# Get folder names (excluding the first two directory entries if necessary)
# In Python, os.listdir typically handles this more cleanly than MATLAB's dir()
date_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# Parameters for convoy detection
distance_threshold = 65  # meters
min_vehicle_count = 3  # vehicles
max_time_gap = 150  # frames/seconds (depending on your frame rate)

# Final results storage
dynamic_convoy_EA_8 = []

# Iterate through each folder (iii loop)
for folder_name in date_folders:
    file_path = os.path.join(base_path, folder_name)

    # Setup import options (CSV reading)
    # Using specific column names as defined in your MATLAB code
    col_names = [
        "frameNum", "carId", "carCenterX", "carCenterY", "headX", "headY", "tailX", "tailY",
        "boundingBox1X", "boundingBox1Y", "boundingBox2X", "boundingBox2Y", "boundingBox3X", "boundingBox3Y",
        "boundingBox4X", "boundingBox4Y", "carCenterXft", "carCenterYft", "headXft", "headYft",
        "tailXft", "tailYft", "boundingBox1Xft", "boundingBox1Yft", "boundingBox2Xft", "boundingBox2Yft",
        "boundingBox3Xft", "boundingBox3Yft", "boundingBox4Xft", "boundingBox4Yft", "speed", "heading", "course",
        "laneId"
    ]

    try:
        # Assuming the data is in a CSV file within the folder
        # Adjust pattern if the filename varies
        csv_files = glob.glob(os.path.join(file_path, "*.csv"))
        if not csv_files:
            continue

        data = pd.read_csv(csv_files[0], names=col_names, skiprows=1)

        # Extract Lane 8 data
        lane_8_data = data[data['laneId'] == 8].copy()

        # Sort by frame and position
        lane_8_data.sort_values(by=['frameNum', 'carCenterX'], inplace=True)

        # Initialize dynamic tracking
        dynamic_convoy_segments = []
        current_convoys = {}  # Using a dictionary to track active convoys
        convoy_counter = 0

        unique_times = lane_8_data['frameNum'].unique()

        # Iterate through time steps
        for current_time in unique_times:
            # Get data for current frame
            frame_data = lane_8_data[lane_8_data['frameNum'] == current_time].sort_values('carCenterXft')
            positions = frame_data['carCenterXft'].values
            vehicles = frame_data['carId'].values

            convoys_at_t = []
            temp_convoy = []

            # Spatial clustering: identify vehicle groups in the same lane at current time
            if len(positions) > 1:
                for i in range(len(positions) - 1):
                    if abs(positions[i + 1] - positions[i]) <= distance_threshold:
                        if not temp_convoy:
                            temp_convoy = [vehicles[i]]
                        temp_convoy.append(vehicles[i + 1])
                    else:
                        if len(temp_convoy) >= min_vehicle_count:
                            convoys_at_t.append(set(temp_convoy))
                        temp_convoy = []
                # Check last group
                if len(temp_convoy) >= min_vehicle_count:
                    convoys_at_t.append(set(temp_convoy))

            # Temporal association: Link current clusters to historical convoys
            for current_cluster in convoys_at_t:
                found_match = False
                for c_id, conv_info in current_convoys.items():
                    # Check if convoy is within time gap and shares at least one vehicle
                    if (current_time - conv_info['end_time'] <= max_time_gap and
                            not current_cluster.isdisjoint(conv_info['vehicles'])):
                        # Update existing convoy
                        conv_info['vehicles'].update(current_cluster)
                        conv_info['end_time'] = current_time
                        found_match = True
                        break

                if not found_match:
                    # Create new convoy entry
                    convoy_counter += 1
                    current_convoys[f"convoy_{convoy_counter}"] = {
                        'vehicles': current_cluster,
                        'start_time': current_time,
                        'end_time': current_time
                    }

            # Clean up expired convoys and move to final segments
            to_remove = []
            for c_id, conv_info in current_convoys.items():
                if current_time - conv_info['end_time'] > max_time_gap:
                    if len(conv_info['vehicles']) >= min_vehicle_count:
                        dynamic_convoy_segments.append({
                            'start_time': conv_info['start_time'],
                            'end_time': conv_info['end_time'],
                            'vehicle_count': len(conv_info['vehicles']),
                            'vehicle_ids': list(conv_info['vehicles'])
                        })
                    to_remove.append(c_id)

            for r_id in to_remove:
                del current_convoys[r_id]

        # Add remaining active convoys to results before switching folders
        for c_id, conv_info in current_convoys.items():
            if len(conv_info['vehicles']) >= min_vehicle_count:
                dynamic_convoy_segments.append({
                    'start_time': conv_info['start_time'],
                    'end_time': conv_info['end_time'],
                    'vehicle_count': len(conv_info['vehicles']),
                    'vehicle_ids': list(conv_info['vehicles'])
                })

        # Final storage for this folder
        dynamic_convoy_EA_8.append([folder_name, dynamic_convoy_segments])
        print(f"Processed folder: {folder_name}, Convoys found: {len(dynamic_convoy_segments)}")

    except Exception as e:
        print(f"Error processing {folder_name}: {e}")

# Display results
# for item in dynamic_convoy_EA_8:
#    print(f"Folder: {item[0]}, Results: {item[1]}")
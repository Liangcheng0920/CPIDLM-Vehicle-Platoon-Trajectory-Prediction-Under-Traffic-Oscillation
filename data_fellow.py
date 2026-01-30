import pandas as pd
import numpy as np

# ---------------------------------------------------------
df = data.copy()

# ==========================================
# Step 1: Retrieve Preceding Vehicle Data
# ==========================================
# Instead of iterating through rows (which is slow), we use a "Self-Merge" approach.
# We prepare a subset of data containing only the potential preceding vehicle information.

# Select relevant columns for the preceding vehicle
prec_cols = ['Vehicle_ID', 'Global_Time', 'v_Vel', 'v_Acc', 'Local_X', 'Local_Y', 'v_Length', 'v_Width']
df_prec = df[prec_cols].copy()

# Rename columns to avoid conflict and indicate they belong to the preceding vehicle
# 'Preceding' in the original dataframe will match 'Vehicle_ID' in this subset
df_prec.columns = ['Preceding', 'Global_Time', 'Prec_speed', 'Prec_acc', 'Prec_Local_x', 'Prec_Local_y',
                   'Prec_v_Length', 'Prec_v_Width']

# Merge current data with preceding data based on Preceding ID and Timestamp
# This acts like a database LEFT JOIN
merged_df = pd.merge(df, df_prec, on=['Preceding', 'Global_Time'], how='left')

# Remove rows where no preceding vehicle data was found (equivalent to isempty check)
merged_df = merged_df.dropna(subset=['Prec_speed'])

# ==========================================
# Step 2: Filter by Car-Following Thresholds
# ==========================================
time_headway_threshold = 6
space_headway_threshold = 65

# Create a boolean mask for the conditions
# Condition: Time Headway <= 6 AND Space Headway <= 65
condition_mask = (merged_df['Time_Headway'] <= time_headway_threshold) & \
                 (merged_df['Space_Headway'] <= space_headway_threshold)

# Apply the filter
filtered_df = merged_df[condition_mask].copy()

# ==========================================
# Step 3: Filter by Duration (>= 10 seconds)
# ==========================================
if not filtered_df.empty:
    # Sort by Vehicle ID and Time to ensure sequential processing
    filtered_df = filtered_df.sort_values(by=['Vehicle_ID', 'Global_Time'])

    # Logic to identify continuous segments:
    # 1. Check if Vehicle_ID changes compared to the previous row.
    # 2. Check if the time gap between rows is larger than the sampling rate (100ms).

    # Calculate time difference between current and previous row
    time_diff = filtered_df['Global_Time'].diff()

    # Check for ID change
    id_change = filtered_df['Vehicle_ID'] != filtered_df['Vehicle_ID'].shift(1)

    # Check for time break (assuming 100ms sampling rate, allowing strict continuity)
    # If the gap is > 100ms, it implies a break in the car-following condition
    time_break = time_diff > 100

    # Assign a unique ID to each continuous segment using cumulative sum
    # Any time a break is detected (ID change or Time gap), the segment_id increments
    filtered_df['segment_id'] = (id_change | time_break).cumsum()

    # Calculate the duration (Peak-to-Peak) for each segment
    # np.ptp calculates (max - min)
    segment_durations = filtered_df.groupby('segment_id')['Global_Time'].transform(np.ptp)

    # Filter segments that last at least 10,000 ms (10 seconds)
    valid_data = filtered_df[segment_durations >= 10000].copy()

    # Clean up auxiliary columns
    valid_data = valid_data.drop(columns=['segment_id'])
else:
    valid_data = pd.DataFrame()

# ==========================================
# Output Results
# ==========================================
print(f"Original data size: {len(data)}")
print(f"Filtered valid data size: {len(valid_data)}")

# Save to CSV
# valid_data.to_csv('filtered_following_data.csv', index=False)
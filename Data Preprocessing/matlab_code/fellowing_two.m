% Load NGSIM data
data = trajectories05150530;  % Replace with your data

% Initialize result storage
valid_data = [];

% Loop through each vehicle
% vehicle_ids = unique(data.Vehicle_ID);
vehicle_ids = unique_ids;
for i = 1:length(vehicle_ids)
    % Filter data for the current vehicle
    current_vehicle_data = data(data.Vehicle_ID == vehicle_ids(i), :);
    
    % Initialize continuous car-following data
    continuous_data = [];
    
    % Iterate through each time step (Global_Time)
    j = 1;
    while j <= height(current_vehicle_data)
        % Get the vehicle ID and preceding vehicle ID at the current time step
        current_time = current_vehicle_data.Global_Time(j);
        current_vehicle_id = current_vehicle_data.Vehicle_ID(j);
        preceding_vehicle_id = current_vehicle_data.Preceding(j);
        
        % Find the preceding vehicle's data from the current time step onwards
        preceding_data = data(data.Vehicle_ID == preceding_vehicle_id & ...
                              data.Global_Time >= current_time, :);
        
        if isempty(preceding_data)
            j = j + 1;  % If no preceding vehicle data is found, skip to the next time step
            continue;
        end
        
        % Get Space_Headway and Time_Headway of the current following vehicle
        space_headway = current_vehicle_data.Space_Headway(j);
        time_headway = current_vehicle_data.Time_Headway(j);
        
        % Car-following conditions: thresholds for time headway and space headway
        time_headway_threshold = 6;  % Assume time headway is less than or equal to 6 seconds
        space_headway_threshold = 65;  % Assume space headway is less than or equal to 65 meters
        
        % Check if the current time step satisfies the car-following conditions
        % (Time headway <= 6 OR Time headway is infinity, AND Space headway <= 65)
        if (time_headway <= time_headway_threshold || isinf(time_headway)) && space_headway <= space_headway_threshold
            % Extract following vehicle data at the current time step
            following_data = current_vehicle_data(j, :);
            
            % Find the corresponding preceding vehicle data at the exact current time step
            preceding_data_at_time = preceding_data(preceding_data.Global_Time == current_time, :);
            
            if ~isempty(preceding_data_at_time)
                % Extract speed, acceleration, and position of the preceding vehicle
                preceding_speed = preceding_data_at_time.v_Vel;
                preceding_acceleration = preceding_data_at_time.v_Acc;
                preceding_Local_x = preceding_data_at_time.Local_X;
                preceding_Local_y = preceding_data_at_time.Local_Y;
                
                % Append preceding vehicle data to the following vehicle data
                following_data.Prec_speed = preceding_speed;
                following_data.Prec_acc = preceding_acceleration;
                following_data.Prec_Local_x = preceding_Local_x;
                following_data.Prec_Local_y = preceding_Local_y;
                following_data.Prec_v_Length = preceding_data_at_time.v_Length;
                following_data.Prec_v_Width = preceding_data_at_time.v_Width;
                
                % Add this record to the continuous data array
                continuous_data = [continuous_data; following_data];
            end
            
            j = j + 1;  % Continue checking the next time step
        else
            % If car-following conditions are no longer met, check if the continuous duration is >= 10 seconds
            if ~isempty(continuous_data)
                % Check the time range of the continuous data (10 seconds = 10,000 milliseconds)
                if max(continuous_data.Global_Time) - min(continuous_data.Global_Time) >= 10*1000
                    valid_data = [valid_data; continuous_data];  % Save the valid continuous data
                end
                continuous_data = [];  % Reset continuous data
            end
            j = j + 1;  % Move to the next time step even if conditions are not met
        end
    end
    
    % Check if the last segment meets the >= 10 seconds duration condition
    if ~isempty(continuous_data)
        if max(continuous_data.Global_Time) - min(continuous_data.Global_Time) >= 10*1000
            valid_data = [valid_data; continuous_data];  % Save the valid continuous data
        end
    end
end

% Save the filtered stable car-following data
% writetable(valid_data, 'filtered_following_data.csv');

% data.Global_Time(2) - data.Global_Time(1)
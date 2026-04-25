save_path = 'E:\matlab_mode_data\traffic_flow\NGSIM'
for L=1
% Time sequence length - Observation window length
sequence_length = 50;
T=L-1; % Prediction window - 1

% Initialize storage
train_data = [];
train_real_speed = [];
train_s_safe = [];
lable_data = [];

% Iterate through each continuous car-following data segment
unique_ids = unique(valid_data.Vehicle_ID);

for i = 1:length(unique_ids)
    % Get continuous data for each vehicle
    current_data = valid_data(valid_data.Vehicle_ID == unique_ids(i), :);
    
    % Skip data that does not meet the required sequence length
    if height(current_data) < sequence_length + 1+T
        continue;
    end
    n=find(current_data.distance_fellow<0); % Remove anomalous data
    if length(n)>0;
           continue;
    end
    n=[];
    
    % Generate time-series data using a sliding window
    for j = 1:(height(current_data) - sequence_length-T)
        % Get the sequence data
        sequence = current_data(j:j+sequence_length-1, :);
        
        % Construct input features
        input_features = [ ...
            sequence.v_Vel, ...                           % Current speed
            sequence.distance_fellow, ...                 % Following distance
            sequence.diff_v, ...                          % Speed difference
            sequence.v_Acc,...
            sequence.Local_Y,...
            sequence.Prec_speed,...
            sequence.Prec_acc,...
            sequence.Prec_Local_y,...
        ];
        
        % Construct actual next speed
        next_speed = current_data.v_Vel(j + sequence_length:j + sequence_length+T);  % Actual speed at the next time step
        
        % Construct next following distance
        next_distance = current_data.distance_fellow(j + sequence_length:j + sequence_length+T);  % Safe/Following distance at the next time step

        % Construct next acceleration
        next_acc=current_data.v_Acc(j + sequence_length:j + sequence_length+T);
        
        % Construct next position
        next_y_piont=current_data.Local_Y(j + sequence_length:j + sequence_length+T);

        % Preceding vehicle's future speed
        next_Prec_speed=current_data.Prec_speed(j + sequence_length:j + sequence_length+T);

        % Preceding vehicle's future position
        next_Prec_Local_y=current_data.Prec_Local_y(j + sequence_length:j + sequence_length+T);
        
        % Construct label features
        lab_features = [ ...
            next_speed, ...                               % Future speed
            next_distance, ...                            % Future following distance
            next_acc, ...                                 % Future acceleration
            next_y_piont,...
            next_Prec_speed,...
            next_Prec_Local_y,...
        ];

        % Append to training data
        train_data = cat(1, train_data, reshape(input_features, 1, sequence_length, size(input_features, 2)));
        lable_data = cat(1, lable_data, reshape(lab_features, 1, 1+T, size(lab_features, 2)));
        % train_real_speed= cat(1, train_real_speed, next_speed);
        % train_s_safe= cat(1, train_s_safe, safe_distance);
    end
    dims=size(train_data);
    length_333=dims(1);
    if length_333>40000
        break
    end

end

    filename = sprintf('data_%.0f.mat', T);  % Create filename
    full_filename = fullfile(save_path, filename);  % Get the full file path
    
    % Save train_data and lable_data as a .mat file
    save(full_filename, 'train_data', 'lable_data');  % Save data to .mat file
    
    % Clear all variables except valid_data, T, and save_path
    vars = who;  % Get all variable names in the current workspace
    vars(strcmp(vars, 'valid_data') | strcmp(vars, 'T') | strcmp(vars, 'save_path')) = [];  % Exclude valid_data, T, and save_path variables
    clear(vars{:});  % Clear the rest of the variables
end

% % Convert to tensor format
% train_data = torch.tensor(train_data, dtype=torch.float32);
% train_real_speed = torch.tensor(train_real_speed, dtype=torch.float32);
% train_s_safe = torch.tensor(train_s_safe, dtype=torch.float32);
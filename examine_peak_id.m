%% examine the where comes the peak and analysis the character of peak
clc
clear
load('data2/data_time.mat');
num_cell = size(time_train,2);
trial_train_num = [];
for i = 1:num_cell
    trial = time_train(i);
    trial = cell2mat(trial);
    num_spl = size(trial,1);
    if num_spl == 0
        continue
    end
    num_inst = num_spl - 5;
    trial_train_num = [trial_train_num num_inst];
end

%calculate the number trial cell changes
trial_num_changes = [];
count = 0;
trial_num_changes = [trial_num_changes count];
for i = 1:size(trial_train_num, 2)
    count = count + trial_train_num(i);
    trial_num_changes = [trial_num_changes count];
end
trial_num_changes = trial_num_changes';
% disp(trial_num_changes);

% get peak position of train_id_x1b1
load('error_id_train_x1b1.mat');
position = []
for i = 1:size(error_id_train,1)
    found = 0;
    flag = zeros(1,9);
    for j = 1:9
        if abs(error_id_train(i,j)) > 0.1
            flag(j) = 1;
            found = 1;
        end
    end
    if found == 1
        position = [position; [i flag]];
    end
end

% plot spread
peak_pos = zeros(1,size(trainX, 1));
for i = 1:size(position,1)
    peak_pos(position(i,1)) = 1;
end
figure
plot(1:size(trainX,1),peak_pos);
xlabel('train sample idx');
ylabel('peak = 1');

% get mini diff between pos and trial change
diffs = [];
for i = 1:size(position,1)
    diffs = [diffs find_close(position(i), trial_num_changes)];
end

figure
plot(diffs)
xlabel('peak idx');
ylabel('mini distance to trial change')

% calculate the peak times in terms of dimensions
count_pos = zeros(1,9);
for i = 1:size(position,1)
    count_pos = count_pos + position(i,2:10)
end
figure
plot(count_pos);
xlabel('x dimension')
ylabel('peak times')

function minimum = find_close(pos, trial_num_changes)
    minimum = 100000000;
    for i = 1:size(trial_num_changes,1)
        diff = abs(pos - trial_num_changes(i))
        if diff < minimum
            minimum = diff
        end
    end
end

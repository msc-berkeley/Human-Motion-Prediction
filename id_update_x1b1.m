%% id algorithm update, trained cell by cell, with time stamp, and prediction model is (t,t+1,t+2) -> (t+1,t+2,t+3)
clear;
clc;
%% prepare training data
str = 'data2/';% with .4 .6 combination
load(strcat(str,'weights1.mat'));
load(strcat(str,'weights2.mat'));
load(strcat(str,'weights3.mat'));
load(strcat(str,'biases1.mat'));
load(strcat(str,'biases2.mat'));
load(strcat(str,'biases3.mat'));
load(strcat(str,'data_time.mat'));

%Extended Kalman Filter
Q = .02^2; % covariance of process noise
R = 0; % convariance of observation noise
U = weights1 * weights2; % approximate parameter of first layer
b1 = biases1 * weights2 + biases2; % approximate parameter of first layer
W = weights3; % parameter of second layer
b2 = biases3; % reconstruction error
k = 20; % learning gain
alpha = 5; % learning gain 
gamma = 50; % learning gain
beta1 = 1.25; % learning gain
flag_step_large = 0;
syms symbo;
plan = [];
obs = [];
num = size(trainX, 1);
for i = 1:num
    plan = [plan;trainX(i,10)];
    obs = [obs;trainX(i,1:9)];
end

%% online identifier-based update
time_step = zeros(1,3);
error_id_train = [];
Gamma_W = 0.1*eye(40);
Gamma_Ux = 0.2*eye(9);
Gamma_Ug = 0.2*eye(1);
[obs_cell, obs_y_cell, obs_p_cell, num_cell] = time_extract(time_train, obs, trainY, plan);
count = 0;
v = zeros(1, 9); % Filippov generalized solution
% fully leverage training data
for c = 1:num_cell
    % training within each cell
    error_id = [];
    trial = time_train(c);
    trial_time = cell2mat(trial);
    obs_x = obs_cell(c);
    obs_y = obs_y_cell(c);
    obs_x = cell2mat(obs_x);
    obs_y = cell2mat(obs_y);
    num = size(obs_x,1);
    if num == 0
        continue;
    end  
%     x_id = obs_x(1,:) - 0.1*rand(1,9); % bound the inital error within 0.2
%     x0_tilde = obs_x(1,:) - x_id;
    x_id = obs_x(1,:);
    x0_tilde = 0;
    
    for i = 1:num
        if i < num - 1
%                 time_step(1) = time_step_train(i) + time_step_train(i+1) + time_step_train(i+2);
            time_step(1) = ((trial_time(i+1,5) - trial_time(i,5))*60 + trial_time(i+1,6)) - trial_time(i,6);
            time_step(2) = ((trial_time(i+2,5) - trial_time(i+1,5))*60 + trial_time(i+2,6)) - trial_time(i+1,6);
            time_step(3) = ((trial_time(i+3,5) - trial_time(i+2,5))*60 + trial_time(i+3,6)) - trial_time(i+2,6)
            for stp_num = 1:3
                if time_step(stp_num) > 0.3
                    flag_step_large = flag_step_large + 1; %monitor if the time_step is too large
                end
            end
        end
        count = count + 1;
        disp(count);
        g = plan(count);
        s_hat = [x_id, g, 1];
        layer1 = s_hat*U;
        activate = arrayfun(@(x) 1/(1 + exp(-x)), layer1);
        layer2 = activate*W;

        % calculate RISE feedback 
        x_tilde = obs_x(i,:) - x_id;
        mu = k*x_tilde - k*x0_tilde + v; % RISE feedback
        v_deri = (k*alpha + gamma)*x_tilde + beta1*sign(x_tilde);
        v1 = v(1,1:3) + time_step(1)*v_deri(1,1:3);
        v2 = v(1,4:6) + time_step(2)*v_deri(1,4:6);
        v3 = v(1,7:9) + time_step(3)*v_deri(1,7:9);
        v = [v1 v2 v3];

        % calculate error (three methods)
        error_id = [error_id;x_tilde]; % error in id method

        % state identification update
        x_deri = layer2 + mu;
        x_id1 = x_id(1,1:3) + time_step(1)*x_deri(1,1:3);
        x_id2 = x_id(1,4:6) + time_step(2)*x_deri(1,4:6);
        x_id3 = x_id(1,7:9) + time_step(3)*x_deri(1,7:9);
        x_id = [x_id1 x_id2 x_id3];

        % update direction(derivation)
        if i < num
          g_deri = (plan(count+1) - plan(count))/time_step(1);
        end
        sigma_deri_exp = gradient(1/(1 + exp(-symbo)), symbo);
        sigma_deri = [];% derivative of1 activation function based on layer1
        for t = 1:size(layer1, 2)
            sigma_deri = [sigma_deri eval(subs(sigma_deri_exp, symbo, layer1(t)))];
        end
        sigma_deri_M = [];
        for p = 1:size(sigma_deri, 2)
            sigma_deri_M = blkdiag(sigma_deri_M, sigma_deri(p));
        end
        U_x = U(1:size(obs, 2), :);
        U_g = U(size(obs, 2) + 1, :);
        W_deri = Gamma_W*sigma_deri_M*U_x'*x_deri'*x_tilde;
        Ux_deri = Gamma_Ux*x_deri'*x_tilde*W'*sigma_deri_M;
        Ug_deri = Gamma_Ug*g_deri*x_tilde*W'*sigma_deri_M;

        % updata parameter
%             W = W + W_deri*time_step;
        W1 = W(:,1:3) + time_step(1)*W_deri(1,1:3);
        W2 = W(:,4:6) + time_step(2)*W_deri(1,4:6);
        W3 = W(:,7:9) + time_step(3)*W_deri(1,7:9);
        W = [W1 W2 W3];

        U_x1 = U_x(1:3,:) + time_step(1)*Ux_deri(1:3,:);
        U_x2 = U_x(4:6,:) + time_step(2)*Ux_deri(4:6,:);
        U_x3 = U_x(7:9,:) + time_step(3)*Ux_deri(7:9,:);
        U_x = [U_x1;U_x2;U_x3];

        U_g = U_g + Ug_deri*time_step(1);
        U = [U_x; U_g; U(11,:)];
    end
    
    
    % update error_id_train
    for idx = 1:size(error_id)
        error_id_train = [error_id_train;error_id(idx,:)];
    end
end


%% plot train
save('error_id_train_x1b1.mat','error_id_train');
figure
num = size(error_id_train, 1);
plot_err(error_id_train, num, 'index', 'error_{id}');

%% plot test
% please run id_update_x1b1_test.m




%% identifier-based algorithm for updating NN parameters 
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
% load('time_stp_train.mat');
% load('time_stp_test.mat');

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
time_step = 0.01;
% time_step_train = time_stp_train;
% time_step_test = time_stp_test;
% vector = sym('vector', [1 9]);
syms symbo;
plan = [];
obs = [];
num = size(trainX, 1);
for i = 1:num
    plan = [plan;trainX(i,10)];
    obs = [obs;trainX(i,1:9)];
end

%% online identifier-based update
v = zeros(1, 9); % Filippov generalized solution
error_id = [];
% error_sigma = [];
% error_sigma_id = [];
% error_ori = [];
Gamma_W = 0.1*eye(40); % constant weighting matrix
Gamma_Ux = 0.2*eye(9); % constant weighting matrix
Gamma_Ug = 0.2*eye(1); % constant weighting matrix
% fully leverage training data
% note that estimation model: (t,t+1,t+2) -> (t+3,t+4,t+5)
% trained sequence(trainX): 1,4,7....2,5,8,....3,6,9,...
for start = 1:3
    x_id = obs(start,:) - rand(1,9); % initiate x_id based on first observation
    x0_tilde = obs(start,:) - x_id; % x0_tilde
    for i = start:3:num
        disp(i);
%         if i < (num - 1) 
%             time_step = time_step_train(i) + time_step_train(i+1) + time_step_train(i+2);
%         end
        g = plan(i);
        s_hat = [x_id, g, 1];
        layer1 = s_hat*U;
        activate = arrayfun(@(x) 1/(1 + exp(-x)), layer1);
        layer2 = activate*W;

        % calculate RISE feedback 
        x_tilde = obs(i,:) - x_id;
        mu = k*x_tilde - k*x0_tilde + v; % RISE feedback
        v_deri = (k*alpha + gamma)*x_tilde + beta1*sign(x_tilde);
        v = v + time_step*v_deri; % update Filippov generalized solution
        
        % calculate error (three methods)
        error_id = [error_id;x_tilde]; % error in id method

        % state identification update
        x_deri = layer2 + mu;
        x_id = x_id + time_step*x_deri;

        % parameter update direction(derivation)
        if i < (num - 1)
          g_deri = (plan(i+1) - plan(i))/time_step; % in our experiment scenario, g_deri = 0 in most cases
        end
        sigma_deri_exp = gradient(1/(1 + exp(-symbo)), symbo);
        sigma_deri = [];% derivative of the activation sigmoid function with respect to input layer1
        for t = 1:size(layer1, 2)
            sigma_deri = [sigma_deri eval(subs(sigma_deri_exp, symbo, layer1(t)))];
        end
        sigma_deri_M = []; % activation derivtion matrix
        for p = 1:size(sigma_deri, 2)
            sigma_deri_M = blkdiag(sigma_deri_M, sigma_deri(p)); 
        end
        U_x = U(1:size(obs, 2), :);
        U_g = U(size(obs, 2) + 1, :);
        W_deri = Gamma_W*sigma_deri_M*U_x'*x_deri'*x_tilde;
        Ux_deri = Gamma_Ux*x_deri'*x_tilde*W'*sigma_deri_M;
        Ug_deri = Gamma_Ug*g_deri*x_tilde*W'*sigma_deri_M;
        % updata parameter
        W = W + W_deri*time_step;
        U_x = U_x + Ux_deri*time_step;
        U_g = U_g + Ug_deri*time_step;
        U = [U_x; U_g; U(11,:)];
        
    end
end


%% modify order error into original
% change the order of 1,4,7..2,5,8,..3,6,9.. to 1,2,3,4,5...num
num = size(error_id, 1);
error_id_ori = zeros(num, 9);
j = 1;
for start = 1:3
    for i = start:3:num
        error_id_ori(i,:) = error_id(j,:);
        j = j + 1;
    end
end
%% plot train
save('error_id_ori_1e-2.mat','error_id_ori');
figure
num = size(error_id, 1);
plot_err(error_id_ori, num, 'index', 'error_{id}');

%% test
plan_test = [];
obs_test = [];   
num = size(TestX, 1);
for i = 1:num
    plan_test = [plan_test;TestX(i,10)];
    obs_test = [obs_test;TestX(i,1:9)];
end
%% online identifier-based update test
% v = zeros(1, 9); % Filippov generalized solution
error_id_test = [];
error_sigma_test = [];
error_sigma_id_test = [];
error_ori_test = [];
% fully leverage training data
for start = 1:3
    x_id = obs_test(start,:) - rand(1,9);
    x0_tilde = obs_test(start,:) - x_id;
    for i = start:3:num
        disp(i);
%         if i < (num - 1) 
%             time_step = time_step_test(i) + time_step_test(i+1) + time_step_test(i+2);
%         end
        g = plan_test(i);
        s_hat = [x_id, g, 1];
        layer1 = s_hat*U;
        activate = arrayfun(@(x) 1/(1 + exp(-x)), layer1);
        layer2 = activate*W;

        % calculate RISE feedback 
        x_tilde = obs_test(i,:) - x_id;
        mu = k*x_tilde - k*x0_tilde + v; % RISE feedback
        v_deri = (k*alpha + gamma)*x_tilde + beta1*sign(x_tilde);
        v = v + time_step*v_deri; % update Filippov generalized solution
%         v = v_deri; % update Filippov generalized solution
        
        % calculate error (three methods)
        error_id_test = [error_id_test;x_tilde]; % error in id method
        
        % state identification update
        x_deri = layer2 + mu;
        x_id = x_id + time_step*x_deri;

        % update direction(derivation)
        if i < num - 1
          g_deri = (plan_test(i+1) - plan_test(i))/time_step;
        end
        sigma_deri_exp = gradient(1/(1 + exp(-symbo)), symbo);
        sigma_deri = [];% derivative of activation function based on layer1
        for t = 1:size(layer1, 2)
            sigma_deri = [sigma_deri eval(subs(sigma_deri_exp, symbo, layer1(t)))];
        end
        sigma_deri_M = [];
        for p = 1:size(sigma_deri, 2)
            sigma_deri_M = blkdiag(sigma_deri_M, sigma_deri(p));
        end
        U_x = U(1:size(obs_test, 2), :);
        U_g = U(size(obs_test, 2) + 1, :);
        W_deri = Gamma_W*sigma_deri_M*U_x'*x_deri'*x_tilde;
        Ux_deri = Gamma_Ux*x_deri'*x_tilde*W'*sigma_deri_M;
        Ug_deri = Gamma_Ug*g_deri*x_tilde*W'*sigma_deri_M;
        
        % updata parameter
        W = W + W_deri*time_step;
        U_x = U_x + Ux_deri*time_step;
        U_g = U_g + Ug_deri*time_step;
        U = [U_x; U_g; U(11,:)];
        
    end
end
%% modify order error into original
num = size(error_id_test, 1);
error_id_test_ori = zeros(num, 9);
j = 1;
for start = 1:3
    for i = start:3:num
        error_id_test_ori(i,:) = error_id_test(j,:);
        j = j + 1;
    end
end
%% plot test
save('error_id_test_ori_1e-2.mat','error_id_test_ori');
figure
num = size(error_id_test, 1);
plot_err(error_id_test_ori, num, 'index', 'error_id_test');

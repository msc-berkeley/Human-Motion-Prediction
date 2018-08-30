% identifier-based algorithm to adapt NN parameter
% a demo of id method
% demo on human motion prediction (cell by cell)
%
% input: human motion data (trainX trainY / TestX TestY)
% output: id algorithm prediction error (error_train / error_test)
% --------------------------------------------------------
% Identifier-based algorithm implementation
% Also see from MATLAB Human-Motion-Prediction 
% https://github.com/msc-berkeley/Human-Motion-Prediction
% --------------------------------------------------------
clc;
clear;
%% -------------------- CONFIG TRAINING --------------------
str = './para/para_cmu/';% with .4 .6 combination
load(strcat(str,'weights1_cmu.mat'));
load(strcat(str,'weights2_cmu.mat'));
load(strcat(str,'weights3_cmu.mat'));
load(strcat(str,'biases1_cmu.mat'));
load(strcat(str,'biases2_cmu.mat'));
load(strcat(str,'biases3_cmu.mat'));
load('data2/cmu_data.mat');

% opts parameter
gamma_time = 3;
alpha_time = 3;
beta_time = 3;
k_time = 3;
opts.U          = weights1 * weights2; % approximate parameter of first layer
opts.W          = weights3; % parameter of second layer
opts.k          = 20*k_time; % learning gain
opts.alpha      = 5*alpha_time; % learning gain 
opts.gamma      = 50*gamma_time; % learning gain
opts.beta1      = 1.25*beta_time; % learning gain
opts.time_step  = 0.03;
opts.Gamma_W    = 0.1*eye(40); % constant weighting matrix
opts.Gamma_Ux   = 0.2*eye(9); % constant weighting matrix
opts.Gamma_Ug   = 0.2*eye(1); % constant weighting matrix
opts.y_dim      = size(trainY, 2);
opts.num        = size(trainX, 1);
error_train     = []; % training output
count           = 0;

% extract train data information
plan            = [];
obs             = [];
for i = 1:opts.num
    plan = [plan;trainX(i,10)];
    obs = [obs;trainX(i,1:9)];
end

% id-based model
id_obj          = id(opts.y_dim, opts.U, opts.W, ...
                   'k',         opts.k, ...
                   'alpha',     opts.alpha, ...
                   'gamma',     opts.gamma, ...
                   'beta1',     opts.beta1, ...
                   'time_step', opts.time_step, ...
                   'Gamma_W',   opts.Gamma_W, ...
                   'Gamma_Ux',  opts.Gamma_Ux, ...
                   'Gamma_Ug',  opts.Gamma_Ug);

%% -------------------- TRAINING --------------------
% id-based upgrating when new observation available
[obs_cell, obs_y_cell, obs_p_cell, num_cell] = time_extract(time_train, obs, trainY, plan);
% fully leverage training data
for c = 1:num_cell
    id_obj.error = [];
    obs_x = obs_cell(c);
    obs_p = obs_p_cell(c);
    obs_x = cell2mat(obs_x);
    obs_p = cell2mat(obs_p);
    num = size(obs_x,1);
    if num == 0
        continue;
    end
    % recover offline trained parameter per new trial
    id_obj          = id(opts.y_dim, opts.U, opts.W, ...
                   'k',         opts.k, ...
                   'alpha',     opts.alpha, ...
                   'gamma',     opts.gamma, ...
                   'beta1',     opts.beta1, ...
                   'time_step', opts.time_step, ...
                   'Gamma_W',   opts.Gamma_W, ...
                   'Gamma_Ux',  opts.Gamma_Ux, ...
                   'Gamma_Ug',  opts.Gamma_Ug);
    
    for start = 1:3
        id_obj.x_id = obs_x(start,:);
        id_obj.x0_tilde = 0;
        for i = start:3:num
            count = count + 1;
            disp(count); % display current training process
            if i > 3 
                g_tilde = obs_p(i,:) - obs_p(i-3,:); % g_tilde feeding to id algorithm
            else
                g_tilde = 0;
            end
            id_obj = id_process(id_obj, obs_p(i), g_tilde, obs_x(i,:));
        end
    end
    error_ori = modi_order(id_obj.error);
    % update error_train
    for idx = 1:size(error_ori)
        error_train = [error_train;error_ori(idx,:)];
    end
end

% train error figure
plot_err(error_train, opts.num, 'index', 'error_{train}_{id}');
% save('./saved_results/id_cmu_t2ini.mat','error_train');
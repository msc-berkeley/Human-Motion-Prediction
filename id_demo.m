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
str = 'data2/';% with .4 .6 combination
load(strcat(str,'weights1.mat'));
load(strcat(str,'weights2.mat'));
load(strcat(str,'weights3.mat'));
load(strcat(str,'biases1.mat'));
load(strcat(str,'biases2.mat'));
load(strcat(str,'biases3.mat'));
load(strcat(str,'data_time.mat'));

% opts parameter
opts.U          = weights1 * weights2; % approximate parameter of first layer
opts.W          = weights3; % parameter of second layer
opts.k          = 20; % learning gain
opts.alpha      = 5; % learning gain 
opts.gamma      = 50; % learning gain
opts.beta1      = 1.25; % learning gain
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


%% -------------------- CONFIG TESTING --------------------
% opts parameter
opts.U          = id_obj.U; % trained parameter of first layer
opts.W          = id_obj.W; % trained parameter of second layer
opts.v          = id_obj.v % trained v 
opts.time_step  = 0.06;
opts.num        = size(TestX, 1);
error_test      = []; % training output
count           = 0;

% extract test data information
plan            = [];
obs             = [];
for i = 1:opts.num
    plan = [plan;TestX(i,10)];
    obs = [obs;TestX(i,1:9)];
end

% id-based model
id_obj_test     = id(opts.y_dim, opts.U, opts.W, ...
                   'v',         opts.v, ...
                   'k',         opts.k, ...
                   'alpha',     opts.alpha, ...
                   'gamma',     opts.gamma, ...
                   'beta1',     opts.beta1, ...
                   'time_step', opts.time_step, ...
                   'Gamma_W',   opts.Gamma_W, ...
                   'Gamma_Ux',  opts.Gamma_Ux, ...
                   'Gamma_Ug',  opts.Gamma_Ug);

               
%% -------------------- TESTING --------------------
% id-based upgrating when new observation available
[obs_cell, obs_y_cell, obs_p_cell, num_cell] = time_extract(time_train, obs, TestY, plan);
% fully leverage training data
for c = 1:num_cell
    id_obj_test.error = [];
    obs_x = obs_cell(c);
    obs_p = obs_p_cell(c);
    obs_x = cell2mat(obs_x);
    obs_p = cell2mat(obs_p);
    num = size(obs_x,1);
    if num == 0
        continue;
    end
    for start = 1:3
        id_obj_test.x_id = obs_x(start,:);
        id_obj_test.x0_tilde = 0;
        for i = start:3:num
            count = count + 1;
            disp(count); % display current training process
            if i > 3 
                g_tilde = obs_p(i,:) - obs_p(i-3,:); % g_tilde feeding to id algorithm
            else
                g_tilde = 0;
            end
            id_obj_test = id_process(id_obj_test, obs_p(i), g_tilde, obs_x(i,:));
        end
    end
    error_ori = modi_order(id_obj_test.error);
    % update error_train
    for idx = 1:size(error_ori)
        error_test = [error_test;error_ori(idx,:)];
    end
end

% test error figure
plot_err(error_test, opts.num, 'index', 'error_{test}_{id}');

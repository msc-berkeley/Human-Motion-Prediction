% least square to get the layer3 parameters
% a demo of rls-paa method
% demo on human motion prediction
%
% input: human motion data (trainX trainY / TestX TestY)
% output: rls-paa prediction error (rls_obj.error / rls_obj_test.error)
% --------------------------------------------------------
% RLS-PAA implementation
% Also see from MATLAB Human-Motion-Prediction 
% https://github.com/msc-berkeley/Human-Motion-Prediction
% --------------------------------------------------------
clc;
clear;
addpath('class_path','lib')
%% -------------------- CONFIG TRAINING --------------------
str = 'data2/';% with .4 .6 combination
load(strcat(str,'weights1.mat'));
load(strcat(str,'weights2.mat'));
load(strcat(str,'weights3.mat'));
load(strcat(str,'biases1.mat'));
load(strcat(str,'biases2.mat'));
load(strcat(str,'biases3.mat'));
load(strcat(str,'data.mat'));
layer1 = max(0,trainX * double(weights1) + double(biases1));
layer2 = max(0,layer1 * double(weights2) + double(biases2));
% observation encoded
encode          = layer2;
encode          = [encode, ones(size(encode,1),1)];
% opts parameter
opts.num        = size(encode, 1);
opts.lambda     = 0.9998;
opts.W          = .02^2;
opts.nn_dim     = size(encode, 2);
opts.y_dim      = size(trainY, 2);

% rls-paa model
rls_obj         = rls(opts.num, opts.nn_dim, opts.y_dim, ...
                                    'W',          opts.W, ...
                                    'lambda',     opts.lambda);

%% -------------------- TRAINING --------------------
% RLS-PAA upgrating when new observation available
for i = 1:opts.num
    disp(i); 
    phi = encode(i, :); 
    rls_obj = rls_obj.rls_update(phi, i, opts.y_dim, trainY);
end

% plot training error
plot_err(rls_obj.error, opts.num, 'index', 'error_{ls}');

%% -------------------- CONFIG TESTING --------------------
layer1 = max(0,TestX * double(weights1) + double(biases1));
layer2 = max(0,layer1 * double(weights2) + double(biases2));
% observation encoded
encode          = layer2;
encode          = [encode, ones(size(encode,1),1)];
% opts parameter
opts.num        = size(encode, 1);
opts.lambda     = 0.9998;
opts.W          = .02^2;
opts.nn_dim     = size(encode, 2);
opts.y_dim      = size(TestY, 2);
% trained parameter
opts.F_M        = rls_obj.F_M;
opts.X_theta    = rls_obj.X_theta;
opts.theta      = rls_obj.theta;

% rls-paa model
rls_obj_test         = rls(opts.num, opts.nn_dim, opts.y_dim, ...
                                    'W',          opts.W, ...
                                    'lambda',     opts.lambda, ...
                                    'F_M',        opts.F_M , ...
                                    'X_theta',    opts.X_theta , ...
                                    'theta',      opts.theta ); 

%% -------------------- TESTING --------------------
% RLS-PAA upgrating when new observation available
for i = 1:opts.num
    disp(i); 
    phi = encode(i, :);
    rls_obj_test = rls_obj_test.rls_update(phi, i, opts.y_dim, TestY);
end

% plot testing error
plot_err(rls_obj_test.error, opts.num, 'index', 'error_{ls}');
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
%% -------------------- CONFIG TRAINING --------------------
% str = 'data2/';% with .4 .6 combination
% load(strcat(str,'weights1.mat'));
% load(strcat(str,'weights2.mat'));
% load(strcat(str,'weights3.mat'));
% load(strcat(str,'biases1.mat'));
% load(strcat(str,'biases2.mat'));
% load(strcat(str,'biases3.mat'));
% load(strcat(str,'data.mat'));
str = './para/para_fake/axis_uniform/';% with .4 .6 combination
load(strcat(str,'weights1_axis_uniform.mat'));
load(strcat(str,'weights2_axis_uniform.mat'));
load(strcat(str,'weights3_axis_uniform.mat'));
load(strcat(str,'biases1_axis_uniform.mat'));
load(strcat(str,'biases2_axis_uniform.mat'));
load(strcat(str,'biases3_axis_uniform.mat'));
load('fake_data/data_axis_noise_uniform/data_time.mat');

layer1 = max(0,trainX * double(weights1) + double(biases1));
layer2 = max(0,layer1 * double(weights2) + double(biases2));
% observation encoded
encode          = layer2;
encode          = [encode, ones(size(encode,1),1)];
% opts parameter
opts.num        = size(encode, 1);
opts.lambda     = 0.998;
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
error = rls_obj.error;
%% offline trained nn for online prediction 
clc;
clear;
%% -------------------- CONFIG TRAINING --------------------
str = './para/para_fake/time_uniform/';% with .4 .6 combination
load(strcat(str,'weights1_time_uniform.mat'));
load(strcat(str,'weights2_time_uniform.mat'));
load(strcat(str,'weights3_time_uniform.mat'));
load(strcat(str,'biases1_time_uniform.mat'));
load(strcat(str,'biases2_time_uniform.mat'));
load(strcat(str,'biases3_time_uniform.mat'));
load('fake_data/data_time_noise_uniform/data_time.mat');

% RLS-PAA NN 
layer1 = max(-inf,trainX*double(weights1) + double(biases1));
layer2 = max(-inf,layer1*double(weights2) + double(biases2));
layer3 = max(-inf,layer2*double(weights3) + double(biases3));

error_rls_nn = trainY - layer3;
num = size(error_rls_nn, 1);
figure
plot_err(error_rls_nn, num, 'index', 'error_{rls-nn}');

% ID NN
U           = weights1 * weights2; % approximate parameter of first layer
W           = weights3; % parameter of second layer
layer1_id   = trainX*U;
num         = size(trainX, 1);
activate    = [];
for i = 1:num
    activate_tmp = arrayfun(@(x) 1/(1 + exp(-x)), layer1_id(i,:));
    activate = [activate;activate_tmp];
end
layer2_id   = activate*W;
error_id_nn = trainY - layer2_id;
figure
plot_err(error_id_nn, num, 'index', 'error_{id-nn}');

%% Extended Kalman Filter implemented version (ignore this file currently)
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
load(strcat(str,'data.mat'));

%Extended Kalman Filter
x = sym('x', [1 9]);
Q = .02^2; % covariance of process noise
R = 0; % convariance of observation noise
U = weights1 * weights2; % approximate parameter of first layer
b1 = biases1 * weights2 + biases2; % approximate parameter of first layer
W = weights3; % parameter of second layer
b2 = biases3; % reconstruction error
plan = [];
obs = [];
num = size(trainX, 1);
for i = 1:num
    plan = [plan;trainX(i,10)];
    obs = [obs;trainX(i,1:9)];
end

%% online identifier-based update
x_id = obs(1,:) - rand(1,9);
% x = assign(x, x_id)
P_hat = rand(9,9);
for i = 1:num
    g = plan(i);
    s_hat = [x, g, 1];
    x_new_hat = (s_hat * U + b1) * W + b2; 
    F = [];
    for t = 1:9
        F = [F, gradient(x_new_hat(t), x)]
    end
    x_new_value_exp = subs(x_new_hat, x, x_id);
    x_new_value = eval(x_new_value_exp);
    P_new = F_k * P_hat * F_k' + Q
end

function X = symbo(x,n,m)
    X = []
    for i = 1:n
        for j = 1:m
            eval(['syms ' x num2str(i) '_' num2str(j)]);
            eval(['X = [X ' x num2str(i) '_' num2str(j) ']']);
        end
    end
end
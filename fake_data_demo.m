function fake_data_demo()
% fake_data_demo()
% A demo of generating human motion fake data 
% --------------------------------------------------------
% fake data operation implementation
% Also see from MATLAB Human-Motion-Prediction 
% https://github.com/msc-berkeley/Human-Motion-Prediction
% --------------------------------------------------------

clc;
clear;

%% init opts1
opts1.time   = 5;
opts1.order  = 2;
opts1.theta  = [4, -4*opts1.time;
                0, .9;
                0, 1.05];
opts1.plan   = 1;
fake_data_generate(opts1.theta, opts1.order, ...
                      'plan',   opts1.plan, ...
                      'time',   opts1.time);
                  
%% init opts2
opts2.time   = 5;
opts2.order  = 2;
opts2.theta  = [4.1, -3.8*opts2.time;
                1, .9;
                0, .95];          
opts2.plan   = 2;
fake_data_generate(opts2.theta, opts2.order, ...
                      'plan',   opts2.plan, ...
                      'time',   opts2.time);
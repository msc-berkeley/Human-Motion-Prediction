function fake_data_demo()
% fake_data_demo()
% A demo of generating human motion fake data 
% Input
%   time        -  total sample time (start from 0)
%   order       -  the order of human motion polynomial respect to time
%   theta       -  parameter for motion polynomial of 3 axis (x,y,z)
%   plan        -  motion plan represented by motion polynomial
%   time_step   -  time interval per sample (frequency)
%   trial       -  noise added sampling trial number
%   time_noise  -  add time noise to polynomial function (t + noise)
%   axis_noise  -  add axis noise to polynomial function ((x,y,z) + noise)
%
% Output
%   dir(fake_data)  -  directory containing generated fake data + ground truth
%   data_time.mat   -  fake data converted training data (found in each noisy data folder)
%
% --------------------------------------------------------
% fake data operation implementation
% Also see from MATLAB Human-Motion-Prediction 
% https://github.com/msc-berkeley/Human-Motion-Prediction
% --------------------------------------------------------

clc;
clear;

%% clear previous fake data 
if ~exist('./fake_data','dir')==0
    rmdir('./fake_data','s');
end

%% init opts1 (plan 1 + time noise )
opts1.time          = 5;
opts1.order         = 2;
opts1.theta         = [4, -4*opts1.time;
                       0, .9;
                       0, 1.05];
opts1.plan          = 1;
opts1.time_step     = .05;
opts1.trial         = 50;
opts1.time_noise    = 1;
opts1.axis_noise    = 0;
fake_data_generate(opts1.theta, opts1.order, ...
                      'plan',       opts1.plan, ...
                      'time',       opts1.time, ...
                      'trial',      opts1.trial, ...
                      'time_step',  opts1.time_step, ...
                      'time_noise', opts1.time_noise, ...
                      'axis_noise', opts1.axis_noise);
                  
%% init opts3 (plan 2 + time noise )
opts2.time          = 5;
opts2.order         = 2;
opts2.theta         = [4.1, -3.8*opts2.time;
                       1, .9;
                       0, .95];          
opts2.plan          = 2;
opts2.time_step     = .05;
opts2.trial         = 50;
opts2.time_noise    = 1;
opts2.axis_noise    = 0;
fake_data_generate(opts2.theta, opts2.order, ...
                      'plan',       opts2.plan, ...
                      'time',       opts2.time, ...
                      'trial',      opts2.trial, ...
                      'time_step',  opts2.time_step, ...
                      'time_noise', opts2.time_noise, ...
                      'axis_noise', opts2.axis_noise);
                  
%% init opts1 (plan 1 + axis noise )
opts3.time          = 5;
opts3.order         = 2;
opts3.theta         = [4, -4*opts3.time;
                       0, .9;
                       0, 1.05];
opts3.plan          = 1;
opts3.time_step     = .05;
opts3.trial         = 50;
opts3.time_noise    = 0;
opts3.axis_noise    = 1;
fake_data_generate(opts3.theta, opts3.order, ...
                      'plan',       opts3.plan, ...
                      'time',       opts3.time, ...
                      'trial',      opts3.trial, ...
                      'time_step',  opts3.time_step, ...
                      'time_noise', opts3.time_noise, ...
                      'axis_noise', opts3.axis_noise);
                  
%% init opts2 (plan 2 + axis noise )
opts4.time   = 5;
opts4.order  = 2;
opts4.theta  = [4.1, -3.8*opts4.time;
                1, .9;
                0, .95];          
opts4.plan   = 2;
opts4.time_step     = .05;
opts4.trial         = 50;
opts4.time_noise    = 0;
opts4.axis_noise    = 1;
fake_data_generate(opts4.theta, opts4.order, ...
                      'plan',   opts4.plan, ...
                      'time',   opts4.time, ...
                      'trial',      opts4.trial, ...
                      'time_step',  opts4.time_step, ...
                      'time_noise', opts4.time_noise, ...
                      'axis_noise', opts4.axis_noise);
                  
                  
%% contruct fake motion training data set 
% state prediction model: x(t,t+1,t+2) -> x(t+3,t+4,t+5)
fake_data2train_data();

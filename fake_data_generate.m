function fake_data_generate(theta, order, varargin)
% --------------------------------------------------------
% Fake data implementation
% Also see from MATLAB Human-Motion-Prediction 
% https://github.com/msc-berkeley/Human-Motion-Prediction
% --------------------------------------------------------

%% inputs
    ip = inputParser;
    ip.addRequired('theta',                              @ismatrix);
    ip.addRequired('order',                              @isscalar);
    ip.addParameter('time_step',          .05,           @isscalar);
    ip.addParameter('time',                 5,           @isscalar);
    ip.addParameter('trial',               50,           @isscalar);
    ip.addParameter('plan',                 1,           @isscalar);

    ip.parse(theta, order, varargin{:});
    opts = ip.Results;
    
%% define time varying model
    % sample time 
    t = 0:opts.time_step:opts.time;
    
    % generate ground truth trajectory
    [x_gt, y_gt, z_gt] = func_struct(opts.theta, opts.order, t);
    p_gt = opts.plan*ones(1, size(t,2));
    data_gt = [x_gt; y_gt; z_gt; p_gt];
    
    % save ground truth 
    gt_path = strcat('./fake_data/ground_truth/plan',num2str(opts.plan));
    mkdir(gt_path);
    save(strcat(gt_path,'/gt.mat'), 'data_gt');   
    
%% generate time noise data
    % make directory for normally distributed noise
    norm_path = strcat('./fake_data/normal/plan',num2str(opts.plan));
    mkdir(norm_path);
    
    % normal noise data
    for i = 1:opts.trial
        % normally distributed noise 
        t_n = t + opts.time_step*randn(1,size(t,2)) - .5*opts.time_step;
        [x_n, y_n, z_n] = func_struct(opts.theta, opts.order, t_n);
        p_n = opts.plan*ones(1, size(t_n,2));
        data_n = [x_n; y_n; z_n; p_n];
        save(strcat(norm_path, '/', num2str(i), '.mat'), 'data_n');
    end

    % make directory for uniformly distributed noise
    uniform_path = strcat('./fake_data/uniform/plan',num2str(opts.plan));
    mkdir(uniform_path);
    
    % uniform noise data
    for i = 1:opts.trial
        % uniformly distributed noise 
        t_u = t + opts.time_step*rand(1,size(t,2)) - .5*opts.time_step;
        [x_u, y_u, z_u] = func_struct(opts.theta, opts.order, t_u);
        p_u = opts.plan*ones(1, size(t_u,2));
        data_u = [x_u; y_u; z_u; p_u];
        save(strcat(uniform_path, '/', num2str(i), '.mat'), 'data_u');
    end
end

    
function [x, y, z] = func_struct(theta, order, t)
    x = 0;
    y = 0;
    z = 0;
    for i = 1:order
        x = x + theta(1,order - i + 1) * t.^i;
        y = y + theta(2,order - i + 1) * t.^i;
        z = z + theta(3,order - i + 1) * t.^i;
    end
end

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
    ip.addParameter('time_noise',           1,           @isscalar);
    ip.addParameter('axis_noise',           0,           @isscalar);
    ip.addParameter('time_vary',            1,           @isscalar);
    ip.addParameter('sample_num',         100,           @isscalar);

    ip.parse(theta, order, varargin{:});
    opts = ip.Results;
    
%% define time varying model
    % sample time 
    t = 0:opts.time_step:opts.time;
    
    % generate ground truth trajectory
    if opts.time_vary == 1
        [x_gt, y_gt, z_gt] = func_struct_time_varying(opts.theta, opts.order, t);
    else
        [x_gt, y_gt, z_gt] = func_struct_time_invar(opts.theta, opts.order, ...
                                                    opts.sample_num, 1, 1, 1);
    end
   
    p_gt = opts.plan*ones(1, size(x_gt,2));
    sample = [x_gt; y_gt; z_gt; p_gt]
%     plot3(x_gt, y_gt , z_gt, 'o-');
    
    % save ground truth 
    gt_path = strcat('./fake_data/data_ground_truth/plan',num2str(opts.plan));
    mkdir(gt_path);
    save(strcat(gt_path,'/gt.mat'), 'sample');   
    
%% generate time noise data
    if opts.time_vary == 1 && opts.time_noise == 1 && opts.axis_noise == 0
        fake_data_time_noise(t, opts);
    end
    
%% generate axis noise data
    if opts.time_vary == 0 && opts.time_noise == 0 && opts.axis_noise == 1
        fake_data_axis_noise(t, opts);
    end
    
end
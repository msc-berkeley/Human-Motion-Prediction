function fake_data_axis_noise(t,opts)
% gernate fake data axis noise data
    % make directory for normally distributed noise
    norm_path = strcat('./fake_data/data_axis_noise_normal/plan',num2str(opts.plan));
    mkdir(norm_path);
    
    % normal noise data (N(0,1^2))
    for i = 1:opts.trial
        % normally distributed noise 
        [x_n, y_n, z_n] = func_struct_time_invar(opts.theta, opts.order, ...
                                                    opts.sample_num, 1, 1, 1);
        x_n = x_n + 0.005*randn(1, size(x_n,2));
        y_n = y_n + 0.005*randn(1, size(y_n,2));
        z_n = z_n + 0.005*randn(1, size(z_n,2));
        p_n = opts.plan*ones(1, size(x_n,2));
        sample = [x_n; y_n; z_n; p_n];
        save(strcat(norm_path, '/', num2str(i), '.mat'), 'sample');
    end

    % make directory for uniformly distributed noise
    uniform_path = strcat('./fake_data/data_axis_noise_uniform/plan',num2str(opts.plan));
    mkdir(uniform_path);
    
    % uniform noise data [-0.5,0.5]
    for i = 1:opts.trial
        % uniformly distributed noise 
        [x_u, y_u, z_u] = func_struct_time_invar(opts.theta, opts.order, ...
                                                    opts.sample_num, 1, 1, 1);
        x_u = x_u + 0.01*rand(1, size(x_u,2)) - 0.5;
        y_u = y_u + 0.01*rand(1, size(y_u,2)) - 0.5;
        z_u = z_u + 0.01*rand(1, size(z_u,2)) - 0.5;
        p_u = opts.plan*ones(1, size(x_u,2));
        sample = [x_u; y_u; z_u; p_u];
        save(strcat(uniform_path, '/', num2str(i), '.mat'), 'sample');
    end
end
function fake_data_time_noise(t,opts)
% gernate fake data time noise data
    % make directory for normally distributed noise
    norm_path = strcat('./fake_data/data_time_noise_normal/plan',num2str(opts.plan));
    mkdir(norm_path);
    
    % normal noise data
    for i = 1:opts.trial
        % normally distributed noise 
        t_n = t + opts.time_step*randn(1,size(t,2)) - .5*opts.time_step;
        [x_n, y_n, z_n] = func_struct_time_varying(opts.theta, opts.order, t_n);
        p_n = opts.plan*ones(1, size(t_n,2));
        sample = [x_n; y_n; z_n; p_n];
        save(strcat(norm_path, '/', num2str(i), '.mat'), 'sample');
    end

    % make directory for uniformly distributed noise
    uniform_path = strcat('./fake_data/data_time_noise_uniform/plan',num2str(opts.plan));
    mkdir(uniform_path);
    
    % uniform noise data
    for i = 1:opts.trial
        % uniformly distributed noise 
        t_u = t + opts.time_step*rand(1,size(t,2)) - .5*opts.time_step;
        [x_u, y_u, z_u] = func_struct_time_varying(opts.theta, opts.order, t_u);
        p_u = opts.plan*ones(1, size(t_u,2));
        sample = [x_u; y_u; z_u; p_u];
        save(strcat(uniform_path, '/', num2str(i), '.mat'), 'sample');
    end
end
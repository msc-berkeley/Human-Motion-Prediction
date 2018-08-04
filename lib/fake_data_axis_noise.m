function fake_data_axis_noise(t,opts)
% gernate fake data axis noise data

    % make directory for normally distributed noise
    norm_path = strcat('./fake_data/data_axis_noise_normal/plan',num2str(opts.plan));
    mkdir(norm_path);
    
    % normal noise data (N(0,1^2))
    for i = 1:opts.trial
        % normally distributed noise 
        [x_n, y_n, z_n] = func_struct(opts.theta, opts.order, t);
        x_n = x_n + randn(1, size(t,2));
        y_n = y_n + randn(1, size(t,2));
        z_n = z_n + randn(1, size(t,2));
        p_n = opts.plan*ones(1, size(t,2));
        sample = [x_n; y_n; z_n; p_n];
        save(strcat(norm_path, '/', num2str(i), '.mat'), 'sample');
    end

    % make directory for uniformly distributed noise
    uniform_path = strcat('./fake_data/data_axis_noise_uniform/plan',num2str(opts.plan));
    mkdir(uniform_path);
    
    % uniform noise data [-0.5,0.5]
    for i = 1:opts.trial
        % uniformly distributed noise 
        [x_u, y_u, z_u] = func_struct(opts.theta, opts.order, t);
        x_n = x_n + rand(1, size(t,2)) - 0.5;
        y_n = y_n + rand(1, size(t,2)) - 0.5;
        z_n = z_n + rand(1, size(t,2)) - 0.5;
        p_u = opts.plan*ones(1, size(t,2));
        sample = [x_u; y_u; z_u; p_u];
        save(strcat(uniform_path, '/', num2str(i), '.mat'), 'sample');
    end
end
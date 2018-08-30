function [x ,y ,z] = func_struct_time_invar(theta, order, num, x0, y0, z0)
% time invariant system motion data 
    x = [];
    y = [];
    z = [];
    % pre-stored value
    x_pre = x0;
    y_pre = y0;
    z_pre = z0;
    for i = 1:num
        % tmp-stored value
        x_tmp = 0;
        y_tmp = 0;
        z_tmp = 0;
        for j = 1:order
            x_tmp = x_tmp + theta(1,order - j + 1) * x_pre.^j;
            y_tmp = y_tmp + theta(2,order - j + 1) * y_pre.^j;
            z_tmp = z_tmp + theta(3,order - j + 1) * z_pre.^j;
        end
        x_pre = x_tmp;
        y_pre = y_tmp;
        z_pre = z_tmp;
        x = [x x_pre];
        y = [y y_pre];
        z = [z z_pre];
    end
end
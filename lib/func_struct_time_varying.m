function [x, y, z] = func_struct_time_varying(theta, order, t)
% time varying system motion data 
    x = 0;
    y = 0;
    z = 0;
    for i = 1:order
        x = x + theta(1,order - i + 1) * t.^i;
        y = y + theta(2,order - i + 1) * t.^i;
        z = z + theta(3,order - i + 1) * t.^i;
    end
end

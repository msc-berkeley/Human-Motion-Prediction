function grad = central_grad_id(var_value, h)
% central difference implementation (accelerated)
% note that the function is 
% approximate for MATLAB gradient operation
% input: 
% f         = function with respect to variable: var
% var       = symbolic variable 
% var_value = value for symbolic variable 
% output:
% approximate central difference
%     f1 = subs(f, var, var_value - 0.5*h);
%     f2 = subs(f, var, var_value + 0.5*h);
%     grad = eval((f2 - f1) / h);
    f1 = 1/(1 + exp(-(var_value - 0.5*h)));
    f2 = 1/(1 + exp(-(var_value + 0.5*h)));
    grad = (f2 - f1)/h;
end

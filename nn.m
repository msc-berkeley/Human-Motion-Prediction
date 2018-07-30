%% NN function 
function x_t=nn(x,U,b1,W,b2)
    x_t = (x * U + b1) * W + b2;
end
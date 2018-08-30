%% class identifier-based algorithm
classdef id
   properties
      y_dim; % train instance 2nd dimension
      U; % first layer parameter
      W; % second layer parameter
      v; % Filippov generalized solution
      k; % learning gain
      alpha; % learning gain 
      gamma; % learning gain
      beta1; % learning gain
      time_step; % learning rate
      x_id; % estimation of initial state
      x0_tilde; % variance of initial state
      Gamma_W; % constant weighting matrix
      Gamma_Ux; % constant weighting matrix
      Gamma_Ug; % constant weighting matrix
      error = [];
      id_state = [];
   end
   
   % construct method
   methods
       function obj = id(y_dim, U, W, varargin)
%% inputs
           ip = inputParser;
           ip.addRequired('y_dim',                             @isscalar);
           ip.addRequired('U',                                 @ismatrix); 
           ip.addRequired('W',                                 @ismatrix);
           ip.addParameter('v',           zeros(1,9),          @isvector);
           ip.addParameter('k',           20,                  @isscalar);
           ip.addParameter('alpha',       5,                   @isscalar);
           ip.addParameter('gamma',       50,                  @isscalar);
           ip.addParameter('beta1',       1.25,                @isscalar);
           ip.addParameter('time_step',   0.03,                @isscalar);
           ip.addParameter('Gamma_W',     0.1*eye(40),         @ismatrix);
           ip.addParameter('Gamma_Ux',    0.2*eye(9),          @ismatrix);
           ip.addParameter('Gamma_Ug',    0.2*eye(1),          @ismatrix);

           ip.parse(y_dim, U, W, varargin{:});
           opts = ip.Results;
%% init
           obj.y_dim = opts.y_dim;
           obj.U = opts.U;
           obj.W = opts.W;
           obj.v = opts.v;
           obj.k = opts.k;
           obj.alpha = opts.alpha;
           obj.gamma = opts.gamma;
           obj.beta1 = opts.beta1;
           obj.time_step = opts.time_step;
           obj.Gamma_W = opts.Gamma_W;
           obj.Gamma_Ux = opts.Gamma_Ux;
           obj.Gamma_Ug = opts.Gamma_Ug;
           obj.x_id = zeros(1,obj.y_dim);
           obj.x0_tilde = zeros(1,obj.y_dim);
       end
   end
   
   % ordinary method
   methods
       function [x_deri, x_tilde, layer1] = id_x_tilde(obj, g, obs)
           s_hat = [obj.x_id, g, 1];
           layer1 = s_hat*obj.U;
           activate = arrayfun(@(x) 1/(1 + exp(-x)), layer1);
           layer2 = activate*obj.W;
           x_tilde = obs - obj.x_id;
           mu = obj.k*x_tilde - obj.k*obj.x0_tilde + obj.v; % RISE feedback
           x_deri = layer2 + mu;
       end
       
       function [x_deri,v_deri,W_deri,Ux_deri,Ug_deri] = id_update_direction(obj, x_deri, x_tilde, g_tilde, layer1)
           v_deri = (obj.k*obj.alpha + obj.gamma)*x_tilde + obj.beta1*sign(x_tilde);
           g_deri = (g_tilde)/obj.time_step;
%            syms symbo;           
%            sigma_deri_exp = gradient(1/(1 + exp(-symbo)), symbo);
%            sigma_deri = [];% derivative of the activation sigmoid function with respect to input layer1
%            for t = 1:size(layer1, 2)
%                sigma_deri = [sigma_deri eval(subs(sigma_deri_exp, symbo, layer1(t)))];
%            end

           % central approximate gradient 
           sigma_deri = [];
           for t = 1:size(layer1, 2)
               sigma_deri = [sigma_deri central_grad_id(layer1(t), 0.1)];
           end
           sigma_deri_M = []; % activation derivtion matrix
           for p = 1:size(sigma_deri, 2)
               sigma_deri_M = blkdiag(sigma_deri_M, sigma_deri(p)); 
           end
           U_x = obj.U(1:obj.y_dim,:);
           W_deri = obj.Gamma_W*sigma_deri_M*U_x'*x_deri'*x_tilde;
           Ux_deri = obj.Gamma_Ux*x_deri'*x_tilde*obj.W'*sigma_deri_M;
           Ug_deri = obj.Gamma_Ug*g_deri*x_tilde*obj.W'*sigma_deri_M;
       end
       
       function obj = id_update(obj,x_deri,v_deri,W_deri,Ux_deri,Ug_deri,x_tilde)
           p_detect = peak_detect(x_tilde,0.2);
           if p_detect == 0
               obj.x_id = obj.x_id + obj.time_step*x_deri;
               obj.id_state = [obj.id_state; obj.x_id];
               obj.v = obj.v + obj.time_step*v_deri;
               obj.W = obj.W + W_deri*obj.time_step;
               U_x = obj.U(1:obj.y_dim,:);
               U_g = obj.U(obj.y_dim + 1, :);
               U_x = U_x + Ux_deri*obj.time_step;
               U_g = U_g + Ug_deri*obj.time_step;
               obj.U = [U_x; U_g; obj.U(obj.y_dim + 2,:)];
           else
               obj.x_id = obj.x_id + obj.time_step*x_deri;
               obj.id_state = [obj.id_state; obj.x_id];
           end
       end
       
       function obj = id_process(obj, g, g_tilde, obs)
           [x_deri, x_tilde, layer1] = id_x_tilde(obj, g, obs);
           [x_deri,v_deri,W_deri,Ux_deri,Ug_deri] = id_update_direction(obj, x_deri, x_tilde, g_tilde, layer1);
           obj = id_update(obj,x_deri,v_deri,W_deri,Ux_deri,Ug_deri,x_tilde);
           obj.error = [obj.error;x_tilde];
       end
   end
end % classdef
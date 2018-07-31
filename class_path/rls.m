%% class RLS-PAA
classdef rls
   properties
      W; 
      lambda;
      Xx;
      theta = [];
      F = [];
      X_theta = [];
      F_M = [];
      error = [];
   end
   
   % construct method
   methods
       function obj = rls(num, nn_dim, y_dim, varargin)
%% inputs
           ip = inputParser;
           ip.addRequired('num',                               @isscalar);
           ip.addRequired('nn_dim',                            @isscalar);
           ip.addRequired('y_dim',                             @isscalar);  
           ip.addParameter('W',                0,              @isscalar);
           ip.addParameter('lambda',           1,              @isscalar);
           ip.addParameter('F_M',              0,              @ismatrix);
           ip.addParameter('X_theta',          0,              @ismatrix);
           ip.addParameter('theta',            0,              @ismatrix);

           ip.parse(num, nn_dim, y_dim, varargin{:});
           opts = ip.Results;
%% init
           obj.W = opts.W;
           obj.lambda = opts.lambda;
           obj.error = 100*ones(num, y_dim);
           obj.F = 10000*eye(nn_dim);
           if opts.theta == 0
               obj.theta = zeros(nn_dim,y_dim);
           else
               obj.theta = opts.theta;
           end
           
           if opts.X_theta == 0
               obj.X_theta = rand(nn_dim*y_dim,nn_dim*y_dim);
           else
               obj.X_theta = opts.X_theta;
           end
               
           if opts.F_M == 0
               obj.F_M = obj.F;
               for k = 1:8
                   obj.F_M = blkdiag(obj.F_M, obj.F);
               end
           else
               obj.F_M = opts.F_M;
           end
       end
   end
   
% ordinary method
   methods
       function obj = rls_update(obj, phi, i, y_dim, obs_Y)
           for j = 1:y_dim
               obj.F = obj.F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j);
               k = obj.F*phi'/(obj.lambda + phi*obj.F*phi');
               obj.theta(:,j) =  obj.theta(:,j) + k*(obs_Y(i,j) - phi*obj.theta(:,j));
               obj.F = (obj.F - k*phi*obj.F)/obj.lambda;
               obj.F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j) = obj.F;
               obj.error(i,j) = obs_Y(i,j) - phi*obj.theta(:,j);
           end
% calculate variance of states
           Phi = phi;
           for k = 1:8
              Phi = blkdiag(Phi, phi);
           end
           obj.Xx = Phi*obj.X_theta*Phi' + obj.W;
           obj.X_theta = obj.F_M*Phi'*obj.Xx*Phi*obj.F_M -...
                         obj.X_theta*Phi'*Phi*obj.F_M -...
                         obj.F_M*Phi'*Phi*obj.X_theta + obj.X_theta;
       end
   end
end % classdef
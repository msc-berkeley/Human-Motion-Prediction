% ls algorithm, trainY_x1b1 x_t -> x_{t+1}
%% prepare training data
clc;
clear;
str = 'data2/';% with .4 .6 combination
load(strcat(str,'weights1.mat'));
load(strcat(str,'weights2.mat'));
load(strcat(str,'weights3.mat'));
load(strcat(str,'biases1.mat'));
load(strcat(str,'biases2.mat'));
load(strcat(str,'biases3.mat'));
load(strcat(str,'data_x1b1.mat'));
layer1 = max(0,trainX * double(weights1) + double(biases1));
layer2 = max(0,layer1 * double(weights2) + double(biases2));
encode = layer2;
num = size(encode, 1);
encode = [encode, ones(num,1)];
lambda=0.9998;
%% online ls
error_train = 100*ones(num, 9);
theta = zeros(41,9);
F = 10000*eye(41);
F_M = F;
for k = 1:8
    F_M = blkdiag(F_M, F);
end
j = 1;
E = ones(num,9);
% X_theta = zeros(41*9,41*9);
X_theta = rand(41*9,41*9);
W = .02^2; % for now
for i = 1:num
    disp(i);
    for j = 1:9
        F = F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j);
        phi = encode(i, :);
        k = F*phi'/(lambda+phi*F*phi');
        theta(:,j) =  theta(:,j) + k*(trainY_x1b1(i,j) - phi*theta(:,j));
        F = (F - k*phi*F)/lambda;
        F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j) = F;
        err = trainY_x1b1(:,j) - encode*theta(:,j);
        E(i,j)=norm(err,2);
        error_train(i, j) = trainY_x1b1(i,j) - phi*theta(:,j);
    end
    % calculate variance of states
    Phi = phi;
    for k = 1:8
        Phi = blkdiag(Phi, phi);
    end
    Xx = Phi*X_theta*Phi' + W;
    X_theta = F_M*Phi'*Xx*Phi*F_M - X_theta*Phi'*Phi*F_M - F_M*Phi'*Phi*X_theta + X_theta;
end

%% check the learning process
figure
save('error_ls_train_x1b1.mat','error_train');
plot_err(error_train, num, 'index', 'error_{train}_{ls}');

%% test
% layer1 = max(0,TestX * double(weights1) + double(biases1));
% layer2 = max(0,layer1 * double(weights2) + double(biases2));
% encode = layer2;
% num = size(encode,1);
% encode = [encode, ones(num,1)];
% E = []
% tsigma = [];
% error = 100*ones(num, 9);
% count = 0;
% for i = 1:num
%     for j = 1:9
%         F = F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j);
%         phi = encode(i, :);
%         k = F*phi'/(lambda+phi*F*phi');
%         theta(:,j) =  theta(:,j) + k*(TestY_x1b1(i,j) - phi*theta(:,j));
%         F = (F - k*phi*F)/lambda;
%         F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j) = F;
%         err = TestY_x1b1(i,j) - phi*theta(:,j);
% %         sigma = sqrt(Xx(1,1));
%         sigma = sqrt(Xx(j,j)); %we should make sure that each of the dimension matched to its corresponding standard deviation
%         nn = size(find(err>3*sigma | err<-3*sigma),1);
%         count = count + nn;
% %         E(i,j)=norm(err,2);
%         error(i, j) = TestY_x1b1(i,j) - phi*theta(:,j);
%     end
%     % calculate variance of states
%     Phi = phi;
%     for k = 1:8
%         Phi = blkdiag(Phi, phi);
%     end
%     Xx = Phi*X_theta*Phi' + W;
%     X_theta = F_M*Phi'*Xx*Phi*F_M - X_theta*Phi'*Phi*F_M - F_M*Phi'*Phi*X_theta + X_theta;
% end
% disp(1- count/num/9)

%% test with trained theta
F_M_ori = F_M;
theta_ori = theta;
layer1 = max(0,TestX * double(weights1) + double(biases1));
layer2 = max(0,layer1 * double(weights2) + double(biases2));
encode = layer2;
num = size(encode,1);
encode = [encode, ones(num,1)];
E = []
tsigma = [];
error = 100*ones(num, 9);
count = 0;
current = 0;
for c = 1:size(trial_test_num, 2)
    inst_num = trial_test_num(c);
    % recover theta
    F_M = F_M_ori;
    theta = theta_ori;
    for i = current + 1:current + inst_num
        for j = 1:9
            F = F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j);
            phi = encode(i, :);
            k = F*phi'/(lambda+phi*F*phi');
            theta(:,j) =  theta(:,j) + k*(TestY_x1b1(i,j) - phi*theta(:,j));
            F = (F - k*phi*F)/lambda;
            F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j) = F;
            err = TestY_x1b1(i,j) - phi*theta(:,j);
    %         sigma = sqrt(Xx(1,1));
            sigma = sqrt(Xx(j,j)); %we should make sure that each of the dimension matched to its corresponding standard deviation
            nn = size(find(err>3*sigma | err<-3*sigma),1);
            count = count + nn;
    %         E(i,j)=norm(err,2);
            error(i, j) = TestY_x1b1(i,j) - phi*theta(:,j);
        end
        % calculate variance of states
        Phi = phi;
        for k = 1:8
            Phi = blkdiag(Phi, phi);
        end
        Xx = Phi*X_theta*Phi' + W;
        X_theta = F_M*Phi'*Xx*Phi*F_M - X_theta*Phi'*Phi*F_M - F_M*Phi'*Phi*X_theta + X_theta;
    end
    current = current + inst_num;
end
disp(1- count/num/9)

%% count the 3 sigma error
figure
save('error_ls_test_x1b1_train_init.mat','error');
plot_err(error, num, 'index', 'error_{test}_{ls}');

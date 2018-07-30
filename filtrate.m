%% use 3 sigma rule to filtrate noisy sample points
% least square to get the layer3 parameters
%% prepare training data
clear;
str = 'data2/';% with .4 .6 combination
load(strcat(str,'weights1.mat'));
load(strcat(str,'weights2.mat'));
load(strcat(str,'weights3.mat'));
load(strcat(str,'biases1.mat'));
load(strcat(str,'biases2.mat'));
load(strcat(str,'biases3.mat'));
load(strcat(str,'data.mat'));
layer1 = max(0,trainX * double(weights1) + double(biases1));
layer2 = max(0,layer1 * double(weights2) + double(biases2));
encode = layer2;
num = size(encode, 1);
encode = [encode, ones(num,1)];
lambda=0.9998;
%% online ls 
%filtrate training data
train_filte = zeros(num, 1)
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
sigma = 0;
for i = 1:num
    for j = 1:9
        F = F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j);
        phi = encode(i, :);
        k = F*phi'/(lambda+phi*F*phi');
        theta(:,j) =  theta(:,j) + k*(trainY(i,j) - phi*theta(:,j));
        F = (F - k*phi*F)/lambda;
        F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j) = F;
        err = trainY(:,j) - encode*theta(:,j);
        err_current = trainY(i,j) - encode(i,:)*theta(:,j)
        %%%filte train
        if i > 2
         sigma = sqrt(Xx(j,j))
         if (err_current < -3*sigma || err_current > 3*sigma)
             train_filte(i) = 1;
         end
        end
        %%%
        E(i,j)=norm(err,2);
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
plot(1:num, E(:,4), '*-')       
for j = 1:9
    err_NN(j) = norm(double( trainY(:,j) - layer2*weights3(:,j) - biases3(j)),2);
end
%% test
% filte test data
layer1 = max(0,TestX * double(weights1) + double(biases1));
layer2 = max(0,layer1 * double(weights2) + double(biases2));
encode = layer2;
num = size(encode,1);
test_filte = zeros(num,1);
encode = [encode, ones(num,1)];
E = []
tsigma = [];
error = 100*ones(num, 9);
count = 0;
for i = 1:num
    for j = 1:9
        F = F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j);
        phi = encode(i, :);
        k = F*phi'/(lambda+phi*F*phi');
        theta(:,j) =  theta(:,j) + k*(TestY(i,j) - phi*theta(:,j));
        F = (F - k*phi*F)/lambda;
        F_M(41*(j-1)+1:41*j, 41*(j-1)+1:41*j) = F;
        err = TestY(i,j) - phi*theta(:,j);
        %%%filte test
        sigma = sqrt(Xx(j,j))
        if (err < -3*sigma || err > 3*sigma)
            test_filte(i) = 1;
        end
        %%%
        error(i, j) = TestY(i,j) - phi*theta(:,j);
    end
    % calculate variance of states
    Phi = phi;
    for k = 1:8
        Phi = blkdiag(Phi, phi);
    end
    Xx = Phi*X_theta*Phi' + W;
    X_theta = F_M*Phi'*Xx*Phi*F_M - X_theta*Phi'*Phi*F_M - F_M*Phi'*Phi*X_theta + X_theta;
end
%% count the 3 sigma error
figure
plot(1:num, error(:,1),1:num, error(:,2),1:num, error(:,3),1:num, error(:,4),...
    1:num, error(:,5),1:num, error(:,6),1:num, error(:,7),1:num, error(:,8),...
    1:num, error(:,9))

legend('x in k+1', 'y in k+1', 'z in k+1',...
    'x in k+2', 'y in k+2', 'z in k+2',...
    'x in k+3', 'y in k+3', 'z in k+3');
xlabel('index')
ylabel('error')
hold on

%% save the new training and testing files
new_trainX = [];
new_trainY = [];
new_testX = [];
new_testY = [];
num_train = size(trainX, 1);
for i = 1:num_train
    if train_filte(i) == 0
        new_trainX = [new_trainX; trainX(i, :)];
        new_trainY = [new_trainY; trainY(i, :)];
    end
end
num_test = size(TestX, 1);
for i = 1:num_test
    if test_filte(i) == 0
        new_testX = [new_testX; TestX(i, :)];
        new_testY = [new_testY; TestY(i, :)];
    end
end
save('new_testX.mat', 'new_testX');
save('new_testY.mat', 'new_testY');
save('new_trainX.mat', 'new_trainX');
save('new_trainY.mat', 'new_trainY');

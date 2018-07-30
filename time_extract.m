%% extract time step of id algorithm from Kinect data and reconstruct data
function [trainX_cell, trainY_cell, obs_p_cell, num_cell] = time_extract(time_train, trainX, trainY, plan)
    num_cell = size(time_train,2);
    trainX_cell = [];
    trainY_cell = [];
    obs_p_cell = [];
    current = 0;
    for i = 1:num_cell
        trial = time_train(i);
        trial = cell2mat(trial);
        num_spl = size(trial,1);
        num_inst = num_spl - 5;
        trainX_mat = [];
        trainY_mat = [];
        p_mat = [];
        for j = current + 1:current + num_inst
            trainX_mat = [trainX_mat;trainX(j,:)];
            trainY_mat = [trainY_mat;trainY(j,:)];
            p_mat = [p_mat;plan(j,:)];
        end
        current = current + num_inst;
        trainX_cell = [trainX_cell;{trainX_mat}];
        trainY_cell = [trainY_cell;{trainY_mat}];
        obs_p_cell = [obs_p_cell;{p_mat}];
    end
end
% clc
% clear
% str = 'data2/';
% load([str 'data_time.mat']);
% % str = 'raw_data/';
% % load([str 'processed_data1.mat']);
% num_train = size(time_train, 2);
% time_stp_train = []
% total = 0;
% for i = 1:num_train
%     trial = time_train(i);
%     trial = cell2mat(trial);
%     num_trial = size(trial,1);
%     if num_trial == 0
%         continue;
%     end
%     for j = 1:(num_trial - 6)
%         time_step = ((trial(j+1,5) - trial(j,5))*60 + trial(j+1,6)) - trial(j,6);
%         time_stp_train = [time_stp_train time_step];
%     end
%     time_stp_train = [time_stp_train time_step];
% end
% % instance_num = size(time_stp_train,2);
% % disp(instance_num)
% save('time_stp_test.mat','time_stp_train');
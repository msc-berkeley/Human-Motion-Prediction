function [trainX_cell, trainY_cell, obs_p_cell, num_cell] = time_extract(time_train, trainX, trainY, plan)
% extract time step of id algorithm from Kinect data and reconstruct data
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

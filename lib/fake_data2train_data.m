function fake_data2train_data()
% construct prediction used dataset from generated fake data
    path_root = './fake_data';
    data_kind = dir([path_root '/data*']);
    plan_path = dir([path_root '/' data_kind(1).name '/plan*']);
    plan_num = length(plan_path);
    for i = 1:length(data_kind)
        % ground truth directory 
        if strcmp(data_kind(i).name, 'data_ground_truth')
            path_root1 = strcat(path_root, '/', data_kind(i).name);
            plan_path = dir([path_root1 '/plan*']);
            for j = 1:plan_num
                path_root2 = strcat(path_root1, '/', plan_path(j).name);
                path_data = dir([path_root2 '/' 'gt*']);
                path_root3 = strcat(path_root2, '/', path_data.name);
                load(path_root3);
                [trainX_gt, trainY_gt, train_time] = train_const(sample);
                save([path_root2 '/data.mat'], 'trainX_gt', 'trainY_gt');
            end
        % noise trial directory
        else
            path_root1 = strcat(path_root, '/', data_kind(i).name);
            plan_path = dir([path_root1 '/plan*']);

            % concatenate all plan data
            trainX = [];
            trainY = [];
            time_train = [];
            for j = 1:plan_num
                trainX_p = [];
                trainY_p = [];
                time_train_p = [];
                path_root2 = strcat(path_root1, '/', plan_path(j).name);
                path_data = dir([path_root2 '/' '*mat']);
                for k = 1:length(path_data)
                    path_root3 = strcat(path_root2, '/', path_data(k).name);
                    load(path_root3);
                    [trainX_cell, trainY_cell, train_time] = train_const(sample);

                    % concatenate train_cell data
                    for n = 1:size(trainX_cell,1)
                        trainX_p = [trainX_p; trainX_cell(n,:)];
                        trainY_p = [trainY_p; trainY_cell(n,:)];
                    end

                    % concatenate train_time cell
                    time_train_p = [time_train_p {train_time}];
                end
                save([path_root2 '/data_time_p.mat'], 'trainX_p', 'trainY_p', 'time_train_p');

                % concatenate train data per plan
                for m = 1:size(trainX_p,1)
                        trainX = [trainX; trainX_p(m,:)];
                        trainY = [trainY; trainY_p(m,:)];
                end

                % concatenate train time cell per plan trial
                for m = 1:size(time_train_p, 2)
                    time_train = [time_train time_train_p(m)];
                end
            end
            save([path_root1 '/data_time.mat'], 'trainX', 'trainY', 'time_train');
        end
    end
end

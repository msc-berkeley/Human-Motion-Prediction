function mocap_data_generation()
% generate training set from CMU motion captured data
    load('./data2/jump.mat')
    load('./data2/run.mat')
    load('./data2/walk.mat')
    num_jump = size(jump,1);
    num_walk = size(part,1);
    num_run = size(run,1);
    joint_num = [27];%[13,17,18,19,20,24,25,26,27];
    for j = 1:size(joint_num,2)
        disp('index is ..')
        disp(joint_num(j))
        for i = 1:num_jump
            data = jump{i};
            rwrist_x = data(joint_num(j)*3+1:31*3:end);
            rwrist_y = data(joint_num(j)*3+2:31*3:end);
            rwrist_z = data(joint_num(j)*3+3:31*3:end);
            rwrist = [rwrist_x;rwrist_y;rwrist_z];
            jump_wrist{i} = rwrist;
        end
        for i = 1:num_run
            data = run{i};
            rwrist_x = data(joint_num(j)*3+1:31*3:end);
            rwrist_y = data(joint_num(j)*3+2:31*3:end);
            rwrist_z = data(joint_num(j)*3+3:31*3:end);
            rwrist = [rwrist_x;rwrist_y;rwrist_z];
            run_wrist{i} = rwrist;
        end
        for i = 1:num_walk
            data = part{i};
            rwrist_x = data(joint_num(j)*3+1:31*3:end);
            rwrist_y = data(joint_num(j)*3+2:31*3:end);
            rwrist_z = data(joint_num(j)*3+3:31*3:end);
            rwrist = [rwrist_x;rwrist_y;rwrist_z];
            walk_wrist{i} = rwrist;
        end
        %% smooth and generate trainable data
        jump_smooth = smooth(jump_wrist);
        run_smooth = smooth(run_wrist);
        walk_smooth = smooth(walk_wrist);
        shuffle = 0;
        [time_train, trainX, trainY, TestX, TestY] = get_training_data_1(jump_smooth, run_smooth,walk_smooth, 3, 3,shuffle);
        save('./data2/cmu.mat','time_train','trainX','trainY');
    end 
end
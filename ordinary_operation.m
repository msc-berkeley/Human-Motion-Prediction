%% ordinary operation (if you want to test your thought, this is a toy file)
% load('data2/data_time.mat');
% 
% num_cell = size(time_test,2);
% trial_test_num = [];
% for i = 1:num_cell
%     trial = time_test(i);
%     trial = cell2mat(trial);
%     num_spl = size(trial,1);
%     if num_spl == 0
%         continue
%     end
%     num_inst = num_spl - 5;
%     trial_test_num = [trial_test_num num_inst];
% end

% TestY_x1b1 = [];
% num_cell = size(time_test,2);
% current = 0;
% for i = 1:num_cell
%     trial = time_test(i);
%     trial = cell2mat(trial);
%     num_spl = size(trial,1);
%     if num_spl == 0
%         continue
%     end
%     num_inst = num_spl - 5;
%     for j = current + 1:current + num_inst - 1
%         TestY_x1b1 = [TestY_x1b1;TestX(j+1,1:9)];
%     end
%     TestY_x1b1 = [TestY_x1b1;TestY(current + num_inst - 2,:)];
%     current = current + num_inst;
% end



% num = size(TestX, 1);
% TestY_x1b1 = []
% for i = 1:num - 1
%     TestY_x1b1 = [TestY_x1b1;TestX(i+1,1:9)];
% end
% TestY_x1b1 = [TestY_x1b1; TestY(num-2,:)];
% save('data_x1b1.mat','TestX','TestY_x1b1','trainX','trainY_x1b1','trial_test_num');




% test obs_cell num
% for i = 1:10000000000
%     x = i;
% end
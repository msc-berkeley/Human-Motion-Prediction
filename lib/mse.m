function mse()
% calculate MSE of prediction error 
    path = './paper_result/';
    p1 = 'id/';
    p2 = 'rls/';
    load(strcat(path, p1, 'id_fk_axis_uni_error.mat'));
    load(strcat(path, p2, 'rls_fk_axis_uni_error.mat'));
    mse_rls = [];
    mse_id = [];
    for i = 1:9
        mse_rls_tmp = mse_cal(error(:,i));
        mse_id_tmp = mse_cal(error_train(:,i));
        mse_rls = [mse_rls mse_rls_tmp];
        mse_id = [mse_id mse_id_tmp];
    end
    % mean MSE of x,y,z dim 
    x_rls = (mse_rls(1) + mse_rls(4) + mse_rls(7)) / 3;
    y_rls = (mse_rls(2) + mse_rls(5) + mse_rls(8)) / 3;
    z_rls = (mse_rls(3) + mse_rls(6) + mse_rls(9)) / 3;
    x_id = (mse_id(1) + mse_id(4) + mse_id(7)) / 3;
    y_id = (mse_id(2) + mse_id(5) + mse_id(8)) / 3;
    z_id = (mse_id(3) + mse_id(6) + mse_id(9)) / 3;
    mse_rls = [x_rls y_rls z_rls];
    mse_id = [x_id y_id z_id];
    disp(mse_rls);
    disp(mse_id);
end

function mse = mse_cal(error)
    num = size(error,1);
    sigma = 0;
    for i = 1:num
        sigma = sigma + error(i,1)^2;
    end
    mse = sigma / num;
end
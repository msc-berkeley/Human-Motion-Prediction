function error_ori = modi_order(error)
% modify error order to original: 
 num_order = size(error,1);
    error_ori = zeros(num_order, 9);
    j = 1;
    for start_order = 1:3
        for i = start_order:3:num_order
         error_ori(i,:) = error(j,:);
         j = j + 1;
        end
    end
end
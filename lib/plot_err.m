%% plot error
function plot_err(error_id, num, x_label, y_label)
    plot(1:num, error_id(:,1),1:num, error_id(:,2),1:num, error_id(:,3),1:num, error_id(:,4),...
        1:num, error_id(:,5),1:num, error_id(:,6),1:num, error_id(:,7),1:num, error_id(:,8),...
        1:num, error_id(:,9))
%     plot(1:num, error_id(:,1), 1:num, error_id(:,2), 1:num, error_id(:,3))
%     plot(1:num, error_id(:,1));
%     ylim([-0.35,0.35]);
%     xlim([0,384]);
    legend('x in k+1', 'y in k+1', 'z in k+1',...
        'x in k+2', 'y in k+2', 'z in k+2',...
        'x in k+3', 'y in k+3', 'z in k+3');
    xlabel(x_label)
    ylabel(y_label)
    
end
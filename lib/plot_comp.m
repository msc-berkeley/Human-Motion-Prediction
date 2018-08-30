function plot_comp(rls, id, num, x_label, y_label, order)
% plot_paper implementation
% compare perdiction errors of ID and RLS
% id is red system and rls is blue system
    % x axis
    plot(1:num, id(:,(order-1)*3+1), 'Color',[0.6 0.2 0])
    hold on
    
    % y axis 
    plot(1:num, id(:,(order-1)*3+2), 'Color',[0.8 0.2 0])
    hold on
    
     % z axis 
    plot(1:num, id(:,(order-1)*3+3), 'Color',[1 0.2 0])
    hold on
    
    % x axis
    plot(1:num, rls(:,(order-1)*3+1), 'Color',[0 0.2 0.6])
    hold on 
    
    % y axis 
    plot(1:num, rls(:,(order-1)*3+2), 'Color',[0 0.2 0.8])
    hold on 
    
    % z axis 
    plot(1:num, rls(:,(order-1)*3+3), 'Color',[0 0.2 1])
    hold on 
    
    
    lgd = legend(['x_{id} in k+' num2str(order)],['y_{id} in k+' num2str(order)], ...
           ['z_{id} in k+' num2str(order)],['x_{rls} in k+' num2str(order)], ...
           ['y_{rls} in k+' num2str(order)],['z_{rls} in k+' num2str(order)]);
    lgd.FontSize = 10;
    lgd.Location = 'southeast';
%     xlim([4800,5200]);
    ylim([-0.3, 0.1]);
    xlabel(x_label,'FontSize',15);
    ylabel(y_label,'FontSize',15);
    
    set(gca,'FontSize',15)
    
end
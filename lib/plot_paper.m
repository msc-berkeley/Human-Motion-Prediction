function plot_paper()
% plot paper graph using matlab2tikz, please uncomment the last part
% subsitution using MATLAB plot 
% saved as PDF
    path = './paper_result/';
    p1 = 'id/';
    p2 = 'rls/';
    
    load(strcat(path, p1, 'id_fk_axis_uni_error.mat'));
    load(strcat(path, p2, 'rls_fk_axis_uni_error.mat'));
    num = size(error_train, 1);
    h = figure;
    plot_comp(error, error_train, num, 'index', 'TC prediction error', 3);
    set(h,'Units','Inches');
    pos = get(h,'Position');
    set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    saveas(gcf,'./human-motion-prediction/fk_axis_3.pdf');
    % cleanfigure('minimumPointsDistance', 0.1);
    % matlab2tikz('/Users/Caesar/Desktop/human-motion-prediction/fk_tm_1.tex', 'width', '\fwidth');
end
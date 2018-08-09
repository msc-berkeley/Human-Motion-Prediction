%% plot error spectrum
function plot_err_freq(error_id)
    figure; hold on;
    Fs = 30; % samples per second
    dF = Fs/size(error_id,1); % gaps between frequencies
    f = - Fs/2:dF:Fs/2-dF;
    for i = 1:size(error_id,2)
        plot(f, fftshift(fft(error_id(:,i))))
    end
    legend('x in k+1', 'y in k+1', 'z in k+1',...
        'x in k+2', 'y in k+2', 'z in k+2',...
        'x in k+3', 'y in k+3', 'z in k+3');
    xlabel('Frequncy (Hz)')
    ylabel('Magnitude (dB)')
end
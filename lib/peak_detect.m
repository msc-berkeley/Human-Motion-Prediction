function detect = peak_detect(x,th)
    detect = 0;
    for i = 1:9
        if abs(x(i)) > th
            detect = 1;
        end
    end
end
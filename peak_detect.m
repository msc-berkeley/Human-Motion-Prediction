function detect = peak_detect(x)
    detect = 0;
    for i = 1:9
        if abs(x(i)) > 0.2
            detect = 1;
        end
    end
end
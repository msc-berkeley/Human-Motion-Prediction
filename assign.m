%% assign values to symbolic matrix 
function X = assign(X,Y)
    row = size(X,1);
    col = size(X,2);
    for i = 1:row
        for j = 1:col
            X(i,j) = Y(1,j);
        end
    end
end

function trainX = concatenate(trainX, cell)
% concatenate new gained training data
    row = size(cell,1);
    for i = 1:row
        trainX = [trainX;cell(i,:)];
    end
end
% construct training instance from observation 
function [trainX, trainY, train_time] = train_const(sample)
    trainX = [];
    trainY = [];
    train_time = [];
    for t = 1:size(sample,2) - 5
        tmpX = [sample(1:3,t)' sample(1:3,t+1)' sample(1:3,t+2)' sample(4,t) 1];
        tmpY = [sample(1:3,t+3)' sample(1:3,t+4)' sample(1:3,t+5)'];
        trainX = [trainX; tmpX];
        trainY = [trainY; tmpY];
    end
    for t1 = 1:size(sample,2)
        train_time = [train_time; t1];
    end
end
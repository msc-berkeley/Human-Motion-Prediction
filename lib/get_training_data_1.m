function [time_train, trainX, trainY, testX, testY] = get_training_data_1(Pos1, Pos2, Pos3, m, k, shuffle)
%     m = 3; % past
%     k = 3; % future
    [~,P] = size(Pos1);
    [~,N] = size(Pos2);
    [~,K] = size(Pos3);
    trainX = [];
    trainY = [];
    time_train = [];
    testX = [];
    testY = [];
    %% pos1
    for j = int16(P/5):P
        pos = Pos1{j};
        if pos
            [~,len] = size(pos);
            for i = 1:len-m-k+1
                x = pos(:,i:i+m-1);
                y = pos(:,i+m:i+m+k-1);
                trainX = [trainX; reshape(x,[1,3*m]),1, 1];
                trainY = [trainY; reshape(y, [1,3*k])];
            end 
            % add sample record
            t_cell = [];
            for t = 1:len-m-k+6
                t_cell = [t_cell; t];
            end
        end
        time_train = [time_train {t_cell}];
    end
    %% pos2
    for j = int16(N/5):N
        pos = Pos2{j};
        if pos
            [~,len] = size(pos);
            for i = 1:len-m-k+1
                x = pos(:,i:i+m-1);
                y = pos(:,i+m:i+m+k-1);
                trainX = [trainX; reshape(x,[1,3*m]),2,1];
                trainY = [trainY; reshape(y, [1,3*k])];
            end   
            % add sample record
            t_cell = [];
            for t = 1:len-m-k+6
                t_cell = [t_cell; t];
            end
        end
        time_train = [time_train {t_cell}];
    end
    
    %% pos3
     for j = int16(K/5):K
        pos = Pos3{j};
        if pos
            [~,len] = size(pos);
            for i = 1:len-m-k+1
                x = pos(:,i:i+m-1);
                y = pos(:,i+m:i+m+k-1);
                trainX = [trainX; reshape(x,[1,3*m]),3,1];
                trainY = [trainY; reshape(y, [1,3*k])];
            end   
            % add sample record
            t_cell = [];
            for t = 1:len-m-k+6
                t_cell = [t_cell; t];
            end
        end
        time_train = [time_train {t_cell}];
    end  
    %% test
    for j = 1:int16(P/5)
        pos = Pos1{j};
        if pos
            [~,len] = size(pos);
            for i = 1:len-m-k+1
                x = pos(:,i:i+m-1);
                y = pos(:,i+m:i+m+k-1);
                testX = [testX; reshape(x,[1,3*m]),1,1];
                testY = [testY; reshape(y, [1,3*k])];
            end   
        end
    end
        %% pos2
        for j = 1:int16(N/5)
            pos = Pos2{j};
            if pos
                [~,len] = size(pos);
                for i = 1:len-m-k+1
                    x = pos(:,i:i+m-1);
                    y = pos(:,i+m:i+m+k-1);
                    testX = [testX; reshape(x,[1,3*m]),2,1];
                    testY = [testY; reshape(y, [1,3*k])];
                end   
            end
        end

    %% pos3
    for j = 1:int16(K/5)
        pos = Pos3{j};
        if pos
            [~,len] = size(pos);
            for i = 1:len-m-k+1
                x = pos(:,i:i+m-1);
                y = pos(:,i+m:i+m+k-1);
                testX = [testX; reshape(x,[1,3*m]),3,1];
                testY = [testY; reshape(y, [1,3*k])];
            end   
        end
    end      
    %%
    if shuffle == 1
        num = size(trainX,1);
        idx = randperm(num);
        trainX_shuffle = trainX;
        trainY_shuffle = trainY;
        trainX_shuffle(1:num,:) = trainX(idx,:);
        trainY_shuffle(1:num,:) = trainY(idx,:);
        trainX = trainX_shuffle;
        trainY = trainY_shuffle;
    end
end

function Pos_s = smooth(Pos1)
    [~,K] = size(Pos1);
    for i = 1:K
        if Pos1{i}
            pos = Pos1{i};
            [~,N] = size(pos);
            pos_s = zeros(3,N);
            pos_s(:,1) = pos(1:3,1);
            pos_s(:,2:end) = .4*pos(1:3,2:end) + .6*pos(1:3,1:end-1);
            Pos_s{i} = pos_s;
        end
    end
end
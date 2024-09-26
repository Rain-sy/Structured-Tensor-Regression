function [X, Y, B] = generate_slice_lowrank(d, s, r, n,sigma)
    
    B = zeros(d, d, s);
    for i = 1:s
        U = rand(d, r); % 左因子
        V = rand(r, d); % 右因子
        B(:, :, i) = U * V;
    end
    X = cell(1, n);
    for i = 1:n
        X{i} = rand(d, d, s);
    end
    
    Y = zeros(n, 1);
    for i = 1:n
        Y_mean = 0;
        for j = 1:s
            Y_mean = Y_mean + trace(B(:, :, j) * X{i}(:, :, j)');
        end
        e = randn*sigma; 
        Y(i) = Y_mean + e;
    end

    %disp('generate ended')
end




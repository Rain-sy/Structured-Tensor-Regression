function [X, Y, B] = generate_slice_lowrank(d, s, r, n, sigma)
    B = zeros(d, d, s);
    for i = 1:s
        U = normrnd(0, 1, [d, r]); 
        V = rand(r, d); % 右因子，从正态分布生成
        B(:, :, i) = U * V; % 低秩张量 B 的生成
    end
    
    X = cell(1, n);
    for i = 1:n
        X{i} = rand(d,d,s); % 生成 X，正态分布 (0, 1)
    end
    
    Y = zeros(n, 1);
    for i = 1:n
        Y_mean = 0;
        for j = 1:s
            Y_mean = Y_mean + trace(B(:, :, j) * X{i}(:, :, j)');
        end
        e = normrnd(0, sigma); % 噪声，从正态分布 (0, sigma) 生成
        Y(i) = Y_mean + e; 
    end

    % disp('generate ended')
end

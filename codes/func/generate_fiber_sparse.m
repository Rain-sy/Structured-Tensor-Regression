function [X, Y, B] = generate_fiber_sparse(d, s, sparsity, n, sigma)
    % 随机选择要设置为零的 fibers
    zero_fibers = randperm(d, d-sparsity);
    
    % 初始化 B 为随机数
    B = normrnd(0, 1, [d, d, s]);
    
    % 将选定的 fibers 设置为 0
    for i = 1:d-sparsity
        B(zero_fibers(i), :, :) = 0;  % 设置整条 fiber 为0
    end
    
    % 生成随机输入 X
    X = cell(1, n);
    for i = 1:n
        X{i} = normrnd(0, 1, [d, d, s]); 
    end
    
    % 计算输出 Y
    Y = zeros(n, 1);
    for i = 1:n
        Y_mean = 0;
        for j = 1:s
            Y_mean = Y_mean + trace(B(:, :, j) * X{i}(:, :, j)'); 
        end
        e = normrnd(0, sigma);
        Y(i) = Y_mean + e; 
    end

    % disp('generate_fiber_sparse ended')
end

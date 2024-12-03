function [X, Y, B] = generate_slice_sparse(d, s, sparsity, n, sigma)
    zero_slices = randperm(s, s-sparsity);
    
    % 初始化 B 为随机数
    B = normrnd(0, 1, [d, d, s]);
    
    % 将选定的切片设置为 0
    for i = 1:s-sparsity
        B(:, :, zero_slices(i)) = 0;  % 设置整片为0
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

    % disp('generate ended')
end

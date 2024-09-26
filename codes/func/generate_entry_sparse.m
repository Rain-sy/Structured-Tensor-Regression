function [X, Y, B] = generate_entry_sparse(d, s, sparsity, n, sigma)
    total_elements = d * d * s;
    num_zero_elements = round(total_elements * sparsity);
    
    % 初始化 B 为随机数
    B = rand(d, d, s);
    
    % 随机选择要设置为 0 的位置
    zero_indices = randperm(total_elements, num_zero_elements);
    B(zero_indices) = 0; % 将这些位置设置为 0
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
        e = randn * sigma; 
        Y(i) = Y_mean + e; 
    end

    % disp('generate ended')
end
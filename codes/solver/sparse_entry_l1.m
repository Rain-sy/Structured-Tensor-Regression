function [A] = sparse_entry_l1(X, Y, size, n, lambda)
    % 使用 CVX 求解优化问题
    A = zeros(size);
    loss = zeros(n, 1);
    cvx_clear;
    
    cvx_begin
        variable A(size)  % 定义变量 A，大小为提供的 size 数组
        expressions loss(n)
        
        % 计算每个样本的损失
        for i = 1:n
            inner_product = 0;
            for j = 1:size(3)  % 假设第三维对应 s
                inner_product = inner_product + trace(A(:, :, j) * X{i}(:, :, j)');
            end
            loss(i) = (Y(i) - inner_product)^2;
        end
        
        disp('start minimize')
        % 定义优化目标函数，包含损失和 L1 范数正则化项
        minimize ( (1/(2*n)) * sum(loss) + lambda * sum(abs(A(:))) )
    cvx_end

end

function [A] = aos_lowrank(X, Y, size, n, lambda)
    % 使用 CVX 求解优化问题
    A = zeros(size);
    loss = zeros(n, 1);
    nuclear_norms = zeros(size(3), 1);  % Assuming the third dimension of size is similar to s
    cvx_clear;
    cvx_begin
        variable A(size)  % Define the variable A with the provided size array
        expressions loss(n) nuclear_norms(size(3))
        
        % 计算每个样本的损失
        for i = 1:n
            inner_product = 0;
            for j = 1:size(3)  % Assuming the third dimension corresponds to s
                inner_product = inner_product + trace(A(:, :, j) * X{i}(:, :, j)');
            end
            loss(i) = (Y(i) - inner_product)^2;
        end
        
        % 计算张量 A 的核范数正则化项
        for j = 1:size(3)
            nuclear_norms(j) = norm_nuc(A(:, :, j));
        end
        disp('start minimize')
        % 定义优化目标函数
        minimize ( (1/(2*n)) * sum(loss) + lambda * sum(nuclear_norms) )
    cvx_end

end

function A_opt = niAPG(Y, X_cell, n, lambda, l, eta, delta, L, K)
    % 输入:
    %   Y_cell - 包含n个样本的Y^(i)，大小为cell(1, n)，每个元素是d*d*s的三阶tensor
    %   X_cell - 包含n个样本的X^(i)，大小为cell(1, n)，每个元素是d*d*s的三阶tensor
    %   lambda - 正则化参数
    %   eta - 步长参数
    %   delta - 步长参数
    %   L - Lipschitz常数
    %   K - 最大迭代次数
    %   q - 用于计算Delta_k的窗口大小
    q = 5 ;
    [d1, d2, s] = size(X_cell{1}); % 假设每个张量的维度相同
    A = cell(1, K); % 创建一个cell数组保存每个A_k的值
    A{1} = zeros(d1, d2, s); % 初始A_1为随机张量
    
    for k = 1:K-1
        if k == 1
            y_k = A{k};
        else
            y_k = A{k} + (k - 1) / (k + 2) * (A{k} - A{k-1});
        end
        
        F_values = zeros(1, min(q, k));
        for t = max(1, k-q):k
            F_values(t - max(1, k-q) + 1) = objective_function(Y, X_cell, A{t}, lambda,l, n);
        end
        Delta_k = max(F_values);

        if objective_function(Y, X_cell, y_k, lambda,l, n) <= Delta_k
            v_k = y_k;
        else
            v_k = A{k};
        end

        grad_v_k = gradient_function(Y, X_cell, v_k, n)
        z_k = v_k - eta * grad_v_k;

        % 更新 A_{k+1}，执行prox运算
        A{k+1} = prox_operator(z_k, lambda, l, eta, [d1, d2, s]);
        
        % 可选：输出当前的均方误差
        Y_pred = zeros(n, 1);
        for i = 1:n
            Y_pred(i) = innerprod(tensor(A{k+1}), tensor(X_cell{i}));
        end
        mse = mean((Y - Y_pred).^2);
        fprintf('Iteration %d, MSE: %4f\n', k, mse);
    end
    
    A_opt = A{K};
end

function F_val = objective_function(Y, X_cell, A, lambda,l, n)
    % 计算目标函数值
    residual = 0;
    for i = 1:n
        residual = residual + (Y(i) - innerprod(tensor(X_cell{i}), tensor(A)))^2;
    end
    F_val = (1/(2*n)) * residual + regularization_term(A, lambda,l);
end

function grad = gradient_function(Y, X_cell, A, n)
    % 计算梯度
    grad = zeros(size(A));
    for i = 1:n
        grad = grad + (innerprod(tensor(X_cell{i}), tensor(A)) - Y(i)) * X_cell{i};
    end
    grad = grad / n;
end

function A_next = prox_operator(Z, lambda, l, eta, original_size)
    % 计算proximal operator，涉及到核范数和SCAD惩罚
    A_next = zeros(size(Z));
    for k = 1:3
        Ak = mode_n_unfold(Z, k); % 使用新的 unfold 函数
        [U, S, V] = svd(Ak, 'econ');
        S_prox = scad_prox_operator(S, l, eta); % 使用 l 参数进行 SCAD 操作
        A_next = A_next + mode_n_fold(U * (S_prox + max(S - lambda * eta, 0)) * V', k, original_size); % 核范数正则化使用 lambda
    end
end

function S_prox = scad_prox_operator(S, l, eta)
    a = 3.7; % SCAD的常用参数
    singular_values = diag(S);
    prox_singular_values = zeros(size(singular_values));
    
    for i = 1:length(singular_values)
        sigma = singular_values(i);
        if abs(sigma) <= l * eta
            prox_singular_values(i) = max(0, abs(sigma) - l * eta) * sign(sigma);
        elseif abs(sigma) <= a * l * eta
            prox_singular_values(i) = ((a - 1) * sigma - sign(sigma) * a * l * eta) / (a - 2);
        else
            prox_singular_values(i) = sigma;
        end
    end
    
    S_prox = diag(prox_singular_values);
end


function reg = regularization_term(A, lambda, l)
    % 计算正则化项，包括核范数和SCAD惩罚
    reg = 0;
    for k = 1:3
        Ak = mode_n_unfold(A, k);
        [~, S, ~] = svd(Ak, 'econ');
        reg = reg + lambda * sum(diag(S)) + sum(scad_penalty(diag(S), l)); % 核范数使用 lambda，SCAD 使用 l
    end
end


function penalty = scad_penalty(sigma, l)
    a = 3.7;
    penalty = zeros(size(sigma));
    for i = 1:length(sigma)
        if sigma(i) <= l
            penalty(i) = l * sigma(i);
        elseif sigma(i) <= a * l
            penalty(i) = (-sigma(i)^2 + 2 * a * l * sigma(i) - l^2) / (2 * (a - 1));
        else
            penalty(i) = (a + 1) * l^2 / 2;
        end
    end
end


function unfolded = mode_n_unfold(A, n)
    % Unfold tensor A along the specified mode n
    dims = size(A);
    order = 1:length(dims);
    order([1 n]) = order([n 1]); % Move dimension n to the first position
    unfolded = reshape(permute(A, order), dims(n), []);
end

function folded = mode_n_fold(Ak_new, mode, original_size)
    % Fold the unfolded matrix back into a tensor
    dims = original_size;
    order = 1:length(dims);
    order([1 mode]) = order([mode 1]); % Move the first dimension back to mode
    folded = ipermute(reshape(Ak_new, dims(order)), order);
end


function A_opt = niAPG(Y, X, n, lambda, eta, K)
    % 输入:
    %   Y - 包含n个样本的Y^(i)，大小为cell(1, n)，每个元素是d*d*s的三阶tensor
    %   X - 包含n个样本的X^(i)，大小为cell(1, n)，每个元素是d*d*s的三阶tensor
    %   lambda - 正则化参数
    %   eta - 步长参数
    %   K - 最大迭代次数
    %   q - 用于计算Delta_k的窗口大小
    q = 5;
    [d1, d2, s] = size(X{1}); % 假设每个张量的维度相同
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
            F_values(t - max(1, k-q) + 1) = objective_function(Y, X, A{t}, lambda, n);
        end
        Delta_k = max(F_values);

        if objective_function(Y, X, y_k, lambda, n) <= Delta_k
            v_k = y_k;
        else
            v_k = A{k};
        end

        grad_v_k = gradient_function(Y, X, v_k, n);
        z_k = v_k - eta * grad_v_k;

        % 更新 A_{k+1}，执行prox运算
        A{k+1} = prox_operator(z_k, lambda, eta, [d1, d2, s]);
        
        % 可选：输出当前的均方误差
        Y_pred = zeros(n, 1);
        for i = 1:n
            Y_pred(i) = innerprod(tensor(A{k+1}), tensor(X{i}));
        end
        mse = mean((Y - Y_pred).^2);
        %fprintf('Iteration %d, MSE: %4f\n', k, mse);
    end
    
    A_opt = A{K};
end

function F_val = objective_function(Y, X, A, lambda, n)
    % 计算目标函数值
    loss = 0;
    for i = 1:n
        loss = loss + (Y(i) - innerprod(tensor(X{i}), tensor(A)))^2;
    end
    Ak = mode_n_unfold(A, 1);
    [~, S, ~] = svd(Ak, 'econ');
    a=3.7;
    reg = sum(scad(diag(S), lambda,a)); 
    F_val = (1/(2*n)) * loss + reg;
end

function grad = gradient_function(Y, X, A, n)
    grad = zeros(size(A));
    for i = 1:n
        grad = grad + (innerprod(tensor(X{i}), tensor(A)) - Y(i)) * X{i};
    end
    grad = grad / n;
end

function A_next = prox_operator(Z, lambda, eta, original_size)
    A_next = zeros(size(Z));
    Ak = mode_n_unfold(Z, 1); 
    [U, S, V] = svd(Ak, 'econ');
    diag_S = diag(S);
    for idx = 1:length(diag_S)
        diag_S(idx) = max((diag_S(idx)-lambda*eta),0);
    end
    A_next = A_next + mode_n_fold(U * diag(diag_S) * V', 1, original_size);
end

function penalty_single = scad(x, lambda, a)
    if abs(x) <= lambda
        penalty_single = lambda * abs(x);
    elseif abs(x) <= a * lambda
        penalty_single = (-x.^2 + 2 * a * lambda * abs(x) - lambda^2) / (2 * (a - 1));
    else
        penalty_single = (lambda^2 * (a^2 - 1)) / 2;
    end
end

% function penalty_derivative = scad_derivative(x, lambda, a)
%     if abs(x) <= lambda
%         penalty_derivative = lambda * sign(x);
%     elseif abs(x) <= a * lambda
%         penalty_derivative = (a * lambda - abs(x)) / (a - 1) * sign(x);
%     else
%         penalty_derivative = 0;
%     end
% end


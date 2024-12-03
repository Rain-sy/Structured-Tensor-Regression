function [A] = slice_solver(X, Y, siz, n, lambda, l)
    % Initialize A
    A_init = zeros(siz);
    
    % Define smoothF for the loss function
    smoothF = @(A) compute_slice_loss_and_gradient(A, X, Y, siz, n, lambda, l);

    % Set options for the solver
    opts.alg = 'AT';
    opts.tol = 1e-10;
    opts.maxITs = 3000;
    opts.stopFcn = @my_stopFcn; % Use the custom stop function
    
    % Solve using TFOCS
    A = tfocs(smoothF, [], [], A_init, opts);
end

function [loss, gradient] = compute_slice_loss_and_gradient(A, X, Y, siz, n, lambda, l)
    % Initialize loss, gradient, and nonsmooth_value
    loss = 0;
    gradient = zeros(siz);
    nonsmooth_value = 0;
    subgradient = zeros(siz);

    % Compute the smooth part: loss and gradient
    for i = 1:n
        % Extract the i-th slice of X and the corresponding A
        Xi = X{i};
        Yi = Y(i);
        
        % Compute the inner product and residual
        inner_product = sum(sum(sum(A .* Xi, 1), 2), 3); % Sum over slices
        residual = Yi - inner_product;
        loss = loss + residual^2;
        
        % Compute the gradient
        gradient = gradient - 2 * residual * Xi;
    end
    
    % Average the loss and gradient
    loss = loss / (2 * n);
    gradient = gradient / n;

    % Compute the nonsmooth part: nonsmooth_value and subgradient
    for j = 1:siz(3) % For each slice
        Ak = A(:, :, j);
        [U, S, V] = svd(Ak, 'econ');
        singular_values = diag(S);

        scad_penalty_value = sum(scad_penalty(singular_values, l));
        nonsmooth_value = nonsmooth_value + lambda* scad_penalty_value / siz(3);

        % Compute the subgradient for this slice
        S_subgrad = diag(scad_penalty_gradient(singular_values, l));
        Ak_subgrad = U * S_subgrad * V';
        subgradient(:, :, j) = Ak_subgrad;
    end

    % Combine the smooth and nonsmooth parts
    loss = loss + nonsmooth_value;
    gradient = gradient + subgradient;
end

% SCAD penalty and gradient functions
function penalty = scad_penalty(sigma, l)
    a = 3.7;
    penalty = arrayfun(@(x) scad_penalty_single(x, l, a), sigma);
end

function penalty_single = scad_penalty_single(x, l, a)
    if abs(x) <= l
        penalty_single = l * abs(x);
    elseif abs(x) <= a * l
        penalty_single = (-x^2 + 2 * a * l * abs(x) - l^2) / (2 * (a - 1));
    else
        penalty_single = (l^2 * (a^2 - 1)) / 2;
    end
end

function grad = scad_penalty_gradient(sigma, l)
    a = 3.7; % Common choice for a in SCAD
    grad = arrayfun(@(x) scad_gradient_single(x, l, a), sigma);
end

function grad_single = scad_gradient_single(x, l, a)
    if abs(x) <= l
        grad_single = l * sign(x);
    elseif abs(x) <= a * l
        grad_single = ((a * l - abs(x)) / (a - 1)) * sign(x);
    else
        grad_single = 0;
    end
end

% Custom stopping function
function stop = my_stopFcn(f, ~, ~)
    % Persistent variables to store the previous function values and iteration count
    persistent prev_f iter_count moving_avg moving_avg_window
    
    % Initialize persistent variables on the first call
    if isempty(prev_f)
        prev_f = f; % Set the initial function value
        iter_count = 0; % Initialize iteration count
        moving_avg = f; % Initialize moving average
        moving_avg_window = 25; % Set window size for moving average
        stop = false; % Do not stop on the first iteration
        return;
    end
    
    % Increment the iteration count
    iter_count = iter_count + 1;

    % Update the moving average with the new function value
    moving_avg = (moving_avg * (moving_avg_window - 1) + f) / moving_avg_window;

    % Check if the iteration count exceeds 100 before evaluating the stopping criteria
    if iter_count > 100
        % Calculate the difference between current and moving average
        diff = abs(f - moving_avg);

        % Check if the difference is below a threshold
        if diff < 1e-7
            stop = true; % Stop if the condition is met
        else
            stop = false; % Continue otherwise
        end
    else
        stop = false; % Continue if iteration count is 100 or less
    end
    
    % Update the previous function value
    prev_f = f;
end

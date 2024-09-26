function [A, out] = sdp_solver(X, Y, siz, n, reg)
    A_init = zeros(siz);
    
    % Start timing
    tic;
    smoothF = @(A) compute_loss_and_gradient(A, X, Y, siz, n, reg.lambda, reg.l);

    opts.alg = 'AT';
    opts.tol = 1e-10;
    if(isfield(reg, 'maxITs'))
        opts.maxITs = reg.maxITs;
    else
        opts.maxITs = 3000;
    end
    opts.stopFcn = @my_stopFcn; 
    
    A = tfocs(smoothF, [], [], A_init, opts);
    
    % Stop timing
    elapsedTime = toc;

    % Prepare the output structure
    out.time = elapsedTime;
end


function [loss, gradient] = compute_loss_and_gradient(A, X, Y, siz, n, lambda, l)
    loss = 0;
    nonsmooth_value = 0;
    gradient = zeros(siz);
    subgradient = zeros(siz);

    % Compute the smooth part: loss and gradient
    for i = 1:n
        inner_product = 0;
        for j = 1:siz(3) 
            inner_product = inner_product + trace(A(:, :, j) * X{i}(:, :, j)');
        end
        residual = Y(i) - inner_product;
        loss = loss + residual^2;

        for j = 1:siz(3)
            gradient(:, :, j) = gradient(:, :, j) - 2 * residual * X{i}(:, :, j);
        end
    end
    loss = loss / (2 * n);
    gradient = gradient / n;

    % Compute the nonsmooth part: nonsmooth_value and subgradient
    for k = 1:length(siz)
        Ak = mode_n_unfold(A, k);
        [U, S, V] = svd(Ak, 'econ');
        singular_values = diag(S);

        scad_penalty_value = sum(scad_penalty(singular_values, l));
        nonsmooth_value = nonsmooth_value + lambda * scad_penalty_value / length(siz);

        % Compute the subgradient for this mode
        S_subgrad = diag(scad_penalty_gradient(singular_values, l));
        Ak_subgrad = U * S_subgrad * V';
        subgradient = subgradient + mode_n_fold(Ak_subgrad, k, siz);
    end

    % Combine the smooth and nonsmooth parts
    loss = loss + nonsmooth_value;
    gradient = gradient + subgradient;
end

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
function stop = my_stopFcn(f, ~, ~)
    % Persistent variables to store the previous function values and iteration count
    persistent prev_f iter_count moving_avg moving_avg_window

    % Initialize persistent variables on the first call
    if isempty(prev_f)
        prev_f = f; % Set the initial function value
        iter_count = 0; % Initialize iteration count
        moving_avg = f; % Initialize moving average
        moving_avg_window = 30; % Set window size for moving average
        stop = false; % Do not stop on the first iteration
        return;
    end

    % Increment the iteration count
    iter_count = iter_count + 1;

    % Update the moving average with the new function value
    moving_avg = (moving_avg * (moving_avg_window - 1) + f) / moving_avg_window;

    % Check if the iteration count exceeds 100 before evaluating the stopping criteria
    if iter_count > 50
        % Calculate the difference between current and moving average
        diff = abs(f - moving_avg);

        % Check if the difference is below a threshold
        if diff < 1e-8
            fprintf('end at iter: %d', iter_count);
            stop = true; % Stop if the condition is met
            prev_f = 0;
        else
            stop = false; % Continue otherwise
        end
    else
        stop = false; % Continue if iteration count is 100 or less
    end

    % Update the previous function value
    prev_f = f;
end





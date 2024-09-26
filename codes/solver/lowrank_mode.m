function [A, out] = lowrank_mode(X, Y, siz, n, lambda,l)
    A = zeros(siz);
    loss = zeros(n, 1);
    scad_penalty_value = zeros(length(siz),1);  % Assuming the third dimension of size is similar to s

    tic;
    
    % Initialize CVX
    cvx_clear;
    cvx_begin
        variable A(siz);
        expressions loss(n) scad_penalty_value(length(siz));

        % Calculate the smooth part of the loss function
        for i = 1:n
            inner_product = 0;
            for j = 1:siz(3)
                inner_product = inner_product + trace(A(:, :, j) * X{i}(:, :, j)');
            end
            loss(i) = (Y(i) - inner_product)^2;  % Squared loss for each sample
        end

        % Calculate SCAD penalty for each mode
        val_A = cvx_value(A) ;
        for k = 1:length(siz) 
            Ak = mode_n_unfold(val_A, k); 
            [~, S, ~] = svd(Ak, 'econ');
            singular_values = diag(S);
            a=3.7;
            % SCAD penalty for the singular values
            scad_penalty_value(k) = sum(scad_penalty(singular_values,l,a));
        end
        total_scad_penalty = sum(scad_penalty_value);
        
        minimize((1/(2*n)) * sum(loss) + lambda * (total_scad_penalty))
    cvx_end

    elapsedTime = toc;
    out.time = elapsedTime;
end


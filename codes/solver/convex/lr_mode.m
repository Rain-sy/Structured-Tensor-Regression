function A = lr_mode(X, Y, siz, n, lambda)
    A = zeros(siz);
    loss = zeros(n, 1);
    total_singular_value = 0;
    % Initialize CVX
    cvx_clear;
    cvx_precision low
    cvx_begin        
        variable A(siz);
        expressions loss(n) total_singular_value;

        % Calculate the smooth part of the loss function
        for i = 1:n
            inner_product = 0;
            for j = 1:siz(3)
                inner_product = inner_product + trace(A(:, :, j) * X{i}(:, :, j)');
            end
            loss(i) = (Y(i) - inner_product)^2;  % Squared loss for each sample
        end
        % Calculate SCAD penalty for each mode
        for k = 1:length(siz) 
            Ak = mode_n_unfold(A, k) ;
            total_singular_value = total_singular_value + norm_nuc(Ak) ;
        end
        
        minimize((1/(2*n)) * sum(loss) +  lambda * total_singular_value ) %/3?
    cvx_end

end


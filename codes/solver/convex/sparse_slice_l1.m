function [A] = sparse_slice_l1(X, Y, size, n, lambda)
    A = zeros(size);
    loss = zeros(n, 1);

    cvx_clear;
    cvx_begin
        cvx_precision low
        variable A(size) 
        expressions loss(n)
        % loss
        for i = 1:n
            inner_product = 0;
            for j = 1:size(3) 
                inner_product = inner_product + trace(A(:, :, j) * X{i}(:, :, j)');
            end
            loss(i) = (Y(i) - inner_product)^2;
        end
        
        A_3 = mode_n_unfold(A,3);

        minimize ( (1/(2*n)) * sum(loss) + lambda * norm(A_3,'fro'))
    cvx_end

end


function A = lr_slice(X, Y, size, n, lambda)
    A = zeros(size);
    loss = zeros(n, 1);
    singular_values = zeros(size(3),1); 

    cvx_clear;
    cvx_begin
        cvx_precision low
        variable A(size)  % Define the variable A with the provided size array
        expressions loss(n) singular_values(size(3)) 
        
        % loss func
        for i = 1:n
            inner_product = 0;
            for j = 1:size(3)  % Assuming the third dimension corresponds to s
                inner_product = inner_product + trace(A(:, :, j) * X{i}(:, :, j)');
            end
            loss(i) = (Y(i) - inner_product)^2;
        end
        
        % Slice-wise
        for j = 1:size(3)
            singular_values(j) = norm_nuc(A(:,:,j)) ;
        end

        minimize ( (1/(2*n)) * sum(loss) + sum(singular_values) + lambda * norm(mode_n_unfold(A,3),1) )
    cvx_end

end

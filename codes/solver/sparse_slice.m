function [A] = sparse_slice(X, Y, size, n, lambda, l)
    A = zeros(size);
    loss = zeros(n, 1);
    a = 3.7;
    scad_penalty_value = zeros(size(1)*size(2),1);
    cvx_clear;
    cvx_begin
        variable A(size) 
        expressions loss(n) scad_penalty_value(size(1)*size(2),1)
        
        % loss
        for i = 1:n
            inner_product = 0;
            for j = 1:size(3) 
                inner_product = inner_product + trace(A(:, :, j) * X{i}(:, :, j)');
            end
            loss(i) = (Y(i) - inner_product)^2;
        end
        
        A_3 = mode_n_unfold(A,3);
        for k = 1:size(3)
            for j = 1:size(1)*size(2)
                scad_penalty_value(j) = scad_penalty_value(j) + scad_penalty(cvx_value(A(j*k)),l,a);
            end
        end
        scad_penalty_value = scad_penalty_value/numel(A);
        minimize ( (1/(2*n)) * sum(loss) + lambda * sum(scad_penalty_value)* norm(A_3,1))
    cvx_end

end


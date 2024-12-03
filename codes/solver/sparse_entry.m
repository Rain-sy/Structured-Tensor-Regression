function [A] = sparse_entry(X,Y,size,n, lambda,l)
    A = zeros(size);
    loss = zeros(n,1);
    a = 3.7;
    scad_penalty_value = zeros(size);
    cvx_clear;
    cvx_begin
        variable A(size) 
        expressions loss(n) scad_penalty_value(size) 
        % loss
        for i = 1:n
            inner_product = 0;
            for j = 1:size(3) 
                inner_product = inner_product + trace(A(:, :, j) * X{i}(:, :, j)');
            end
            loss(i) = (Y(i) - inner_product)^2;
        end
        for j = 1:numel(A)
            scad_penalty_value(j) = scad_penalty(cvx_value(A(j)), l, a);
        end
        minimize ( (1/(2*n)) * sum(loss) +  ...
            lambda * norm(mode_n_unfold(scad_penalty_value,3).*mode_n_unfold(A,3) , 1))
    cvx_end

end


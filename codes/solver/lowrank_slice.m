function [A,out] = lowrank_slice(X, Y, siz, n, lambda,l)
    A = zeros(siz);
    loss = zeros(n, 1);
    scad_penalty_value = zeros(siz(3),1); 

    tic;
    cvx_clear;
    cvx_begin
        variable A(size)  % Define the variable A with the provided size array
        expressions loss(n) scad_penalty_value(size(3))
        
        % loss func
        for i = 1:n
            inner_product = 0;
            for j = 1:siz(3)  % Assuming the third dimension corresponds to s
                inner_product = inner_product + trace(A(:, :, j) * X{i}(:, :, j)');
            end
            loss(i) = (Y(i) - inner_product)^2;
        end
        
        % Slice-wise
        val_A = cvx_value(A);
        for j = 1:siz(3)
            [~, S, ~] = svd(val_A(:,:,j), 'econ'); 
            singular_values = diag(S);
            a=3.7;
            scad_penalty_value(j) = sum(scad_penalty(singular_values,l,a));
        end
        % Total SCAD penalty
        total_scad_penalty = sum(scad_penalty_value);

        minimize ( (1/(2*n)) * sum(loss) + lambda * total_scad_penalty )
    cvx_end
    
    elapsedTime = toc;
    out.time = elapsedTime;
end

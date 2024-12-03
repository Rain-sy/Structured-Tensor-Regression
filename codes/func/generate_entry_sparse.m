function [X, Y, B] = generate_entry_sparse(d, s, sparsity, n, sigma)
    total_elements = d * d * s;
    num_non_zero_elements = round(total_elements * sparsity); 
    num_zero_elements = total_elements - num_non_zero_elements; 

    B = normrnd(0, 1, [d, d, s]);
    
    zero_indices = randperm(total_elements, num_zero_elements); 
    B(zero_indices) = 0; 
    
    X = cell(1, n);
    for i = 1:n
        X{i} = normrnd(0, 1, [d, d, s]); 
    end
    
    Y = zeros(n, 1);
    for i = 1:n
        Y_mean = 0;
        for j = 1:s
            Y_mean = Y_mean + trace(B(:, :, j) * X{i}(:, :, j)'); 
        end
        e = normrnd(0, sigma); 
        Y(i) = Y_mean + e; 
    end

    % disp('generate ended')
end

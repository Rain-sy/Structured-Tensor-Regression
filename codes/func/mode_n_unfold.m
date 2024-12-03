function unfolded = mode_n_unfold(A, n)
    % Unfold tensor A along the specified mode n
    dims = size(A);
    order = 1:length(dims);
    order([1 n]) = order([n 1]); % Move dimension n to the first position
    unfolded = reshape(permute(A, order), dims(n), []);
end


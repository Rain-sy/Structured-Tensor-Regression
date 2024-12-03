function folded = mode_n_fold(Ak_new, mode, original_size)
    % Fold the unfolded matrix back into a tensor
    dims = original_size;
    order = 1:length(dims);
    order([1 mode]) = order([mode 1]); % Move the first dimension back to mode
    folded = ipermute(reshape(Ak_new, dims(order)), order);
end
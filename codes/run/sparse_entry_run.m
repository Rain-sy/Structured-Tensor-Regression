%%%%%%%%%%%%%%%%%%%%%%%%% settings %%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
spar = 0.5;
sigma = 0.1;
total_experiments = 20; % 总实验次数
n_values = [1000, 2000, 3000]; % Number of samples
d_values = [6, 8, 10, 12, 14, 16, 18, 20]; % Dimension values

%%%%%%%%%%%%%%%%%%%%%%%%% change d %%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:length(d_values) % Loop over d values
    d = d_values(k);
    s = d;
    %%%%%%%%%%%%%%%%%%%%% change n %%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:length(n_values) % Loop over n values
        n = n_values(j);
        filename = sprintf('convex_data/sparse/entry/n=%d_d=%d.mat', n, d);
        
        % Load existing results if the file exists
        if isfile(filename)
            load(filename, 'A_spars', 'Asta', 'Rmse_results');
            current_experiment_index = numel(Rmse_results) + 1; % 计算已完成的实验数量
        else
            % Initialize variables if the file does not exist
            A_spars = cell(1, total_experiments);
            Rmse_results = zeros(1, total_experiments);
            Asta = cell(1, total_experiments);
            current_experiment_index = 1; % 从1开始
            lambda = sqrt(d * d * s / n) * 0.5;
            %l = lambda / 2;
            spar_sig_lambda = [spar, sigma, lambda];

            % Save parameters only once
            save(filename, 'spar_sig_lambda');
        end
        
        size = [d, d, s];
        
        % 继续实验直到 total_experiments
        for i = current_experiment_index:total_experiments
            [X, Y, B] = generate_entry_sparse(d, s, spar, n, sigma);
            Asta{i} = B;

            A = sparse_entry_l1(X, Y, size, n, lambda);  % Ensure size is defined correctly
            A_spars{i} = A;

            Rmse_results(i) = sqrt(mean((B(:) - A(:)).^2));
            % Display progress
            fprintf('d = %d, n = %d, Iteration %d completed with RMSE: %f\n', d, n, i, Rmse_results(i));
        end
        
        % Save results for current d and n values
        save(filename, 'A_spars', 'Asta', 'Rmse_results', '-append');
    end
end

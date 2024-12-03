%%%%%%%%%%%%%%%%%%%%%%%%% settings %%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
sigma = 0.1;
total_experiments = 4; % 总实验次数
n_values = [1000, 2000, 3000]; % Number of samples
d = 16; % Fix dimension value to 16
s=d;
spar_values = 8:2:14; % Change spar from 0.4 to 0.8

% Loop over spar values
for spar = spar_values
    %%%%%%%%%%%%%%%%%%%%% change n %%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:length(n_values) % Loop over n values
        n = n_values(j);
        filename = sprintf('sparse_data/fiber/spar/n=%d_spar=%d.mat', n, spar);
        
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
        end
        
        lambda = 1;
        l = 0.3 + sqrt(d*s*spar/ n) * 0.06;
        spar_sig_lambda = [spar, sigma, lambda];

        % Save the parameters only once
        if current_experiment_index == 1
            save(filename, 'spar_sig_lambda');
        end
        
        size = [d, d, s];

        % 继续实验直到 total_experiments
        for i = current_experiment_index:total_experiments
            [X, Y, B] = generate_fiber_sparse(d, s, spar, n, sigma);
            Asta{i} = B;

            A = sparse_fiber(X, Y, size, n, lambda, l);  % Ensure size is defined correctly
            A_spars{i} = A;

            Rmse_results(i) = sqrt(mean((B(:) - A(:)).^2));
            % Display progress
            fprintf('spar = %d, d = %d, n = %d, Iteration %d completed with RMSE: %f\n', spar, d, n, i, Rmse_results(i));
            save(filename, 'A_spars', 'Asta', 'Rmse_results', '-append');
        end 
    end
end

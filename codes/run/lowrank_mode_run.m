%%%%%%%%%%%%%%%%%%%%%%%%% settings %%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
r = 5;
sigma = 0.1;
total_experiments = 5; % 总实验次数
n_values = [1000, 2000, 3000]; % Number of samples
d_values = [6,8,12,14]; % Dimension values

%%%%%%%%%%%%%%%%%%%%%%%%% change d %%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:length(d_values)
    d = d_values(k);
    s = d;
    %%%%%%%%%%%%%%%%%%%%% change n %%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:length(n_values)
        n = n_values(j);
        filename = sprintf('lowrank_data/mode/n=%d_d=%d.mat', n, d);
        
        % Load existing results if the file exists
        if isfile(filename)
            load(filename, 'A_results', 'Asta', 'Rmse_results');
            current_experiment_index = numel(Rmse_results) + 1; % 计算已完成的实验数量
        else
            % Initialize variables if the file does not exist
            A_results = cell(1, total_experiments);
            Rmse_results = zeros(1, total_experiments);
            Asta = cell(1, total_experiments);
            current_experiment_index = 1; % 从1开始
            dsrnsig = [d, s, r, n, sigma];

            % Save parameters only once
            save(filename, 'dsrnsig');
        end
        
        size = [d, d, s];

        % 继续实验直到 total_experiments
        for i = current_experiment_index:total_experiments
            [X, Y, B] = generate_Tucker_lowrank(d, s, [r, r, r], n, sigma);
            Asta{i} = B;
            
            % Call the function
            lambda = 1; 
            l = 0.3 + sqrt(d * r * s / n) * 0.06;
            
            A = lowrank_mode(X, Y, size, n, lambda,l);
            A_results{i} = A;
            
            Rmse_results(i) = sqrt(mean((B(:) - A(:)).^2));
            % Display progress
            fprintf('d = %d, n = %d, Iteration %d completed with RMSE: %f\n', d, n, i, Rmse_results(i));
            % Save results for current d and n values
            save(filename, 'A_results', 'Asta', 'Rmse_results', '-append');
        end
        
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%% settings %%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
sigma = 0.1;
total_experiments = 30; % 总实验次数
n_values = [2000, 3000,4000]; % Number of samples
d = 16; % Fix dimension value to 16
s = 20;

%%%%%%%%%%%%%%%%%%%%%%%%% change rank %%%%%%%%%%%%%%%%%%%%%%%%
for r = 2:8 
    %%%%%%%%%%%%%%%%%%%%% change n %%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:length(n_values)
        n = n_values(j);
        filename = sprintf('lowrank_data/slice/rank/n=%d_r=%d.mat', n, r); 
        
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
            [X, Y, B] = generate_slice_lowrank(d, s, r, n, sigma);
            Asta{i} = B;
            
            % Call the function
            lambda = sqrt(d * d * s / n) * 0.5; 
            l = lambda / 2;
            
            A = lowrank_slice(X, Y, size, n, lambda, l);
            A_results{i} = A;
            
            Rmse_results(i) = sqrt(mean((B(:) - A(:)).^2));
            % Display progress
            fprintf('d = %d, n = %d, r = %d, Iteration %d completed with RMSE: %f\n', d, n, r, i, Rmse_results(i));
        end
        
        % Save results for current d and n values
        save(filename, 'A_results', 'Asta', 'Rmse_results', '-append');
    end
end

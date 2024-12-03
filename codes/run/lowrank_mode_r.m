%%%%%%%%%%%%%%%%%%%%%%%%% settings %%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
sigma = 0.1;
n_experiments = 30; 
n_values = [2000, 3000,4000]; % Number of samples
d = 16; % Fix dimension value to 16

%%%%%%%%%%%%%%%%%%%%%%%%% change rank %%%%%%%%%%%%%%%%%%%%%%%%
for r = 2:8 
    s = d;  
    %%%%%%%%%%%%%%%%%%%%% change n %%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:length(n_values)
        n = n_values(j);
        A_results = cell(1, n_experiments);
        Rmse_results = zeros(1, n_experiments);
        Asta = cell(1, n_experiments);
        dsrnsig = [d, s, r, n, sigma];
        
        % Generate filename for current settings
        filename = sprintf('lowrank_data/mode/rank/n=%d_r=%d.mat', n, r); 
        size = [d, d, s];
        save(filename, 'dsrnsig');

        for i = 1:n_experiments
            [X, Y, B] = generate_Tucker_lowrank(d, s, [r,r,r], n, sigma);  % Use r directly
            Asta{i} = B;
            
            % Call the function
            lambda = sqrt(d * d * s / n) * 0.5; 
            l = lambda / 2;
            
            A = lowrank_mode(X, Y, size, n, lambda, l);
            A_results{i} = A;
            
            Rmse_results(i) = sqrt(mean((B(:) - A(:)).^2));
            % Display progress
            fprintf('d = %d, n = %d, r = %d, Iteration %d completed with RMSE: %f\n', d, n, r, i, Rmse_results(i));
        end
        
        % Save results for current d and n values
        save(filename, 'A_results', 'Asta', 'Rmse_results');
    end
end

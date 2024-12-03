d = 10;
s = d;
r = 2;
n = 1000;
sigma_values = [0.1,0.5,1,5];  % Array of sigma values from 0.1 to 1 with a step size of 0.1
num_trials = 10;  % Number of trials for each sigma
rm1_values = cell(length(sigma_values), 1);  % Cell array to store rm1 for each trial
rm1_std_values = zeros(length(sigma_values), 1);  % Array to store standard deviation for each sigma

for i = 1:length(sigma_values)
    sigma = sigma_values(i);
    rm1_values{i} = zeros(num_trials, 1);  % Initialize an array to store rm1 for each trial
    
    for trial = 1:num_trials
        [X, Y, B] = generate_Tucker_lowrank(d, s, [r, r, r], n, sigma);
        
        size = [d, d, s];
        lambda = 0.3 + sqrt(d*r*s/n)*0.06;
        eta = 0.1;
        K = 24;  % steps
        
        A_opt = niAPG(Y, X, n, lambda, eta, K);
        rm1 = norm((B - A_opt), 'fro');  % Frobenius norm error
        rm1_values{i}(trial) = rm1;  % Store rm1 for this trial
    end
    
    avg_rm1 = mean(rm1_values{i});  % Calculate the average rm1 for this sigma
    rm1_std = std(rm1_values{i});   % Calculate the standard deviation of rm1 for this sigma
    
    rm1_std_values(i) = rm1_std;  % Store the standard deviation value for this sigma
    
    fprintf('For sigma = %.1f, average rm1 over %d trials = %f, standard deviation = %f\n', ...
            sigma, num_trials, avg_rm1, rm1_std);
end

clear; clc;
n_values = [2000,2500, 3000,3500,4000];  % n 的不同值
d_values = [8, 12, 16, 20];      % 修改后的 d 值

mean_rmse_lr = zeros(length(n_values), length(d_values));
max_rmse_lr = zeros(length(n_values), length(d_values));
min_rmse_lr = zeros(length(n_values), length(d_values));

figure;

hold on;
colors = [0.2, 0.45, 0.79;  % 蓝色
          0.13, 0.6, 0.13;  % 绿色
          1, 0.27, 0;      % 红色
          1, 0.74, 0.3];        % 黄色

for j = 1:length(d_values)
    d = d_values(j);  % 当前的 d 值
    
    for i = 1:length(n_values)
        n = n_values(i);
        
        % 读取 lr 的文件
        filename = sprintf('lowrank_data/slice/n/n=%d_d=%d.mat', n, d);
        data = load(filename);
        
        % 提取 cell 数组
        A_sta = data.Asta;
        A_results = data.A_results;
        rmse_results = data.Rmse_results;
        %variance = sqrt(var(rmse_results)) * (2000/n)* sqrt(d/16);
        variance = sqrt(var(rmse_results)) * 2000*d/ (n*20);
        % 重新计算均值、最大值和最小值
        mean_rmse_lr(i, j) = mean(rmse_results);  % 计算均值
        max_rmse_lr(i, j) = mean(rmse_results) + variance;
        min_rmse_lr(i, j) = mean(rmse_results) - variance;
    end
    
    % 计算误差
    error_lr = [mean_rmse_lr(:, j) - min_rmse_lr(:, j), max_rmse_lr(:, j) - mean_rmse_lr(:, j)];
    
    % 绘图
    errorbar(n_values, mean_rmse_lr(:, j), error_lr(:, 1), error_lr(:, 2), ...
             '.-', 'Color', colors(j, :), 'LineWidth', 1.3, ...
             'MarkerSize', 6, 'DisplayName', sprintf('d=%d', d));
end


txt = ylabel('$\mathrm{MSFE}$');
set(txt, 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'bold');
txt2 = xlabel('$n$');
set(txt2, 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'bold');
lgd = legend('show', 'Location', 'NorthEast');
lgd.FontSize = 18;



grid on;

xticks(n_values);xlim([1900, 4100]);% 设置 x 轴范围
set(gcf, 'unit', 'centimeters', 'Position', [12, 4, 16, 12]); % 设置图形窗口尺寸

hold off;

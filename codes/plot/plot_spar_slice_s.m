clear; clc;
n_values = [1000, 2000,3000];  % n 的不同值
s_values = 2:2:14;

mean_rmse_lr = zeros(length(n_values), length(s_values));
max_rmse_lr = zeros(length(n_values), length(s_values));
min_rmse_lr = zeros(length(n_values), length(s_values));

figure;

hold on;
colors = [0.2,0.45,0.79;  
         0.13,0.6,0.13;  
         1,0.27,0];

for j = 1:length(n_values)
    n = n_values(j);  % 当前的 n 值
    
    for i = 1:length(s_values)
        s = s_values(i);
        
        % 读取 lr 的文件
        filename = sprintf('sparse_data/slice/spar/n=%d_spar=%d.mat', n, s);
        data = load(filename);
        
        % 提取 cell 数组
        A_sta = data.Asta;
        A_results = data.A_spars;

        % 计算 RMSE
        rmse_results = zeros(1, length(A_sta));  % 存储 RMSE 结果
        for k = 1:length(A_sta)
            rmse_results(k) = sqrt(mean((A_sta{k}(:) - A_results{k}(:)).^2));
        end
        variance = var(rmse_results)*s/6;
        % 重新计算均值、最大值和最小值
        mean_rmse_lr(j, i) = mean(rmse_results);  % 计算均值
        max_rmse_lr(j, i) = mean(rmse_results)+ sqrt(variance);    % 计算最大值
        min_rmse_lr(j, i) = mean(rmse_results)-sqrt(variance);    % 计算最小值
    end
    
    % 计算误差
    error_lr = [mean_rmse_lr(j, :) - min_rmse_lr(j, :); max_rmse_lr(j, :) - mean_rmse_lr(j, :)];

    
    % 绘图
    errorbar(s_values, mean_rmse_lr(j, :), error_lr(1, :), error_lr(2, :), ...
             '.-','Color', colors(j, :), 'LineWidth', 1.3, ...
             'MarkerSize', 6, 'DisplayName', sprintf('n=%d', n));

end


lgd = legend('show', 'Location', 'NorthWest');
lgd.FontSize = 18;

ax = gca; % 获取当前坐标轴
%ax.GridColor = [1,1,1];   % 设置网格线颜色为白色
%ax.Color = [0.88, 0.88, 0.88]; % 设置背景为浅灰色
ax.GridLineWidth = 1;         % 设置坐标轴线宽

grid on;

txt = ylabel('$\mathrm{MSFE}$');
set(txt,'Interpreter', 'latex', 'FontSize', 18,'FontWeight','bold');
txt2 = xlabel('$\vert \mathcal{S}_3 \vert$');
set(txt2,'Interpreter', 'latex', 'FontSize', 18,'FontWeight','bold');
xticks(2:2:14);
xlim([1.6,14.4]);ylim([0.08,.9]);
set(gcf, 'unit', 'centimeters','Position', [12, 4, 16, 12]); % 设置图形窗口尺寸



hold off;
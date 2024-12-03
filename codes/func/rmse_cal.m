function rm = rmse_cal(cell1, cell2)
    % 检查两个 cell 是否具有相同的大小
    if numel(cell1) ~= numel(cell2)
        error('两个 cell 的大小不相同。');
    end

    % 初始化 RMSE
    total_error = 0;
    count = 0;

    % 遍历每个 cell 元素
    for i = 1:numel(cell1)
        tensor1 = cell1{i};
        tensor2 = cell2{i};

        % 检查张量的大小是否相同
        if ~isequal(size(tensor1), size(tensor2))
            error(['第 ' num2str(i) ' 个张量的大小不相同。']);
        end

        % 计算每个张量的误差
        total_error = total_error + norm(tensor1(:) - tensor2(:))^2;
        count = count + numel(tensor1);  % 统计元素总数
    end

    % 计算 RMSE
    rm = sqrt(total_error / count);
end

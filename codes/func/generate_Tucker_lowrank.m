function [X, Y, B] = generate_Tucker_lowrank(d, s, r, n, sigma)
    % 使用正态分布生成 U、V 和 W 矩阵
    U = rand(d,r(1));  % 左因子 U
    V = rand(d,r(2)); % 右因子 V
    W = normrnd(0, 1, [s, r(3)]); % 第三因子 W
    
    % 使用张量乘法生成低秩张量 B
    B = tensor(ktensor({U, V, W})).data;

    % 初始化输入张量 X
    X = cell(1, n);
    for i = 1:n
        X{i} = normrnd(0, 1, [d, d, s]); % 从正态分布生成 X
    end

    % 初始化输出向量 Y
    Y = zeros(n, 1);
    for i = 1:n
        Y_mean = 0;
        for j = 1:s
            Y_mean = Y_mean + trace(B(:, :, j) * X{i}(:, :, j)');
        end
        e = normrnd(0, sigma); % 从正态分布生成噪声
        Y(i) = Y_mean + e; 
    end

    % disp('generate ended');
end

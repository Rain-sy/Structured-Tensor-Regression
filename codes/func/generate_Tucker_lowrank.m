function [X, Y, B] = generate_Tucker_lowrank(d, s, r, n, sigma)
    U = rand(d, r(1));
    V = rand(d, r(2)); 
    W = rand(s, r(3)); 
    
    % 使用张量乘法生成低秩张量
    B = tensor(ktensor({U, V, W})).data;


    % 初始化输入张量 X
    X = cell(1, n);
    for i = 1:n
        X{i} = rand(d, d, s);
    end

    % 初始化输出向量 Y
    Y = zeros(n, 1);
    for i = 1:n
        Y_mean = 0;
        for j = 1:s
            Y_mean = Y_mean + trace(B(:, :, j) * X{i}(:, :, j)');
        end
        e = randn * sigma; 
        Y(i) = Y_mean + e;
    end

    % disp('generate ended');
end

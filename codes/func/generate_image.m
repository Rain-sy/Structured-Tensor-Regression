function [X, Y, B] = generate_image(file_name, n, sigma)
    % 读取图像数据
    B = double(imread(file_name)); % 读取 PNG 图像文件
    [x,y, s] = size(B); % 获取图像的尺寸，d为高度，s为颜色通道数

    % 初始化输入 X 和输出 Y
    X = cell(1, n);
    Y = zeros(n, 1);
    
    for i = 1:n
        % 生成随机数据
        X{i} = rand(x, y, s); 
        e = normrnd(0, sigma); 
        Y(i) = innerprod(tensor(X{i}), tensor(B)) + e; 
    end
end

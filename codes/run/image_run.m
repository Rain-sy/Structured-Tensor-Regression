clear;clc;
file = 'figures/realworld/tiny_1.JPEG';
n = 4000;
sigma = 0;
[X, Y, B] = generate_image(file, n, sigma);
%% 
[x,y,z] = size(B);
lambda = 0.3+ sqrt(x*y*z/n)*0.06;
%% 
eta =  0.1;
K = 50; % steps
A_opt = niAPG(Y, X, n, lambda, eta, K);
%% 
F_error = sqrt(mean((B(:) - A(:)).^2));
Y_pred2 = zeros(n, 1);
for p = 1:n
    inner_product = 0;
    for q = 1:s
        inner_product = inner_product + trace(A(:, :, q) * B(:, :, q)');
    end
    Y_pred2(p) = inner_product;
end
predict_error = sqrt(sum((Y - Y_pred2).^2))/sqrt(sum(Y.^2));
fprintf('\n F_error = %f, prdict_error = %f',F_error,predict_error/n);
imshow(uint8(A));
%% 

originalImage = (uint8(A));
% 2. 定义火山灰色彩调色板
volcanicAshPalette = [
    0.2, 0.2, 0.2;  % 深灰
    0.3, 0.3, 0.3;  % 中灰
    0.4, 0.4, 0.4;  % 浅灰
    0.1, 0.1, 0.1;  % 几近黑色
    0.25, 0.25, 0.35; % 蓝灰色
    0.15, 0.15, 0.25; % 更深的蓝灰色
];

% 3. 将原始图像转换为索引图像
numColors = size(volcanicAshPalette, 1);
[indexImage, ~] = rgb2ind(originalImage, numColors, 'nodither');

% 4. 应用火山灰调色板
volcanicAshImage = ind2rgb(indexImage, volcanicAshPalette);

% 5. 显示和保存结果图像
figure;
imshow(volcanicAshImage);
title('火山灰配色图像');





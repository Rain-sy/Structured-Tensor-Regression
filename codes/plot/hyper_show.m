% 假设 hyperspectralData 是你的高光谱数据，大小为 145x145x220
%hyperspectralData = load('realworld\Indian_pines.mat').indian_pines;
%hyperspectralData = load('realworld\Salinas.mat').salinas;
hyperspectralData = load('realworld\Salinas_corrected.mat').salinas_corrected;
% 假设选择第30、20和10波段作为R、G、B通道
R = hyperspectralData(:,:,10);
G = hyperspectralData(:,:,100);
B = hyperspectralData(:,:,200);


% 标准化波段到 [0, 1] 范围
R = mat2gray(R);
G = mat2gray(G);
B = mat2gray(B);

% 合并为RGB图像
RGB = cat(3, R, G, B);

% 显示RGB图像
figure;
%imshow(hyperspectralData(:,:,10));
imshow(RGB);
title('RGB Composite of Hyperspectral Image');

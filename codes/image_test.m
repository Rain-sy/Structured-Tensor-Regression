% 读取图像
file = 'realworld/tiny_2.JPEG';
originalImage = imread(file);

% 将图像转换为双精度，以便处理
originalImage = im2double(originalImage);

% 添加高斯噪声
meanNoise = 0;          % 噪声均值
varianceNoise = 0.005;  % 噪声方差
noisyImage = imnoise(originalImage, 'gaussian', meanNoise, varianceNoise);

% 显示原始图像和添加噪声后的图像

imshow(noisyImage);

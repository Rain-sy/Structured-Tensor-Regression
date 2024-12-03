clear;clc

% 读取高光谱图像数据
data = load('Indian_pines.mat').indian_pines; % 假设数据保存在'indian_pines.mat'中，变量名为'indian_pines'

hcube = hypercube(data);
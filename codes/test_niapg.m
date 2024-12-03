clear;

d = 10;
s= d;
r= 2;
n= 1000;
sigma =0.1;
[X, Y, B] = generate_Tucker_lowrank(d, s, [r,r,r], n, sigma);
%% 

size = [d,d,s];
lambda = 0.3+ sqrt(d*r*s/n)*0.06;
eta =  0.1;
K = 24; % steps
tic;
A_opt = niAPG(Y, X, n, lambda, eta, K);
toc;
%A = lowrank_mode(X,Y,size,n,lambda);
%% 


rm1 = norm((B-A_opt),'fro');
%rm2 = norm((B-A),'fro');
%fprintf('rm1 = %f,rm2 = %f',rm1,rm2);
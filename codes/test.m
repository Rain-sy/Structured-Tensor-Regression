clear;

d = 16;s=d; r= 5; n=1000;sigma = 0.1;spar=4;
% [X, Y, B] = generate_Tucker_lowrank(d, s, r, n, sigma);
% [M,N,L] = generate_slice_lowrank(d, s, 2, n, sigma);
[X, Y, B] = generate_slice_sparse(d,s,spar,n,sigma);
size = [d,d,s];
lambda = sqrt(d*d*s/n)*0.2;
l= lambda/2;

[A] = sparse_entry(X,Y,size,n,lambda,l);
[A_s] = sparse_slice(X,Y,size,n,lambda,l);
% [A,out] = lowrank_mode(X,Y,size,n,lambda,l);
% [A_s,out2] = lowrank_slice(X,Y,size,n,lambda,l);


rm1 = sqrt(mean((B(:) - A(:)).^2));
rm2 = sqrt(mean((B(:) - A_s(:)).^2));
fprintf('n=%d,  RMSE: mode: %f, slice: %f\n', n, rm1, rm2);


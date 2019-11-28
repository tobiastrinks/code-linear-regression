function visualize(X, y)

% PCA
[U] = pca(X);
k = 1;
Z = (U(:,1:k)'*X')';
% mxk = (kxn * nxm)'
size(Z);
figure;
scatter(Z(:,1), y);

end
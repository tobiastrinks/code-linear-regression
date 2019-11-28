function [U, k] = pca(X, retained_variance)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S

[m, n] = size(X);

Sigma = (1 / m) * X' * X;

[U, S, V] = svd(Sigma);

% choose number of principal components
k = n;
variance = 1;
while variance >= retained_variance
    variance = sum(S(1:k,:)(:)) / sum(S(1:n,:)(:));
    k = k - 1;
end

end

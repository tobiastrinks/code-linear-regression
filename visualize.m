function visualize(X, y)

% PCA
[U] = pca(X, 0);
k = 2;
Z = (U(:,1:k)'*X')';
% mxk = (kxn * nxm)'

filter_factor = 5;

y_mu = mean(y);
y_sigma = std(y);
y_size = y ./ y_sigma * 200 + 1;
y_color = (y - y_mu) ./ y_sigma;

figure;
scatter(
    Z(2:filter_factor:end,1),
    Z(2:filter_factor:end,2),
    y_size(2:filter_factor:end, 1),
    y_color(2:filter_factor:end, 1),
    "filled"
);

figure;
scatter(
    Z(2:filter_factor:end,1),
    Z(2:filter_factor:end,2),
    60,
    y_color(2:filter_factor:end, 1),
    "filled"
);


end
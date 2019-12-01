function [J, grad] = computeCostMulti(X, y, theta, lambda)

m = length(y);

h = X * theta;
s = sum((h - y).^2);

J = (1 / (2 * m)) * s  + (lambda / (2 * m)) * sum(theta(2:end).^2);
grad = X' * (1 / m) * (h - y) + [0 ; (lambda / m) * theta(2:end)];

end
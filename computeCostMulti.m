function J = computeCostMulti(X, y, theta)

m = length(y);

h = X * theta;
s = sum((h - y).^2);
J = (1 / (2 * m)) * s;

end
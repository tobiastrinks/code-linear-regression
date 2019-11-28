function [theta] = normalEquation(X, y, lambda)

L = eye(size(X, 2));
L(1, 1) = 0;

theta = pinv(X' * X + lambda * L) * X' * y;

end

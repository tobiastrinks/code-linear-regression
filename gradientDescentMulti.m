function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters, lambda)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    d = (1 / m) * (X' * (X * theta - y));

    theta = theta - (alpha * d);

    % Save the cost J in every iteration
    [J, grad] = computeCostMulti(X, y, theta, lambda);
    J_history(iter) = J;
end

end

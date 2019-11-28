function [theta] = measurePerformance(X_test, y_test, theta)

% Mean Absolute Error
predicted_y_test = X_test * theta;

MAE = abs(sum(predicted_y_test - y_test) / size(y_test, 1))

end
function [MAE, R2] = measurePerformance(X_test, y_test, theta)

predicted_y_test = X_test * theta;

% Mean Absolute Error
MAE = sum(abs(predicted_y_test - y_test) / size(y_test, 1));

% Coefficient of Determination (R-Squared)
R2 = (sum(abs(predicted_y_test - mean(y_test)))^2) / (sum(abs(y_test - mean(y_test)))^2);

end
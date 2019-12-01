function [U, k] = plotPerformanceInRelationToTrainingSetSize(training_alg, X_train, y_train, X_test, y_test)

training_set_sizes = [100:100:length(X_train)]';
train_error = zeros(length(training_set_sizes), 1);
test_error = zeros(length(training_set_sizes), 1);

for i = 1:length(training_set_sizes)
    N = training_set_sizes(i,1);

    [theta] = training_alg(X_train(1:N,:), y_train(1:N,:), 1.2);
    train_error(i, 1) = measurePerformance(X_train(1:N,:), y_train(1:N,:), theta);
    test_error(i, 1) = measurePerformance(X_test, y_test, theta);
end

figure;
plot(
    training_set_sizes, train_error, '-b;train_error;', 'LineWidth', 2, 'color', 'b',
    training_set_sizes, test_error, '-b;test_error;', 'LineWidth', 2, 'color', 'g'
);
xlabel('N (training set size)');
ylabel('error');

end
X_preprocessed = csvread('preprocess/preprocessed_features.csv');
X_preprocessed_normalized = csvread('preprocess/preprocessed_normalized_features.csv');
y = csvread('preprocess/prices.csv');

fprintf('Normalizing features ...\n');

X = [ featureNormalize(X_preprocessed) , X_preprocessed_normalized ];

%visualize(X, y);

% --- reducing dimensionality
retained_variance = 0.99;
[U, k] = pca(X, retained_variance);
fprintf('Reducing dimensionality from %d to %d and retain %d of variance\n', size(X, 2), k, retained_variance);
Z = (U(:,1:k)'*X')';
X = Z;

X = [ ones(size(X, 1), 1) , X ];

X_train = X(1:3680, :);
y_train = y(1:3680, :);

%X_valid = X(3681:4600, :);
%y_valid = y(3681:4600, :);

X_test = X(3681:4600, :);
y_test = y(3681:4600, :);

%fprintf('Running gradient descent ...\n');
%
%alpha = 0.01;
%num_iters = 600;
%
%theta = zeros(size(X_train, 2), 1);
%[theta, J_history] = gradientDescentMulti(X_train, y_train, theta, alpha, num_iters);
%measurePerformance(X_test, y_test, theta);

%fprintf('Running normal equation ...\n');

%[theta] = normalEquation(X_train, y_train);
%measurePerformance(X_test, y_test, theta);

fprintf('Plotting learning curve ...\n');

training_set_sizes = [100:10:length(X_train)]';
train_error = zeros(length(training_set_sizes), 1);
test_error = zeros(length(training_set_sizes), 1);

for i = 1:length(training_set_sizes)
    N = training_set_sizes(i,1);

    [theta] = normalEquation(X_train(1:N,:), y_train(1:N,:), 1.2);
    train_error(i, 1) = measurePerformance(X_train, y_train, theta);
    test_error(i, 1) = measurePerformance(X_test, y_test, theta);
end

figure;
plot(
    training_set_sizes, train_error, '-b;train_error;', 'LineWidth', 2, 'color', 'b',
    training_set_sizes, test_error, '-b;test_error;', 'LineWidth', 2, 'color', 'g'
);
xlabel('N (training set size)');
ylabel('error');

%figure;
%plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
%xlabel('Number of iterations');
%ylabel('Cost J');

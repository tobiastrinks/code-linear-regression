X_preprocessed = csvread('preprocess/preprocessed_features.csv');
X_preprocessed_normalized = csvread('preprocess/preprocessed_normalized_features.csv');
y = csvread('preprocess/prices.csv');

fprintf('Normalizing features ...\n');

X = [ featureNormalize(X_preprocessed) , X_preprocessed_normalized ];
X = [ ones(size(X, 1), 1) , X ];

X_train = X(1:3680, :);
y_train = y(1:3680, :);

%X_valid = X(3681:4600, :);
%y_valid = y(3681:4600, :);

X_test = X(3681:4600, :);
y_test = y(3681:4600, :);

fprintf('Running gradient descent ...\n');

alpha = 0.01;
num_iters = 600;

theta = zeros(size(X_train, 2), 1);
[theta, J_history] = gradientDescentMulti(X_train, y_train, theta, alpha, num_iters);
measurePerformance(X_test, y_test, theta);

fprintf('Running normal equation ...\n');

[theta] = normalEquation(X_train, y_train);
measurePerformance(X_test, y_test, theta);



%figure;
%plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
%xlabel('Number of iterations');
%ylabel('Cost J');

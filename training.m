X_preprocessed = csvread('preprocess/preprocessed_features.csv');
y = csvread('preprocess/prices.csv');

fprintf('Normalizing features ...\n');

X = featureNormalize(X_preprocessed);

%visualize(X, y);

% --- reducing dimensionality
%retained_variance = 0.99;
%[U, k] = pca(X, retained_variance);
%fprintf('Reducing dimensionality from %d to %d and retain %d of variance\n', size(X, 2), k, retained_variance);
%Z = (U(:,1:k)'*X')';
%X = Z;

X = [ ones(size(X, 1), 1) , X ];

m = size(X, 1);
train_size = ceil(m * 0.8);

X_train = X(1:train_size, :);
y_train = y(1:train_size, :);

X_test = X(train_size+1:m, :);
y_test = y(train_size+1:m, :);

% regularization
lambda = 0;

%fprintf('Running gradient descent ...\n');
%alpha = 0.01;
%num_iters = 600;
%theta = zeros(size(X_train, 2), 1);
%[theta, J_history] = gradientDescentMulti(X_train, y_train, theta, alpha, num_iters, lambda);

fprintf('Running normal equation ...\n');
[theta] = normalEquation(X_train, y_train, lambda);

[MAE, R2] = measurePerformance(X_test, y_test, theta)

plotPerformanceInRelationToTrainingSetSize(
    @(X, y, lambda) normalEquation(X, y, lambda),
    X_train,
    y_train,
    X_test,
    y_test
);

%figure;
%plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
%xlabel('Number of iterations');
%ylabel('Cost J');

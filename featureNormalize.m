function [X_norm, mu, sigma] = featureNormalize(X)

mu = mean(X);
sigma = std(X);

% apply feature scaling and normalization in order to
% speed up training, as big feature ranges affect performance
% negatively
X_norm = (X - mu) ./ sigma;

end

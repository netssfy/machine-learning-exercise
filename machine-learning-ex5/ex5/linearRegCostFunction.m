function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

# Cost Function
H = X * theta; # m x n * n * 1 => m x 1
Hbias = H - y; # m x 1
Twb = theta(2:end);# theta without bias 
J = ((Hbias' * Hbias) + Twb' * Twb * lambda) / m / 2;

# Gradient
# 		 n x m  * m x 1 + 				 n x 1
grad = (X' * Hbias + lambda * theta) / m; # n x 1
grad(1) = (Hbias' * ones(m, 1)) / m;



% =========================================================================

grad = grad(:);

end

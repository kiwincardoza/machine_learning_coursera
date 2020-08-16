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

% Compute cost J

hypo = X*theta;

diff = hypo .- y;

cost = sum(diff.^2)/(2*m);

org_theta = theta(2:end);

reg_cost = (lambda/(2*m))*(sum(org_theta.^2));

J = cost + reg_cost;


% Compute gradient grad

X_upd = X(:, 2:end);  % Remove bias

no_of_feat = size(X, 2);
diff_upd = repmat(diff, 1, no_of_feat);

first_grad = sum(diff_upd.*X, 1)./m;

reg_grad = (lambda/m).*org_theta;
reg_grad = reg_grad';
reg_grad = [0 reg_grad];

grad = first_grad .+ reg_grad;








% =========================================================================

grad = grad(:);

end

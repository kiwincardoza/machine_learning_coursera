function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

thetaX = X*theta;
hypo = sigmoid(thetaX);
%disp(thetaX);

logF1 = log(hypo);
logF2 = log(1.-hypo);

mulY1 = y.*logF1;
mulY1 = -mulY1;

mulY2 = y.*logF2;

pre_final = mulY1+mulY2-logF2;

J = sum(pre_final)/m;


FM = hypo - y;
no_features = size(X,2);
FM = repmat(FM, 1, no_features);
FM = FM.*X;
FM_upd = sum(FM,1);
FM_upd = FM_upd./m;

grad = FM_upd';
%disp(size(grad));







% =============================================================

end

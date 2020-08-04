function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

%X = [ones(m, 1) X];
%theta = [theta(1);theta];

thetaX = X*theta;
hypo = sigmoid(thetaX);
%disp(thetaX);

logF1 = log(hypo);
logF2 = log(1.-hypo);

mulY1 = y.*logF1;
mulY1 = -mulY1;
mulY2 = y.*logF2;

pre_final = mulY1+mulY2-logF2;

Jtemp = sum(pre_final)/m;



theta1 = theta(2:end);  % Removing the zeroth parameter

theta1 = theta1.^2;
sum1 = sum(theta1);
J = Jtemp + ((lambda*sum1)/(2*m));


% To compue gradient
%X = X(:,2:end);   % Revert X
%theta = theta(2:end);

%thetaX = X*theta;
%hypo = sigmoid(thetaX);

FM = hypo - y;
no_features = size(X,2);
FM = repmat(FM, 1, no_features);
FM = FM.*X;

FM_upd = sum(FM,1);
FM_upd = FM_upd./m;

theta_temp = theta(2:end);
extra1 = lambda.*theta_temp;
extra2 = extra1./m;
extra2 = [0;extra2];
extra2 = extra2';

FM_upd = FM_upd + extra2;
grad = FM_upd';


% =============================================================

end

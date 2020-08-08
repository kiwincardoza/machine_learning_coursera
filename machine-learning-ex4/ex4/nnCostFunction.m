function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m, 1) X];  % adding the bias unit to first layer

tempP1 = X*Theta1';
tempP1 = sigmoid(tempP1);

tempP1 = [ones(m, 1) tempP1];

tempP2 = tempP1*Theta2';
tempP2 = sigmoid(tempP2);

hofx = tempP2;

%disp(size(y));
ytemp = zeros(m,num_labels);
for i = 1:m
    pos = y(i);
    %pos = mod(pos,10);
    ytemp(i,pos) = 1;
endfor

logF1 = log(tempP2);
logF2 = log(1.-tempP2);

mulY1 = ytemp.*logF1;
mulY1 = -mulY1;
mulY2 = ytemp.*logF2;

pre_final = mulY1+mulY2-logF2;

%sum_pf = sum(pre_final);

%J = sum(sum(pre_final))/m;

%disp(size(pre_final));

sum1 = sum(pre_final,2);

J = sum(sum1)/m;


Theta1t = Theta1(:,2:input_layer_size+1);
Theta2t = Theta2(:,2:hidden_layer_size+1);
Theta2t = Theta2t';
Theta = [Theta1t Theta2t];

%disp(size(Theta));

Theta_sqr = Theta.^2;
sum1 = sum(Theta_sqr,2);
rextra = (sum(sum1).*lambda)/(2*m);
%disp(rextra);
J = J + rextra;


% -------------------------------------------------------------

% Gradient Calc

X = X(:,2:end);

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for t=1:m
    a1 = X(t,:);
    %a1 = [1 a1];
    a1 = a1';
    a1 = [1;a1];
    z2 = Theta1*a1;
    a2 = sigmoid(z2);
    a2 = [1;a2];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
    ytemp = zeros(num_labels, 1);
    pos = y(t);
    ytemp(pos) = 1;
    last_err = a3 - ytemp;
    z2_t = [1;z2];
    err = (Theta2'*last_err).*sigmoidGradient(z2_t);
    
    err = err(2:end);
    %delta1 = zeros(size(Theta1));
    delta1_t = err*a1';
    delta1 = delta1 + delta1_t;

    %delta2 = zeros(size(Theta2));
    delta2_t = last_err*a2';
    delta2 = delta2 + delta2_t;

    % For regularization
    delta1(:,2:end) = delta1(:,2:end) + (Theta1(:,2:end).*lambda)/m;
    delta2(:,2:end) = delta2(:,2:end) + (Theta2(:,2:end).*lambda)/m;


    Theta1_grad = delta1./m;
    Theta2_grad = delta2./m;
    
endfor



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

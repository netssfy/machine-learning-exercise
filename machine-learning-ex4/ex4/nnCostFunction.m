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
#is there a grace way?
Y = zeros(m, num_labels); # 5000 x 10
for i=1:m
	Y(i, y(i)) = 1;
end

############forward propagation##############
A1 = [ones(m, 1) X];# 5000x401

Z2 = A1 * Theta1';# 5000 x 401 * 401 x 25 = 5000 x 25
A2 = [ones(m, 1) sigmoid(Z2)]; # 5000 x 26

Z3 = A2 * Theta2'; # 5000 x 26 * 26 x 10 = 5000 * 10
A3 = sigmoid(Z3); # 5000 x 10

#below for code is passed, try to use vectorized version later
temp = 0;
for i = 1:m
	for k = 1:num_labels
		temp = temp + (-Y(i, k) * log(A3(i, k)) - (1 - Y(i, k)) * log(1 - A3(i, k)));
	end
end

part1 = 0;
for j = 1:hidden_layer_size
	for k = 2:input_layer_size + 1
		part1 += Theta1(j, k) ^ 2;
	end
end

part2 = 0;
for j = 1:num_labels
	for k = 2:hidden_layer_size + 1
		part2 += Theta2(j, k) ^ 2;
	end
end


J = temp / m + (part1 + part2) * lambda / 2 / m;


#########back propagation###########
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
for i = 1:m
  d3 = (A3(i, :) - Y(i, :))'; # 10 x 1
  d2 = (Theta2' * d3) .* sigmoidGradient([1 Z2(i, :)])';# 26 x 1

  D2 += d3 * A2(i, :); # 10 x 1 * 1 x 26 => 10 x 26
  D1 += d2(2:end) * A1(i, :); # 25 x 1 * 1 x 401 => 25 x 401
end

Theta1_grad = D1 / m;
Theta1_grad(:, 2:end) += lambda / m * Theta1(:, 2:end);
Theta2_grad = D2 / m;
Theta2_grad(:, 2:end) += lambda / m * Theta2(:, 2:end);







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

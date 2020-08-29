function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for i = 1:K  % Looping thru centroids

    no_of_ex = size(find(idx==i), 1);  % No of examples matching centroid i
    filterM = (idx == i); % Index with 1 for cluster i examples
    
    filterX = X .* filterM; % To make [0,0] for examples which isnt under
                            % cluster i

    sumX = sum(filterX, 1); % Sum of examples column-wise

    avgX = sumX ./ no_of_ex; % Avg of examples under cluster i

    centroids(i,:) = avgX;

endfor










% =============================================================


end



load('data.txt');
load('labels.txt');


% add a 1 at the end
data = [data ones(4601,1)];

dataTrain = data(1:2000,:);
dataTest = data(2001:4601,:);

labelsTrain = labels(1:2000,:);
labelsTest = labels(2001:4601,:);

epsilon = .00001;

maxiter = 1;

sizesToTest = [20 500 800 1000 1500 2000];

for sz = sizesToTest
    weightVec = logisticRegressionTrain(dataTrain(1:sz,:), labelsTrain(1:sz,:), epsilon, maxiter);
    
    % calculate accuracy. Run program on the test set, see what error rate I get
    % predict readings
    predictions = dataTest * weightVec;
    sig = mySigma(predictions);
    roundedSig = round(sig);

    % find number of times elements don't line up
    difference = numel(find(roundedSig ~= labelsTest));
    error = difference / size(labelsTest,1);
    acc = 1 - error;

    disp(acc)

end




% crunch into error function
%error = myLoss(dataTest, labelsTest, weightVec);
% error = 0;
% see what the error is
% disp(error)
% disp(1-error)

% get predictions
% get sigmoid of predictions
% round sigmoid
% compare to labels


function [weights] = logisticRegressionTrain(myData, myLabels, myEpsilon, myMaxiter)
%                       
% code to train a logistic regression classifier
%
% INPUTS:
%   data    =   n * (d+1) matrix withn samples and d features, where
%               column d+1 is all ones (corresponding to the intercept term)
%
%   labels  =   n * 1 vector of class labels (taking values 0 or 1)
%
%   epsilon =   optional argument specifying the convergence
%               criterion - if the change in the absolute difference in
%               predictions, from one iteration to the next, averaged across
%               input features, is less than epsilon, then halt
%               (if unspecified, use a default value of 1e-5)
%
%   maxiter =   optional argument that specifies the maximum number of
%               iterations to execute (useful when debugging in case your
%               code is not converging correctly!)
%               (if unspecified can be set to 1000)
%
% OUTPUT:
%   weights =   (d+1) * 1 vector of weights where the weights correspond to
%               the columns of "data"
%

    % Initialize at step t=0 to w(0)
    weights = zeros(58,1);
    oldweights = ones(58,1);
    numOfRows = size(myData,1);

    eta = 1;

    % for t in range maxiter (number of steps to take down the slope)
    for iter = 1:myMaxiter
        % compute gradients gt = gradient(Error(w(t)))
        % gradient = -1/N * sum_{i}( ((y_{i} - \sigma(x_{i}^{T}*w)) * x_{i} )
        sig = mySigma(myData * weights);
        labelWeight = myLabels - sig;

        gt = -1/numOfRows .* sum(labelWeight .* myData);       % *?* How to take the sum of all rows?
                                                                                    % *?* is x_{i} a row in myData?
                                                                                    
        if(abs(sum(weights) - sum(oldweights)) < myEpsilon)
            % quit because there isn't much change happening
            break
        end
        
        % update weights w(t+1) = w(t) + eta*(-gt)
        oldweights = weights;
        weights = weights + eta * -gt';
    end

%return weights
% do nothing because this is matlab
end


% function error = myLoss(myData, myLabels, myWeights)
% 
% % -1/N * sum_{i}(  y_{i} * ln( \sigma(x_{i}^{T} * w) )  +  (1-y_{i})( 1 - ln( \sigma(x_{i}^{T} * w) )  )
% sig = mySigma(myData * myWeights);
% pos = myLabels .* log(sig);
% neg = (1 - myLabels) .* (1 - log(sig));
% 
% error = -1/size(myData,1) .* sum(pos + neg);
% 
% %predictions = myData * myWeights;
% 
% 
% end


function sig = mySigma(t)

% sigma = (1 + exp(-t))^-1)
sig = 1./(1 + exp(-t));

end
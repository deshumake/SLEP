
load('ad_data.mat')
load('feature_name.mat')

% can't use 0 because it isn't positive
par = [ 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];

% Try each different regularization parameter
for num = par
    [w,c] = mySparse_train(X_train, y_train, num);
    
    % output of sigmoid function not binary prediction : sigmoid gives the
    % prob that the inputted data belongs to positive class
    
    predictions = X_test * w;
    y_test_score = 1./(1 + exp(-predictions));
    
    % compute AUC for w,c
    [None,None,None,AUC] = perfcurve(y_test, y_test_score, 1);
    
    disp('Area under curve: ')
    disp(AUC)
                                                
    
    % non-zero features in w
    disp('Number of non-zero features: ')
    disp(nnz(w))
end





function [w, c] = mySparse_train(data, labels, par)
% OUTPUT    w is equivalent to the first d dimension of weights in logistic train
%           c is the bias term, equivalent to the last dimension in weights in logistic train.
% Specify the options (use without modification).
opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations.

[w,c] = LogisticR(data, labels, par, opts);


end
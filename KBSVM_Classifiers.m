clear;
close all;
%{
%
% READ THE DIABETES.CSV FILE
dataset= readmatrix('diabetes.csv');
data_samples=length(dataset(:,1));
dataset(dataset(:,9)==0,9)=-1;
% 500 DATA SAMPLES AS TRAINING SAMPLES
training_samples=500;
% 268 DATA SAMPLES AS TEST SAMPLES
test_samples=268;
% SAMPLING THE INDEX FOR GENERATING TRAINING & TEST DATA
population=1:data_samples;
training_index=randsample(population,training_samples);
test_index=setdiff(population, training_index);
% Replace this s_train_data with training_data :::::::::
%s_train_data=dataset(training_index,:);
training_data=dataset(training_index,:);
writematrix(training_data,'training_data.csv');
test_data=dataset(test_index,:);
writematrix(test_data,'test_data.csv');
training_data=[];
test_data=[];
%}
training_data= readmatrix('training_data.csv');
test_data= readmatrix('test_data.csv');
%
% 10-FOLD CROSS-VALIDATION ::::::::::::::
%
% Length of the training dataset : 500
n1=length(training_data(:,9));
label=unique(training_data(:,9));
% Number of folds can be changed by changing the kfold value
kfold=1;
div = floor(n1/kfold);
% Assigning different hyper-parameter values
%Cdata= 10.^(-3:3);
%Cadv= 10.^(-3:3);
% Hyper-parameters after Cross-Validation
Cdata= 0.1;
Cadv= 0.1;
%
% Train a SVM classifier on the data & advice ....
%
% Advice set 1 for negetive examples....
D1 = [0 1 0 0 0 0 0 0; 0 0 0 0 0 1 0 0];
d1 = [100; 25];
% Advice set 2 for positive examples....
D2 = [0 -1 0 0 0 0 0 0; 0 0 0 0 0 -1 0 0];
d2 = [-126; -30];
%
%
u1 = sdpvar(2, 1);
u2 = sdpvar(2, 1);
eta = sdpvar(8, 1);
zeta = sdpvar(1, 1);
w = sdpvar(8, 1);
b = sdpvar(1, 1);
%
%
% Randomly dividing the Training Dataset into training data & test data for CV
population=1:n1;
for i = 1:kfold
    cv_test_idx{i}=randsample(population,div);
    cv_train_idx{i}=setdiff(population, div);
    
end
%
% Length of the CV training dataset : n2
n2=length(training_data(cv_train_idx{1}, 1:8 ));
xi = sdpvar(n2, 1);
% K-Fold Cross-Validation
for i = Cdata
    for j = Cadv
        for k = 1:kfold
            % Training data for CV
            x= training_data(cv_train_idx{k},1:8);
            y= training_data(cv_train_idx{k}, 9);
            % Testing data for CV
            x_cv_test= training_data(cv_test_idx{k},1:8);
            y_cv_test= training_data(cv_test_idx{k}, 9);
            %
            Constraints_adv = [ diag(y) * (x*w + b) - 1 + xi >= 0, xi >= 0,...
                    u1 >= 0,u2 >= 0, zeta >= 0, -w + D1'*u1 + eta == 0,...
                    +b - 1 - d1'*u1 + zeta >= 0, w + D2'*u2 + eta == 0,...
                    -b - 1 - d2'*u2 + zeta >= 0];
            %
            Objective_adv = sum(abs(w)) + i * sum(xi) + j * (sum(abs(eta)) + sum(zeta));
            %
            diagnostic = optimize(Constraints_adv, Objective_adv, sdpsettings('solver', 'sedumi', 'verbose', 0));
            wOpt_adv = double(w);
            bOpt_adv = double(b);
            % Predicting labels of the CV test data
            y_cv_pred = sign(x_cv_test * wOpt_adv + bOpt_adv);
            % Confusion matrix for error calculation
            mtx=[];
            for l = 1: length(label(:,1))
            s1=0;
            s2=0;
                for l1 = 1:length(y_cv_test(:,1))
                    if ((label(l) == y_cv_test(l1)) && (y_cv_test(l1) == y_cv_pred(l1)))
                        s1 = s1+ 1;
                    elseif ((label(l) == y_cv_test(l1)) && (y_cv_test(l1) ~= y_cv_pred(l1)))
                        s2 = s2 + 1;
                    end
                end
                mtx= [mtx,s1];
                mtx=[mtx,s2];
            end
            %disp(mtx);
            C1= find(Cdata == i);
            C2= find(Cadv == j);
            % Matrix containing error for all possible combination...
            % of hyperparameter values for each fold 
            Error(k, C1, C2) = ((mtx(2)+mtx(4))/(length(y_cv_test(:,1))));

        end
    end
end

%writematrix(Error,'5-fold_CV.csv');
% Testing the classifier on the test data set
x=training_data(:,1:8);
y=training_data(:,9);
xi = sdpvar(n1, 1);
%
Constraints_adv = [ diag(y) * (x*w + b) - 1 + xi >= 0, xi >= 0,...
                    u1 >= 0, u2 >= 0, zeta >= 0, -w + D1'*u1 + eta == 0,...
                    +b - 1 - d1'*u1 + zeta >= 0, w + D2'*u2 + eta == 0,...
                    -b - 1 - d2'*u2 + zeta >= 0];

%Objective_adv = 0.5*(w'*w) + Cdata * sum(xi) + Cadv * (sum(abs(eta)) + sum(zeta));
Objective_adv = sum(abs(w)) + Cdata * sum(xi) + Cadv * (sum(abs(eta)) + sum(zeta));
%
diagnostic = optimize(Constraints_adv, Objective_adv, sdpsettings('solver', 'sedumi', 'verbose', 0));
wOpt_adv = double(w);
bOpt_adv = double(b);
disp(diagnostic);
disp(wOpt_adv);
disp(bOpt_adv);
fprintf('Objective value = %g.\n', double(Objective_adv));
%PREDICTED VALUE WITH ADVICE
x_test=test_data(:,1:8);
y_true=test_data(:,9);
y_pred=sign(x_test * wOpt_adv + bOpt_adv);
%disp(y_pred);
% Confusion matrix for error calculation
mtx=[];
for j = 1: length(label(:,1))
    s1=0;
    s2=0;
    for i = 1:length(y_true(:,1))
        if ((label(j) == y_true(i)) && (y_true(i) == y_pred(i)))
            s1 = s1+ 1;
        elseif ((label(j) == y_true(i)) && (y_true(i) ~= y_pred(i)))
            s2 = s2 + 1;
        end
   end
   mtx= [mtx,s1];
   mtx=[mtx,s2];
end
%disp(mtx);
Error=(mtx(2)+mtx(4))/(length(y_true(:,1)));
fprintf('\n Error on the test data set = %0.4f \n', Error)

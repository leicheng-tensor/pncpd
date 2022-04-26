clc;
clear all;
close all;
randn('state',1); rand('state',1); %#ok<RAND>
addpath(genpath(pwd));

%% Generate a low-rank tensor
DIM = [100,100,100]; % Dimensions of data
R = 10;  % True CP rank
Z = cell(length(DIM),1);
for m=1:length(DIM)
    Z{m} = rand(DIM(m),R);
    Z{m} = proj(Z{m});
end
X = conj(double(ktensor(Z))); % Generate tensor by factor matrices

%% Add Noise
SNR = 0;
sigma2 = var(X(:))*(1/(10^(SNR/10)));
GN = sqrt(sigma2)*randn(DIM);
Y = X + GN;

%% Run Algorithm
ts = tic;
[model] = PNCPD(Y, 'maxRank', min(DIM), 'tol', 1e-6, 'maxiters', 2500, 'verbose', 1, 'isSpeedUp', 1);
t_total = toc(ts);

%% Performance evaluation
X_hat = double(model.X);
err = X_hat(:) - X(:);
rmse = sqrt(mean(err.^2));
rrse = sqrt(sum(err.^2)/sum(X(:).^2));

%% Report results
fprintf('\n------------Bayesian CP with Nonnegative Factors------------------------------------------------------------------------\n')
fprintf('SNR = %g, True Rank = %d\n', SNR, R);
fprintf('RRSE = %g, RMSE = %g, Estimated Rank = %d, \nEstimated SNR = %g, Time = %g\n', ...
    rrse, rmse, model.R, model.SNR, t_total);
fprintf('--------------------------------------------------------------------------------------------------------------------------\n')

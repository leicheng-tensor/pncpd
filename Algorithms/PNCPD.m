function [model] = PNCPD(Y, varargin)
%  Bayesian CP Factorization with Nonnegative Factors
%
%  [model] = BCPF(Y, 'PARAM1', val1, 'PARAM2', val2, ...)
%
%  INPUTS
%     Y              - input tensor
%     'maxRank'      - The initialization of rank (larger than true rank)
%     'maxiters'     - max number of iterations (default: 2500)
%     'tol'          - lower band change tolerance for convergence dection
%                      (default: 1e-6)
%     'verbose'      - visualization of results
%                       - 0: no
%                       - 1: text (default)
%     'isSpeedUp'    - whether to use Nesterov acceleration
%                       - 0: no
%                       - 1: yes
%   OUTPUTS
%      model         - Model parameters and hyperparameters
%
%   Example:
%
%        [model] = BCPF(Y, 'init', 'ml', 'maxRank', 10, 'dimRed', 1, 'maxiters', 100, ...
%                                'tol', 1e-6, 'verbose', 3);
%
% < Bayesian CP Factorization >
% Copyright (C) 2020 Lei Cheng
% Acknowledgement
% This code is built based on the Matlab Code from the following paper:
% Q. Zhao, L. Zhang, and A. Cichocki, "Bayesian CP factorization of incomplete 
% tensors with automatic rank determination," IEEE Transactions on 
% Pattern Analysis and Machine Intelligence, vol. 37, no. 9, pp. 1751-1763, Sep. 2015.
% We highly appreciated the authors of this paper. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set parameters from input or by using defaults
dimY = size(Y);
N = ndims(Y);

ip = inputParser;
ip.addParameter('maxRank', min(dimY), @isscalar);
ip.addParameter('maxiters', 2500, @isscalar);
ip.addParameter('tol', 1e-6, @isscalar);
ip.addParameter('verbose', 1, @isscalar);
ip.addParameter('isSpeedUp', 1, @isscalar);
ip.parse(varargin{:});

R   = ip.Results.maxRank;
maxiters  = ip.Results.maxiters;
tol   = ip.Results.tol;
verbose  = ip.Results.verbose;
is_speed_up = ip.Results.isSpeedUp;
maxiters_inner = 3000;

%% Initialization
Y = tensor(Y);
a_gamma0     = 1e-6;
b_gamma0     = 1e-6;
a_beta0      = 1e-6;
b_beta0      = 1e-6;
gammas = 1*ones(R,1);
beta = 1;

Z = cell(N,1);
ZLambda = cell(N,1);
for n = 1:N
    ZLambda{n} = zeros(R,R);
    [U, S, ~] = svd(double(tenmat(Y,n)), 'econ');
    if R <= size(U,2)
        Z{n} = U(:,1:R)*(S(1:R,1:R)).^(0.5);
    else
        Z{n} = [U*(S.^(0.5))  randn(dimY(n), R-size(U,2))];
    end
end
% --------- E(aa') = cov(a,a) + E(a)E(a')----------------
EZZT = cell(N,1);
for n=1:N
    EZZT{n} = Z{n}'*Z{n};
end

%% Model learning
X = conj(double(ktensor(Z)));
hit = 0;
residual = 0;
for it=1:maxiters
    %% Update factor matrices
    Aw = diag(gammas);
    for n=1:N
        % compute E(Z_{\n}^{T} Z_{\n})
        ENZZT = ones(R,R);
        for m=[1:n-1, n+1:N]
            ENZZT =  ENZZT.*EZZT{m};
        end
        % compute E(Z_{\n})
        DDD = double(tenmat(Y, n));
        ZLambda{n} = beta * ENZZT + Aw;
        Z_lambda = ZLambda{n};
        B = khatrirao_fast(Z{[1:n-1, n+1:N]},'r');
        FFF = beta * double(DDD*B);
        Z_update = Z{n};
        h = eig(Z_lambda);
        L = max(h);
        mu = min(h);
        condition_num = L/mu;
        tau = (sqrt(L)-sqrt(mu))/(sqrt(L)+sqrt(mu));
        obj_c = zeros(1,maxiters_inner);
        if is_speed_up == 1 && it>=5 && condition_num < 100
            hit = hit + 1;
            M_update = Z_update;
            for it_inner = 1 : maxiters_inner
                grad = M_update* Z_lambda  -  FFF;                          
                Z_update_pre = Z_update;
                Z_update_c = M_update - (1/L)*grad;
                Z_update_c = proj(Z_update_c);
                M_update = Z_update_c + tau*(Z_update_c - Z_update_pre);
                obj = 0.5*trace(Z_update_c*Z_lambda*Z_update_c'-2*Z_update_c*FFF');                                               
                obj_c(it_inner) = obj;
                if it_inner >= 1
                    gradZ = Z_update* Z_lambda  -  FFF;
                    Z_update = Z_update - (1/L)*gradZ;
                    Z_update = proj(Z_update);  
                    M_update = Z_update;
                    obj = 0.5*trace(Z_update*Z_lambda*Z_update'-2*Z_update*FFF');
                    obj_c(it_inner) = obj; 
                else
                    Z_update = Z_update_c;
                end
                if (it_inner >1)
                    diff_obj = abs(obj_c(it_inner)-obj_c(it_inner-1));
                    if(diff_obj < .0001)
                        break;
                    end
                end
            end
        else 
            for it_inner = 1 : maxiters_inner
                grad = Z_update* Z_lambda  -  FFF;                
                obj = 0.5*trace(Z_update*Z_lambda*Z_update'-2*Z_update*FFF');
                obj_c(it_inner) = obj;
                if (it_inner >1)
                    diff_obj = abs(obj_c(it_inner)-obj_c(it_inner-1));
                    if(diff_obj < .0001)
                        break;
                    end
                end
                kappa = 1e-3;
                alpha = kappa/(1+it_inner);
                Z_update = Z_update- alpha * grad;
                Z_update = proj(Z_update);
            end
        end
        Z{n} = Z_update;
        EZZT{n} = Z{n}'*Z{n};
    end    
      
    %% Update hyperparameters gamma
    a_gammaN = (sum(dimY(1:N)) + a_gamma0)*ones(R,1);
    b_gammaN = 0;
    for n=1:N
        b_gammaN = b_gammaN + diag(Z{n}'*Z{n});
    end
    b_gammaN = b_gamma0 + b_gammaN;
    gammas = a_gammaN./b_gammaN;
    
    %% Update noise beta
    X = conj(double(ktensor(Z)));
    Y_vec = Y(:);
    X_vec = X(:);
    err = Y_vec'*Y_vec - 2*real(Y_vec'*X_vec) + X_vec'*X_vec;
    a_betaN = a_beta0 + prod(dimY);
    b_betaN = b_beta0 + err;
    beta = a_betaN/b_betaN;
    
    %% Convergence Check
    residual(it) = sqrt(err);
    if (it>1) && (abs(residual(it)-residual(it-1))<=tol)
        break;
    end
    
    %% Display progress
    if verbose
        fprintf('Iter. %d: Residual = %g, R = %d \n', it, residual(it), R);
    end
    
    %% Prune out unnecessary components
    DIMRED_THR = 1e2;
    if it > 1
        MAX_GAMMA = min(gammas) * DIMRED_THR;
        if sum(find(gammas > MAX_GAMMA))
            indices = find(gammas <= MAX_GAMMA);
            for n=1:N
                Z{n} = Z{n}(:,indices);
                ZLambda{n} = ZLambda{n}(indices,indices);
                EZZT{n} = EZZT{n}(indices,indices);
            end
            gammas = gammas(indices);
            R=length(gammas);
        end
    end
end

%% Output
model.X=X;
model.Z=Z;
model.R=R;
model.gammas=gammas;
model.beta=beta;
model.SNR=10*log10(var(X(:))*beta);
end

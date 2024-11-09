% To solve the following problem
% each column is a sample 
% \min r1*\sum_v ||\tilde X_1^v -\tilde X_2^v Z^v||_{2,1} + ||\mathcal{Z}||_{Sp} 
% + r2*\sum_v \tilde alpha_v ||S-Z^v||_F^2 + r3*Tr(F^T\tilde L F)
% s.t. S \geq 0, S^T1 = 1, \tilde alpha 1 = 1, \tilde aplha \geq 0, F^TF = I, F \in R^{N \times k}
% nC: cluster number
% rate: the selecting ratio for samples
% p: 0<= p < 2
% r2: large enough
% NITER: the iteration number for inner loop

function [S, F, clusternum, Y_label,Z,alpha] = TMVSC(X,nC,rate,p,NITER,sle)
addpath('fun_TMVSC')
if nargin<5
    NITER = 1;
end
%% pre-process
itermax = 100;
r1 = 0.18;
r2 = 0.0001;
r3 = 1;
tau = 10;
%% Partition and pre-calculate
nV = length(X);
[X1_label,X2_label] = select1(X{1},rate,sle);
n = length(X1_label);
m = length(X2_label);
for v = 1:nV
    Xt1 = X{v}(:,X1_label);
    Xt2 = X{v}(:,X2_label);
    Xt1 = [Xt1;ones(1,n)*tau];
    Xt2 = [Xt2;ones(1,m)*tau];
    X1{v} = Xt1; 
    X2{v} = Xt2; 
end

%% Initialize
S = zeros(m,n);
alpha = ones(nV,1)/nV;
Q = orth1(X2);
A = cell(1,nV);
for v=1:nV
    A{v} = X2{v}*Q{v};
end
S1 = 0;
%% Main process
for iter = 1:itermax
    %% Multi-view dictionary representation via tensor
    for v = 1:nV
        S1 = S1 + alpha(v)*Q{v}'*S;
    end
    [Z,E] = Trra_Sp(X1,A,S1,r1,r2,alpha,p,1);
%     [Z,E] = Trra_Sp(X1,X2,S,r1,r2,alpha,p,1);
    Z_hat = 0;
    for v=1:nV
        Z{v} = Q{v}*Z{v};
        Z_hat = Z_hat + alpha(v)*Z{v};
    end
    
    %% Optimal bipartite graph learning
    [S,clusternum,F] = coclustering_bipartite_fast(Z_hat',nC,r3/2,NITER);
    S = S';
    % update alpha
    for v = 1:nV
        alpha(v) = 1/norm(S-Z{v},'fro');
    end
%     alpha = [1 1 1];
    alpha = alpha/sum(alpha);
    % convergence condition
    if clusternum == nC %&& iter ==3
        disp(['iter ' num2str(iter)]);
        break
    end
end

%% Output
SS0=sparse(n+m,n+m); SS0(1:m,m+1:end)=S; SS0(m+1:end,1:m)=S';
% [~, y]=graphconncomp(SS0);
y=conncomp(graph(SS0));
% y1=conncomp(graph(SS0));
if length(X1_label) == length(X2_label)
    Y_label = zeros(1,n);
else
    Y_label = zeros(1,m+n);
end
Y_label(X2_label) = y(1:m);
Y_label(X1_label) = y(m+1:end);
Y_label = Y_label';

% [Id Idx] = sort(X1_label,'ascend');
% S2 = S(:,Idx);
% clean(full(S2))
% y1=y(1:n)';
% y2=y(n+1:end)';
% y = y2;
if clusternum ~= nC
    sprintf('Can not find the correct cluster number: %d %d', nC,clusternum)
end
end

function Q = orth1(X2)
nV = length(X2);
Q = cell(1,nV);
for v=1:nV
    Q{v} = orth(X2{v}');
end
end


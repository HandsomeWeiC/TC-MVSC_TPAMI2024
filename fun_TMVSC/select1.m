function [X1_label,X2_label] = select1(X,rate,K)
% [d,N] = size(X);
if nargin < 3
    K = 'Kmeans';
end
[~,N] = size(X);
if K == 0
    X1_label = 1:N;%[41:100,141:200,241:300,341:400,441:500];
    X2_label = 1:N;%[1:40,101:140,201:240,301:340,401:440];
elseif K == 'random'
    % random
    idx = randperm(N);
    X2_label = idx(1:floor(rate*N));
    X1_label = idx(floor(rate*N)+1:end);
    X1 = X(:,X1_label);
    X2 = X(:,X2_label);
else
%     num = find_min(K);
    label = kmeans(X',K,'emptyaction','singleton','replicates',20,'display','off');
    num = K;
%     [label,~] = hKM(X,[1:N],num,1,0);
    X1 = []; X1_label = [];
    X2 = []; X2_label = [];
    for i = 1:num
        idx = find(label == i);
        nu_1 = floor(length(idx)*rate);
        X2 = [X2 X(:,idx(1:nu_1))];
        X2_label = [X2_label;idx(1:nu_1)];
        X1 = [X1 X(:,idx(nu_1+1:end))];
        X1_label = [X1_label;idx(nu_1+1:end)];
    end
end
end

function num = find_min(N)
N = min(N,64);
k = 1;
while N - 2^k > 0
    k = k+1;
end
num = k;
end
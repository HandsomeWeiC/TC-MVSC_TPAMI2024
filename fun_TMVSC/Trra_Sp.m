function [Z,E] = Trra_Sp(X1,X2,S,r1,r2,alpha,p,NITER,display)
% This routine solves the following multi-view Schatten p-norm optimization problem,
% \min r1*\sum_v ||E^v||_{2,1} + ||\mathcal{Z}||_{Sp} + r2*\sum_v alpha_v ||S-Z^v||_F^2 
% s.t. E^v = X_1^v - X_2^v Z^v.
% inputs V views:
%        X1 -- D*n data matrix, D is the data dimension, and n is the number
%             of data vectors.
%        X2 -- D*m matrix of a dictionary, m is the size of the dictionary
if nargin<9
    display = true;
end

tol = 1e-8;
mu = 1e-5; 
max_mu = 10e10; 
pho_mu = 1.1;
rho = 1e-5; 
max_rho = 10e10; 
pho_rho = 1.1;

[d,n] = size(X1{1});
m = size(X2{1},2);
nV = length(X1);
sX = [m n nV];
%% pre-calcuate & Initializing
for v = 1:nV
    X2TX2{v} = X2{v}'*X2{v};
    X2TX1{v} = X2{v}'*X1{v};
    Z{v} = zeros(m,n); %Z{2} = zeros(m,n);
    W{v} = zeros(m,n); % multiplier
    J{v} = zeros(m,n); % for solving J
    E{v} = zeros(size(X1{v},1),n); %E{2} = z eros(size(X{v},1),N);
    Y{v} = zeros(size(X1{v},1),n); %Y{2} = zeros(size(X{k},1),N);
end

%% main
for iter = 1:NITER
    % updating Z^v and tensor Z
    temp_E1 = cell(1,nV);
    tem_max = [];
%     Z_hat = zeros(m,n);
    for v=1:nV
        tmp = 2*r2*alpha(v)*S+X2{v}'*Y{v}+rho*(X2TX1{v}-X2{v}'*E{v})-W{v}+mu*J{v};
        Z{v}=inv((2*r2*alpha(v)+mu)*eye(m,m)+ rho*X2TX2{v})*tmp;
%         Z_hat = Z_hat+alpha(v)*Z{v};
%         temp_E=[temp_E;X1{v}-X2{v}*Z{v}+Y{v}/rho];
        temp_E1{v} = X1{v}-X2{v}*Z{v}+Y{v}/rho;
    end

    % updating E^v, Y^v
%     [Econcat] = solve_l1l2(temp_E,r1/rho);
%     ro_b =0;
%     E{1} =  Econcat(1:size(X1{1},1),:);
%     Y{1} = Y{1} + rho*(X1{1}-X2{1}*Z{1}-E{1});
%     ro_end = size(X1{1},1);
%     for v=2:nV
%         ro_b = ro_b + size(X1{v-1},1);
%         ro_end = ro_end + size(X1{v},1);
%         E{v} =  Econcat(ro_b+1:ro_end,:);
%         Y{v} = Y{v} + rho*(X1{v}-X2{v}*Z{v}-E{v});
%     end
    Econcat = [];
    for v = 1:nV
        E{v} = solve_l1l2(temp_E1{v},r1/rho);
        Y{v} = Y{v} + rho*(X1{v}-X2{v}*Z{v}-E{v});
        Econcat = [Econcat;E{v}];
    end

    % updating tensor J
    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});  %% W的更新在哪里？
    z = Z_tensor(:);
    w = W_tensor(:);
    [j, ~] = wshrinkObj_weight_lp(z + 1/mu*w,1/mu,sX, 0,3,p);
    J_tensor = reshape(j,sX);

%     [~,J_tensor] = GST4J(Z_tensor, W_tensor,1/mu,p); % GST4J是怎么做的？
%     j = J_tensor(:);

    % break condition
    for v = 1:nV
        tem_max = [tem_max;X1{v}-X2{v}*Z{v}];
    end

    leq1 = tem_max-Econcat;
    leq2 = z-j;
    stopC = max(max(max(abs(leq1))),max(abs(leq2)));
    if display && (iter==1 || mod(iter,50)==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ',stopALM=' num2str(stopC,'%2.3e')]);
%         disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
%             ',rank=' num2str(rank(Z,1e-4*norm(Z,2))) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
    % updating multipliers
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
    w = w + mu*(z - j);
    W_tensor = reshape(w, [m n nV]);
    for v = 1:nV
        W{v} = W_tensor(:,:,v);
    end
    end
end

end
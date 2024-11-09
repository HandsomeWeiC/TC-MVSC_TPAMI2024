clear;
clc

X1 = cell(0);
load('dataset/X_high_dimen50dN5.mat');
X1 =[X1 X];
load('dataset/X_high_dimen50dN7.mat');
X1 =[X1 X];
load('dataset/X_high_dimen50dN9.mat');

X1 =[X1 X];

X = X1;
cls_num=length(unique(Y));
nV = length(X);

%% Our method: TCMVSC
addpath("fun_TMVSC/")
rate = 0.3;
p=1;
NITER = 10;

[S, F, clusternum, Y_label,Z_our,alpha_our] = TMVSC(X,cls_num,rate,p,NITER,0);
[m n] = size(S);

S = full(S);
a1 = ClusteringMeasure(Y,Y_label)
close all
figure; imagesc(S)
figure; imagesc(Z_our{1})
figure; imagesc(Z_our{2})
figure; imagesc(Z_our{3})
rmpath("fun_TMVSC/")












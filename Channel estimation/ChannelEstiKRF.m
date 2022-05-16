%% channel estimation 
close all;
clear
clc
load('D:\DansFiles\OneDrive - Technion\project - IEEE Signal Processing Cup 2021\Datasets\dataset1.mat');

%% seperate the dataset
IRS.a=pilotMatrix4N(:,1:N);
IRS.b=pilotMatrix4N(:,N+1:2*N);
% build c,d so it will create a=c, b=d
IRS_rest=pilotMatrix4N(:,2*N+1:4*N);

IRS.c = zeros(N);
IRS.d = zeros(N);
index_c = zeros(1,N);
index_d = zeros(1,N);
% run in a
for i = 1:length(IRS.a)
    i
    % run in c
    for j = 1:length(IRS_rest)
        if (IRS.a(:,i) == IRS_rest(:,j))
            index_c(i) = j;
            IRS.c(:,i) = IRS_rest(:,j);
        end
        
        if (IRS.b(:,i) == IRS_rest(:,j))
            index_d(i) = j;
            IRS.d(:,i) = IRS_rest(:,j);
        end
    end
end

SIG.a=receivedSignal4N(:,1:N);
SIG.b=receivedSignal4N(:,N+1:2*N);
SIG.c=receivedSignal4N(:,2*N+1:3*N);
SIG.d=receivedSignal4N(:,3*N+1:4*N);

%% train set

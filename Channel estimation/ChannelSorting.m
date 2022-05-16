%% channel sorting
% sort to a=c=-b=-d
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
% get index to sort the signal later
index_c = zeros(1,N);
index_d = zeros(1,N);
% run in a
for i = 1:length(IRS.a)
    i
    % run in c
    for j = 1:length(IRS_rest)
        if (IRS.a(:,i) == IRS_rest(:,j))
            index_c(i) = j;
            % IRS.c(:,i) = IRS_rest(:,j);
        end
        
        if (IRS.b(:,i) == IRS_rest(:,j))
            index_d(i) = j;
            % IRS.d(:,i) = IRS_rest(:,j);
        end
    end
end
% set sorting
IRS.c = IRS_rest(:,index_c);
IRS.d = IRS_rest(:,index_d);

disp('chack that the configoration are sorted a-c=0, a+b=0, b-d=0')
sum(abs(IRS.a-IRS.c),'all')
sum(abs(IRS.a+IRS.b),'all')
sum(abs(IRS.b-IRS.d),'all')

%% sort c,d
SIG.a=receivedSignal4N(:,1:N);
SIG.b=receivedSignal4N(:,N+1:2*N);
SIG_rest = receivedSignal4N(:,2*N+1:4*N);
% sorted
SIG.c = SIG_rest(:,index_c);
SIG.d = SIG_rest(:,index_d);

%% save
name = 'sorted_dataset1.mat';
save(name,'IRS','SIG')
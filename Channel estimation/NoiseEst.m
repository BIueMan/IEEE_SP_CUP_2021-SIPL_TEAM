%% load sorted dataset1
clear all;
close all;
load('sorted_dataset1');

N = length(IRS.a);

VarianceEst = zeros(2,N);
for i = 1:length(IRS.a)
    % a-c
    residualnoise = (SIG.a(:,i) - SIG.c(:,i))/sqrt(2);
    VarianceEst(1,i) = var(residualnoise);
    %b-d
    residualnoise = (SIG.b(:,i) - SIG.d(:,i))/sqrt(2);
    VarianceEst(2,i) = var(residualnoise);
end

%%
disp('max var found:')
max_noise = max(VarianceEst,[],'all')
disp('min var found:')
min_noise = min(VarianceEst,[],'all')
disp('mean var:')
mean_noise = mean(VarianceEst,'all')

%% save
noise_est = mean_noise;
save('noise_est.mat', 'noise_est')
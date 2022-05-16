% clear all
% close all
% clc
load('dataset2.mat');
load('bestConfigs_net2_seed_565.mat');
load('GQA_results.mat');
%%
load('h_denoised_imag.mat');
load('h_denoised_real.mat');
load('V_denoised_imag.mat');
load('V_denoised_real.mat');
%%
h=hd_real+1j*hd_imag;
V=V_real+1j*V_imag;;
figure
subplot(131)
imshow(reshape(user_17_bestConfing,[64,64]))
subplot(132)
imshow(reshape(SolutionMay(:,17),[64,64]))
%%
best_config=[user_1_bestConfing;user_2_bestConfing;user_3_bestConfing;user_4_bestConfing; user_5_bestConfing;user_6_bestConfing;user_7_bestConfing;user_8_bestConfing;user_9_bestConfing;user_10_bestConfing;
user_11_bestConfing;user_12_bestConfing;user_13_bestConfing;user_14_bestConfing; user_15_bestConfing;user_16_bestConfing;user_17_bestConfing;user_18_bestConfing;user_19_bestConfing;user_20_bestConfing;
    user_21_bestConfing;user_22_bestConfing;user_23_bestConfing;user_24_bestConfing;user_25_bestConfing;user_26_bestConfing;user_27_bestConfing;user_28_bestConfing;user_29_bestConfing;user_30_bestConfing;
    user_31_bestConfing;user_32_bestConfing;user_33_bestConfing;user_34_bestConfing; user_35_bestConfing;user_36_bestConfing;user_37_bestConfing;user_38_bestConfing;user_39_bestConfing;user_40_bestConfing;
    user_41_bestConfing;user_42_bestConfing;user_43_bestConfing;user_44_bestConfing;user_45_bestConfing;user_46_bestConfing;user_47_bestConfing;user_48_bestConfing;user_49_bestConfing;user_50_bestConfing]';
%%
config_mixed=zeros(N,nbrOfUsers);
rate_mixed=zeros(nbrOfUsers,1);
rate_sol=zeros(nbrOfUsers,1);
worked=zeros(2,nbrOfUsers);
for i=1:nbrOfUsers
   
   config=reshape(SolutionMay(:,i),[64,64]);
   stirr=sign(sum(config,2))*ones(1,64);
   diff=sign(config-stirr);
   diff=reshape(diff,[N,1]);
   ind=diff==0;
   config_mixed(~ind,i)=diff(~ind);
   config_mixed(ind,i)=best_config(ind,i);
   rate_sol(i)=rate(V(:,:,i),SolutionMay(:,i),h(:,i),B,K,M,varianceEstimate_power)*1e-6;
   rate_mixed(i) = rate(V(:,:,i),config_mixed(:,i),h(:,i),B,K,M,varianceEstimate_power)*1e-6;
   
end
%%
worked(1,:)=rate_mixed>rate_sol;
worked(2,:)=rate_mixed-rate_sol;
s=39
figure
subplot(121)
imshow(reshape(config_mixed(:,s),[64,64]))
subplot(122)
imshow(reshape(SolutionMay(:,s),[64,64]))
%%
A=abs(config_mixed)~=1;
t=sum(A,'all');
%%
close all
figure
imshow(reshape(config_mixed(:,16),[64,64]))

%%

load('dataset1.mat')
B = 10e6; %Bandwidth in Hertz
symbolTime = 1/B; %There are B symbols per second
noise=zeros(K,1);
for i=1:N
    
    noisetemp=zeros(K,1);
    if(ones(1,N)*(pilotMatrix4N(:,i)==pilotMatrix4N(:,2*N+i))==4096)
        noisetemp=(receivedSignal4N(:,i)-receivedSignal4N(:,2*N+i))/2;
        noisetemp=noisetemp+(receivedSignal4N(:,N+i)-receivedSignal4N(:,3*N+i))/2;
    elseif(ones(1,N)*(pilotMatrix4N(:,i)==pilotMatrix4N(:,3*N+i))==4096)
        noisetemp=(receivedSignal4N(:,i)-receivedSignal4N(:,3*N+i))/2;
        noisetemp=noisetemp+(receivedSignal4N(:,N+i)-receivedSignal4N(:,2*N+i))/2;
    end
    noise=noise+noisetemp;
end
noise=noise/(sqrt(N));
varnoise=var(noise);
varianceEstimate_power = varnoise/symbolTime;














%%
function ratei = rate(V,theta,h,B,K,M,N0)
    ratei=(B/(K+M-1))*sum(log2(1+((abs(V*theta+h)).^2)/N0),1);
end

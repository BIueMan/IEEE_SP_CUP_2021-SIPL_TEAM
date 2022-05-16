clear;
close all;
clc;
load('dataset1.mat');
load('dataset2.mat');
%%

X=transmitSignal(1);
hd=reshape(sum(receivedSignal,2)/(X*N),[K,nbrOfUsers]);
V=zeros(K,N,nbrOfUsers);
for i=1:nbrOfUsers
    V(:,:,i)=(receivedSignal(:,:,i)-hd(:,i)*ones(1,N)*X)/pilotMatrix/X;
end
%%

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
%% calculate the stm
%% calc F_M
F = zeros(K, M);
for v = 1 : K
    for k = 1 : K
        F(k, v) = exp(-1j* 2*pi* (k-1)*(v-1)/K);
    end
end
Q=F'*F/500;
Vtime=zeros(K,N,nbrOfUsers);
hdtime=zeros(K,nbrOfUsers);
V_denoised=zeros(K,N,nbrOfUsers);
h_denoised=zeros(K,nbrOfUsers);
for i=1:nbrOfUsers
    Vtime(:,:,i)=F'*V(:,:,i)/K;
    hdtime(:,i)=F'*hd(:,i)/K;
    Vtime(M+1:end,:,i)=0;
    hdtime(M+1:end,i)=0;
    V_denoised(:,:,i)=F*Vtime(:,:,i);
    %V_denoised(:,1,i)=0;
    h_denoised(:,i)=F*hdtime(:,i);
    for j=1:63
        V_denoised(:,1,i)=V_denoised(:,1,i)+V_denoised(:,1+64*j,i)/63;
    end
    h_denoised(:,i)=h_denoised(:,i)-V_denoised(:,1,i);
end
%% s
V_real=real(V_denoised);
V_imag=imag(V_denoised);
hd_real=real(h_denoised);
hd_imag=imag(h_denoised);
save('C:\Users\tomerf\Documents\Channels\V_denoised_real','V_real');
save('C:\Users\tomerf\Documents\Channels\V_denoised_imag','V_imag');
save('C:\Users\tomerf\Documents\Channels\h_denoised_real','hd_real');
save('C:\Users\tomerf\Documents\Channels\h_denoised_imag','hd_imag');
%% ddsd
save('C:\Users\tomerf\Documents\Channels\V_denoised', 'V_denoised');
save('C:\Users\tomerf\Documents\Channels\h_denoised', 'h_denoised');
%%load('SIPL_First_Solution.mat');

%%

%% function that calculates the gradient of the rate at a given theta
function gradi = grad(V,theta,h,B,K,M,N_0)
    temp = N_0*ones(K,1)+(abs(V*theta+h)).^2;
    temp2 = 2*(V*theta+h)./temp;
    gradi=(B/(log(2)*(K+M-1)))*V'*temp2;
end
%% function that calculates the rate of a config
function ratei = rate(V,theta,h,B,K,M,N0)
    ratei=(B/(K+M-1))*sum(log2(1+((abs(V*theta+h)).^2)/N0),1);
end























%%

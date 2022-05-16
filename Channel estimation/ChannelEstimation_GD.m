%% channel estimation 

close all;
clear
clc
load('dataset1.mat');


%% separate the dataset
xtest = pilotMatrix4N(:,floor(0.8*4*N+1):4*N);
xval = pilotMatrix4N(:,floor(0.6*4*N+1):floor(0.8*4*N));
xtrain = pilotMatrix4N(:,1:floor(0.6*4*N));
ztest = receivedSignal4N(:,floor(0.8*4*N+1):4*N);
zval = receivedSignal4N(:,floor(0.6*4*N+1):floor(0.8*4*N));
ztrain = receivedSignal4N(:,1:floor(0.6*4*N));
iterN=113;

eta = [0.3,0.7,1,3,7]*2/9830;
%Vtf=zeros(K,N);
%Vtb=zeros(K,N);
Vtm=zeros(iterN+1,K,N);
Om = pilotMatrix4N(:, 3*N+1:4*N);
OmInv = inv(Om);
valscore=zeros(length(eta),iterN+1);

trainscore=zeros(length(eta),iterN+1);
%lambda=10^-2;
testscore= zeros(1,length(eta));

%% GD selecting the step size !!i picked eta=0.0002!! 
close all
 j=3;
    Vt=zeros(K,N);
    valscore(j,1)=sum(abs(zval-Vt*xval*transmitSignal(1)).^2,'all')/3277;
    trainscore(j,1)=sum(abs(ztrain-Vt*xtrain*transmitSignal(1)).^2,'all')/9830; 
    for i=1:iterN
         %Vtf(:,N)=Vt(:,1);
         %Vtf(:,1:N-1)=Vt(:,2:N);
         %Vtb(:,2:N)=Vt(:,1:N-1);
         %Vtb(:,1)=Vt(:,N);
         
         grad =((Vt*xtrain*transmitSignal(1)-ztrain)*transmitSignal(1)*xtrain');%+lambda*(2*Vt-Vtb-Vtf));
         Vt = Vt-eta(j)*grad./sum(abs(grad),'all');
         Vtm(i,:,:)=Vt;
         valscore(j,i+1)=sum(abs(zval-Vt*xval*transmitSignal(1)).^2,'all')/3277;
         trainscore(j,i+1)=sum(abs(ztrain-Vt*xtrain*transmitSignal(1)).^2,'all')/9830;   
    end
   %testscore(j)=sum(abs(ztest-Vt*xtest*transmitSignal(1)).^2,'all');
    figure(j)
    hold on
    plot(0:iterN,valscore(j,:),'r');
    plot(0:iterN,trainscore(j,:),'b');
    hold off


%% looking at the channel estimator  !!the GD looks much worse then the LS but for some reason gets better score (WTF)
phiinv=inv(xtrain(:,1:N));
V=ztrain(:,1:N)*phiinv/transmitSignal(1); %LS ESTIMATOR

V_gd=reshape(Vtm(114,:,:),K,N);%GD ESTIMATOR we get the best Vt on step=114
s=8660;

figure(6)
subplot(2,1,1)
    hold on
    plot(1:K,abs(ztrain(:,s)),'b')
    plot(1:K,abs(V*xtrain(:,s)*transmitSignal(1)),'r')
    plot(1:K,abs(V_gd*xtrain(:,s)*transmitSignal(1)),'g')
    legend('recieved signal','LS estimator','GD estimator');
    hold off
    subplot(2,1,2)
    hold on
    plot(1:K,unwrap(angle(ztrain(:,s))),'b')
    plot(1:K,unwrap(angle(V*xtrain(:,s)*transmitSignal(1))),'r')
    plot(1:K,unwrap(angle(V_gd*xtrain(:,s)*transmitSignal(1))),'g')
    legend('recieved signal','LS estimator','GD estimator');
    hold off 
%% calc the mse of every method
trainscoreLS=sum(abs(ztrain-V*xtrain*transmitSignal(1)).^2,'all')/9830;   
trainscoreGD=sum(abs(ztrain-V_gd*xtrain*transmitSignal(1)).^2,'all')/9830;
valscoreLS=sum(abs(zval-V*xval*transmitSignal(1)).^2,'all')/3277;   
valscoreGD=sum(abs(zval-V_gd*xval*transmitSignal(1)).^2,'all')/3277;
testscoreLS=sum(abs(ztest-V*xtest*transmitSignal(1)).^2,'all')/3277;   
testscoreGD=sum(abs(ztest-V_gd*xtest*transmitSignal(1)).^2,'all')/3277;

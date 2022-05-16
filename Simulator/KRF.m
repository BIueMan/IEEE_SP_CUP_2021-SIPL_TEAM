%% Channel Estimation- KRF Algorithm

close all
clear
clc
load('dataset1.mat');

X=kron(transmitSignal.',eye(K));
Theta=X.';
Y=receivedSignal4N(:,1:N).';
S=pilotMatrix4N(:,1:N).';
p_T=pinv(Theta);
p_S=pinv(S);
W_T=p_S*Y*p_T;
H=zeros(N,K);
G=H.';
for i=1:N
    j=i;
    w_i=reshape(W_T(i,:),[K,K]);
    [U,Si,V]=svds(w_i,1);
    H(i,:)=sqrtm(Si)*conj(V);
    G(:,i)=sqrtm(Si)*U;
end

%% Test set
error_Sima=0;
for i=1:N
    error_i=sum((abs(receivedSignal4N(:,i)-c*transmitSignal(1)-G*diag(S(i,:))*H*transmitSignal)).^2);
    error_Sima=error_Sima+error_i
end
error_Sima=error_Sima/N;

%% everything else
error_Simas=0
for j=N+1:4*N
    error_j=sum((abs(receivedSignal4N(:,i)-c*transmitSignal(1)-G*diag(S(i,:))*H*transmitSignal)).^2);
    error_Simas=error_Simas+error_j
    j
end
error_Simas=error_Simas/N
error=error_Simas+error_Sima
%%
v=real(H(:,1));
meow=0;
for i=1:K
    if real(H(:,i))-real(v)>=(1e-25)
        meow=meow+1;
    end
   
end 
% %% testing
% f=1:500;
% esti_12=G*diag(S(12,:))*H*transmitSignal;
% esti_12_2N=G*diag(S(12+2*N,:))*H*transmitSignal;
% figure(1)
% subplot(1,2,1)
% hold on
% xlabel('Subcarrier K')
% ylabel('|y|')
% plot(f,abs(receivedSignal4N(:,26)))
% %plot(f,abs(receivedSignal4N(:,12+2*N)))
% plot(f,abs(esti_12))
% legend('received 12','estimated 12')
% hold off
% subplot(1,2,2)
% plot(f,angle(esti_12))
% hold on
% plot(f,angle(receivedSignal4N(:,26)))
% 
% 
% %% evaluation
% 
% % 
%  f=1:500;
%  esti_1=G*diag(S(1,:))*H*transmitSignal; 
%  figure(1) 
%  hold on
%  xlabel('Subcarrier K')
%  ylabel('|y|')
%  plot(f,abs(receivedSignal4N(:,1)))
%  %plot(f,abs(receivedSignal4N(:,12+2*N)))
% plot(f,abs(esti_1))
% %plo
%  legend('received 1','estimated 1')
% % hold off
% % 
% % figure(2) 
% % hold on
% % xlabel('Subcarrier K')
% % ylabel('Y phase')
% % plot(f,unwrap(angle(receivedSignal4N(:,1))))
% % %plot(f,abs(receivedSignal4N(:,12+2*N)))
% % plot(f,unwrap(angle(esti_1)))
% % legend('received 1','estimated 1')
% % 
% % 
% % %end
% %     
% 
% %% dataset 2
% load('dataset2.mat');
% s=pilotMatrix.';
% esti_user_1=G*diag(s(12,:))*H*transmitSignal;
% f=1:500;
% esti_1=G*diag(S(12,:))*H*transmitSignal; 
% figure(1) 
% hold on
% xlabel('Subcarrier K')
% ylabel('|y|')
% plot(f,abs(receivedSignal4N(:,12)))
% %plot(f,abs(receivedSignal4N(:,12+2*N)))
% plot(f,abs(esti_1))
% legend('received 1','estimated 1')
% hold off
% %%

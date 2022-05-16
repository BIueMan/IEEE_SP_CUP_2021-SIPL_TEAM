function [H,G] = KRF_func(pilotMatrix,receivedSignal,transmitSignal)
[K,N]=size(receivedSignal);
X=kron(transmitSignal.',eye(K));
Theta=X.';
Y=receivedSignal.';
S=pilotMatrix.';
p_T=pinv(Theta);
p_S=pinv(S);
W_T=p_S*Y*p_T;
H=zeros(N,K);
G=H.';
for i=1:N
    w_i=reshape(W_T(i,:),[K,K]);
    [U,Si,V]=svds(w_i,1);
    H(i,:)=sqrtm(Si)*conj(V);
    G(:,i)=sqrtm(Si)*U;
end
end


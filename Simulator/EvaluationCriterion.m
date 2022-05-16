function [meanDR] = EvaluationCriterion(W,N0,H_theta,transmitSignal)
%EvaluationCriterion calculates and returns the mean Data Rate
%   K- number of carriers  M-number of FIR coeff's, where (M-1) is the
%   number of cyclic prefixes per single OFDM block
%   B- bandwidth N0- noise power spectral density   N- number of users 
%   W-wheights per user. W is a vector of length N with elements from {1,2}
%   H_theta- The evaluated filter (in frequency domain) for a single theta 
%   configuration, H_theta is a vector of length K.
B=10e7;
K=length(transmitSignal);
M=20;
N=50;

P = sum(transmitSignal.^2)/K; %check definition of "signal power"
R = sum(log2(1+(P/(B*N0))*(abs(H_theta)).^2));
meanDR = (B/(N*(K+M-1)))*sum(W*R);
end
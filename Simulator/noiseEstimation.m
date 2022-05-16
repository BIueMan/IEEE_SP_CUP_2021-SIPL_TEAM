% plot(abs(receivedSignal4N(:,1)))
% hold on
% plot(abs(receivedSignal4N(:,2)))
% hold on
% plot(abs(receivedSignal4N(:,3)))
% hold on
% plot(abs(receivedSignal4N(:,4)))
K=500;

signal = ifft((K^0.5)*receivedSignal4N(:,2));
%signal_kat = signal(10:250);

E = sum(signal)/length(signal);
noise = signal - E;

figure(1)
plot(abs(noise))
hold on
plot(abs(signal))
legend('noise','signal')
xlabel('time')


N=abs(fft(noise)/(K^0.5));
figure(2)
plot(N)
hold on
plot(abs(receivedSignal4N(:,2)))
legend('fftnoise','fftsignal')
xlabel('freq')

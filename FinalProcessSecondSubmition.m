clear;
close all;
clc;
%% dsds
load('dataset1.mat');
load('dataset2.mat');
gpu = gpuDevice(1)
%%

disp('Noise estimation...');
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
clear pilotMatrix4N
clear receivedSignal4N
disp('Done.');
%% noise cancelation
disp('Channel estimation...');
V = zeros(500,4096,50);
hd = zeros(500,50);

X=transmitSignal(1);
F = zeros(K, M);
for v = 0 : M - 1 
    for k = 0 : K - 1  
        F(k + 1, v + 1) = exp(-1j* 2*pi* k*v/K);
    end
end
pilinv = inv(pilotMatrix);
% for user = 1 : 50
%     user
%     recTime = 1/500 * F' * receivedSignal(:,:,user);
%     recFiltered = zeros(500,4096);
%     for i = 1 : 4096
%         temp = zeros(20,1);
%         [~, kssd] = maxk(recTime(:,i),40);
%         temp(kssd) = (recTime(kssd,i));
%         recFiltered(:,i) = F * temp;
%     end
%     hd(:,user) = sum(recFiltered,2)/(X*N);
%     V(:,:,user)=(recFiltered-hd(:,user)*ones(1,N)*X)*pilinv/X;
% end
load('C:\Users\tomerf\Documents\Channels\V_denoised.mat');
load('C:\Users\tomerf\Documents\Channels\h_denoised.mat');

V = gpuArray(V_denoised);
hd = gpuArray(h_denoised);

X=gpuArray(X);
F = gpuArray(F);
clear recFiltered
clear receivedSignal
disp('Done.');
%% Channel_estimation
% hd=reshape(sum(receivedSignal,2)/(X*N),[K,nbrOfUsers]);
% V=zeros(K,N,nbrOfUsers);
% for i=1:nbrOfUsers
%     V(:,:,i)=(receivedSignal(:,:,i)-hd(:,i)*ones(1,N)*X)/pilotMatrix/X;
% end
%% calc the rate
disp('STM calculation...');
rate_fromdata=gpuArray(zeros(N,nbrOfUsers));
for i=1:nbrOfUsers
    rate_fromdata(:,i) = (B/(K+M-1))*sum(log2(1+((abs(V(:,:,i)*pilotMatrix+hd(:,i))).^2)/varianceEstimate_power),1);
end
%%
rate_fromdata = gather(rate_fromdata);
bestconfig_fromdata=zeros(N,nbrOfUsers);
maxrate_fromconfig=zeros(nbrOfUsers,1);
for i=1:nbrOfUsers
    [maxrate_fromconfig(i),ind]=max(rate_fromdata(:,i));
    bestconfig_fromdata(:,i)=pilotMatrix(:,ind);
end
%%
% implay(reshape(bestconfig_fromdata,64,64,nbrOfUsers))
%% calculate the stm
%% calc F_M
Vtime = zeros(20,4096,50);
hdtime = zeros(20,50);
for i=1:nbrOfUsers
    Vtime(:,:,i)=F'*V(:,:,i)/500;
    hdtime(:,i)=F'*hd(:,i)/500;
end
%% finding the strongest tap
omega_stmforeverytap=gpuArray(zeros(M,N,nbrOfUsers));
omega_optstm=gpuArray(zeros(N,nbrOfUsers));
absulotval = gpuArray(zeros(M,nbrOfUsers));
strongestap=gpuArray(1:nbrOfUsers);
rate_optstm=gpuArray(1:nbrOfUsers);
for i=1:nbrOfUsers
    omega_stmforeverytap(:,:,i)=exp(1j*(angle(hdtime(:,i)*ones(1,N))-angle(Vtime(:,:,i))));
    for l=1:M
        absulotval(l,i)=(abs(Vtime(l,:,i)*omega_stmforeverytap(l,:,i).'+hdtime(l,i)));
    end
    [m,strongestap(i)]=max(absulotval(:,i));
    omega_optstm(:,i)=omega_stmforeverytap(strongestap(i),:,i);
    rate_optstm(i)=rate(V(:,:,i),omega_optstm(:,i),hd(:,i),B,K,M,varianceEstimate_power);
end
figure(1)
hold on
plot(1:50,rate_optstm,'ro', 'displayname', 'Optimal rate from given configurations')
plot(1:50,maxrate_fromconfig,'bo', 'displayname', 'The STM continuous phase rate')
pause(1);
theta_rates = zeros(50,1);
load('C:\Users\tomerf\Documents\FinalResult\theta.mat');
for user = 1 : nbrOfUsers
    theta_rates(user) = rate(V(:,:,user),theta(:,user),hd(:,user),B,K,M,varianceEstimate_power);
end
plot(1:50,theta_rates,'ko', 'displayname', 'theta rates')
disp('Done.');
clear rate_fromdata
clear F
clear noisetemp
clear pilotMatrix
clear omega_stmforeverytap
clear absulotval
%% STM enhence
global_opt_config = omega_optstm;
global_opt_rate = gpuArray(rate_optstm);
EnrEnh = gpuArray(zeros(4096, 50));
EnrEnhRate = gpuArray(zeros(50,1));
for user = 1 : 50
    user
    channel = V(:,:,user);
    %cur_eng = @(theta) norm(channel*theta.*X).^2;
    theta0 = omega_optstm(:,user);
    %VVx2 = channel'*channel;
    %grad = @(theta) VVx2*theta;
    EnrEnh(:,user) = grad_desc(theta0, true(4096,1), 2000, channel, hd(:,user), B, K, M, varianceEstimate_power);
    EnrEnhRate(user) = rate(channel,theta0,hd(:,user),B,K,M,varianceEstimate_power);
    if EnrEnhRate(user) > rate_optstm(user)
        disp('Woot Woot')
        global_opt_config(:,user) = EnrEnh(:,user);
        global_opt_rate(user) = EnrEnhRate(user);
    end
end
% plot(1:50, global_opt_rate, 'o');
% pause(1);
clear EnrEnh
clear EnrEnhRate
clear omega_optstm
clear rate_optstm
%% calculate naive quantization

disp('Naive quantization...');
clc
improv = gpuArray(zeros(1,50));
variance = gpuArray(zeros(1,50));
quant_values = gpuArray(zeros(2,50));
quant_best_rate = gpuArray(zeros(1,50));
quant_best_configurations = gpuArray(zeros(4096, 50));
global_opt_config = gpuArray(global_opt_config);
% figure; hold all;
for examp = 1 : 50
    user = examp
    theta0 = global_opt_config(:,examp);
    channel = V(:,:,examp);
%     cur_eng = @(theta) norm(channel*theta.*X).^2;
%     VVx2 = channel'*channel;
%     grad = @(theta) VVx2*theta;
%     theta0 = grad_desc(theta0, grad, true(4096,1), 2000, cur_eng);
    quant_config = ones(4096,1);
    quant_config((angle(theta0)>-pi/2) & (angle(theta0) < pi/2)) = -1;
    quant_best_rate(examp) = rate(channel,quant_config,hd(:,user),B,K,M,varianceEstimate_power);
    quant_best_configurations(:,examp) = quant_config;
    quant_values(:, examp) = [pi/2; 3*pi/2]; 
%     plot(examp, 10*log10(cur_eng(quant_config)), 'k+');
%     subplot(5,10,examp); hold all;
%     plot(0:100:1000*pi, 10*log10(Ratio(theta0, channel)), 'r');
%     plot(0: 100: 1000*pi, 10*log10(Ratio(quant_config, channel)), 'k');
    for cut = 0 : 0.01 : pi
        quant_config_bet = ones(4096,1);
        meanmean = mean(pi + angle(theta0));
        quant_config_bet((pi + angle(theta0)> cut) & (pi + angle(theta0) < pi+cut)) = -1;
        if rate(channel,quant_config_bet,hd(:,user),B,K,M,varianceEstimate_power) > quant_best_rate(examp)
            quant_best_rate(examp) = rate(channel,quant_config_bet,hd(:,user),B,K,M,varianceEstimate_power);
            quant_values(:, examp) = [cut; cut+pi]; 
            quant_best_configurations(:, examp) = quant_config_bet;
        end
%         disp(['user ', num2str(examp)]);
%         disp(['energy from continuous phase: ', num2str(10*log10(cur_eng(best_config(:, examp))))]);
%         disp(['energy from current quantization: ', num2str(10*log10(cur_eng(quant_config)))]);
%         disp(['energy from improved quantiztion: ', num2str(10*log10(cur_eng(quant_config_bet)))]);
%         dist = abs(abs(angle(best_config(:, examp))) - mean(abs(angle(best_config(:, examp)))));
%         plot(cut*1000, 10*log10(rate(channel,quant_config_bet,hd(:,user),B,K,M,varianceEstimate_power)), 'bo');
    end
%     pause(0.05);
%     improv(examp) = 10*log10(gather(Ratio(best_config(:, examp), channel)))-10*log10(gather(Ratio(quant_best_configurations(:, examp), channel)));
%     variance(examp) = var(angle(best_config(:, examp)));

%     plot(0:100:1000*pi, 10*log10((Ratio(best_config(:,examp), channel))), 'r');
%     plot(0: 100: 1000*pi, 10*log10((Ratio(quant_config, channel))), 'k');
end
% legend({'old quantization', 'continuous phase'});
% figure;
% plot(variance, improv, 'r+');
clear improv
clear variance
clear quant_best_rate 
clear quant_best_configurations 
disp('Done.');
%% GD

disp('GQA optimization...');
process_config24 = [];
rate_in_process24 = zeros(N, 50);
global_opt_config = gather(global_opt_config);
quant_values = gather(quant_values);
V = gather(V);
hd = gather(hd);
for user = [11,16,30,32,33,35,38,45,9,10,13]
    reset(gpu);
    V_user = gpuArray(V(:,:,user));
    hd_user = gpuArray(hd(:,user));
    quants = gpuArray([0,pi]);
    disp(['user number ', num2str(user)]);
    NUM_ITER = 500;
    RANDOM_FACTOR = 1;
    channel = V_user;
%     cur_eng = @(theta) norm(channel*theta.*X).^2;
    theta0 = gpuArray(theta(:,user));
%     VVx2 = channel'*channel;
%     grad = @(theta) VVx2*theta;
    theta_gd = theta0.*exp(1j * 2*pi*(rand(N,1)-0.5)*RANDOM_FACTOR);
    distance_from_quant = sin(angle(theta0) - quants(1)).^2;
    Quant_ones = true(N,1);
    
    for n = 1 : N
        if ~mod(n, 1024)
            disp(num2str(n));
        end
        NUM_ITER = 1500;
        [~,imin] = max(distance_from_quant);
        
        Quant_ones(imin) = false;       
        distance_from_quant = sin(angle(theta_gd) - quant_values(1,user)).^2;
        distance_from_quant(find(Quant_ones == false)) = nan;
        theta_gd = plusminusDilema(theta_gd, channel, imin, quants, Quant_ones, NUM_ITER, X, hd_user,B,K,M,varianceEstimate_power);
        improv_md = true;
        while (mod(n,16) == 0) && (improv_md == true)
            [theta_gd, improv_md] = backCheck(theta_gd, Quant_ones, channel,hd_user,B,K,M,varianceEstimate_power);
            if improv_md
                theta_gd = grad_desc(theta_gd, Quant_ones, NUM_ITER,channel,hd_user,B,K,M,varianceEstimate_power);
            end
        end
        rate_in_process24(n, user) = gather(rate(channel,theta_gd,hd_user,B,K,M,varianceEstimate_power));
        
%     subplot(3,1,1); hold all;
%         plot(10*log10(rate_in_process), 'c*');
%     subplot(3,1,2); polarplot(theta_gd, 'r+');
%     subplot(3,1,3); plot(angle(best_config(find(~isinf(distance_from_quant)), user))/pi*180,distance_from_quant(find(~isinf(distance_from_quant))), 'r+');
%         pause(0.1);
    end
    process_config24 = [process_config24, gather(theta_gd)];
    plot(user, rate_in_process24(N,user), 'gd');
    pause(1);
end

disp('Done.');
%% finish the process

disp('Plus minus one classification...');
SolutionMay = process_config24;
figure;
solution_rates25 = zeros(50,1);
for user = 1 : 50
    solution_rates25(user) = rate(V(:,:,user),SolutionMay(:,user),hd(:,user),B,K,M,varianceEstimate_power);
end
plot(solution_rates25, 'r+'); hold all;

rate_for_grade = solution_rates25;
rate_for_grade(rate_for_grade<9e7) = 2 * rate_for_grade(rate_for_grade<9e7); 
% plot(rate_for_grade, 'bo');

%% +-1
oneminusone_rate24 = zeros(50,1);
for user = 1 : 50
    plus = real(SolutionMay(:,user) ./ SolutionMay(1, user));
    minus = -1*real(SolutionMay(:,user) ./ SolutionMay(1, user));
    plus_rate = rate(V(:,:,user),plus,hd(:,user),B,K,M,varianceEstimate_power);
    minus_rate = rate(V(:,:,user),minus,hd(:,user),B,K,M,varianceEstimate_power);
    if plus_rate > minus_rate
        SolutionMay(:,user) = plus;
    else
        SolutionMay(:,user) = minus;
    end
    SolutionMay = sign(SolutionMay);
    oneminusone_rate24(user) = rate(V(:,:,user),SolutionMay(:,user),hd(:,user),B,K,M,varianceEstimate_power);
end
plot(oneminusone_rate24, 'c>');

disp('Well... Done.');
%% fixing
% 
% for user = 1 : 50
%     
%     if oneminusone_rate24(user) > oneminusone_rate(user)
%         disp(['change in user ', num2str(user), ' from ', num2str(oneminusone_rate(user)), ' to ',num2str(oneminusone_rate24(user))]); 
%         oneminusone_rate(user) = oneminusone_rate24(user);
%         SolutionMay(:,user) = SolutionMay(:,user);
%     end
% end
%% trying realy hard

% userToTry = [47 28 49 50];

% trying_rates = zeros(10, 50);
% trying_final_configs = zeros(4096, 10, 50);
worstUsersConfigs = [SolutionMay(:,10),SolutionMay(:,11),SolutionMay(:,32),SolutionMay(:,33),SolutionMay(:,44)]; 
for user = [10,11,32,33,44]
    user
    channel = V(:,:,user);
%     figure; hold all; plot(1:10, solution_rates(user).*ones(10,1), 'r');
    for i = 1 : 100
        i
        chainchange = (rand(4096,1) > 0.80);
        theta0 = global_opt_config(:,4).*exp(1j*chainchange.*(rand(4096,1)-0.5)*2*pi);
        theta_gd = grad_desc(theta0, true(4096,1), 2000,channel,hd(:,user),B,K,M,varianceEstimate_power);
 
        distance_from_quant = sin(angle(theta_gd) - quant_values(1,user)).^2;
        Quant_ones = true(N,1);
%         disp(num2str(0))
        for n = 1 : N
%             if ~mod(n, 1024)
%                 disp(num2str(n));
%             end
            NUM_ITER = 1500;
            [~,imin] = max(distance_from_quant);

            Quant_ones(imin) = false;       
            distance_from_quant = sin(angle(theta_gd) - quant_values(1,user)).^2;
            distance_from_quant(find(Quant_ones == false)) = nan;
            theta_gd = plusminusDilema(theta_gd, channel, imin, quant_values(:,user), Quant_ones, NUM_ITER, X, hd(:,user),B,K,M,varianceEstimate_power);
            improv_md = true;
            while (mod(log2(n),1) == 0) && (improv_md == true)
                [theta_gd, improv_md] = backCheck(theta_gd, Quant_ones, channel,hd(:,user),B,K,M,varianceEstimate_power);
                if improv_md
                    theta_gd = grad_desc(theta_gd, Quant_ones, NUM_ITER,channel,hd(:,user),B,K,M,varianceEstimate_power);
                end
            end
        end
%         plot(i,trying_rates(i, user), 'bo');
%         pause(1);
        idx = find([10,11,32,33,44] == user);
        rate_gd = rate(V(:,:,user),theta_gd,hd(:,user),B,K,M,varianceEstimate_power);
        if solution_rates25(user) < rate_gd
            disp(['change made in user ', num2str(user), ' from ', num2str(solution_rates25(user)), ' to ', num2str(rate_gd)]);
            solution_rates25(user) = rate_gd;
            worstUserConfigs(:,idx) = theta_gd;
        end
    end

end

function theta_grad = grad_desc(theta0, Quant_ones, NUM_ITER,V,h,B,K,M,N0)

    theta_grad = theta0;
    i = 1;
    diff = 1;
    previous_rate = rate(V,theta0,h,B,K,M,N0);
    while (i <= NUM_ITER) && (diff > 200)
        gd = gradRate(V,theta_grad,h,B,K,M,N0);
        upd_idx = find(Quant_ones == true);
        ang_theta_gd = angle(theta_grad+100*gd);
        theta_grad(upd_idx) = exp(1j*(ang_theta_gd(upd_idx)));
        next_rate = rate(V,theta_grad,h,B,K,M,N0);
        diff = next_rate - previous_rate;
        previous_rate = next_rate;
        if length(find(abs(theta_grad) > 1)) > 1
            disp('ERROR!');
            disp(num2str(max(abs(theta_gd))));
        end
        i = i + 1;
    end

end


function [theta_check, improvment_made] = backCheck(theta0, Quant_ones, channel, hd,B,K,M,N0)
    
    index_to_check = find(Quant_ones == false);
    HowMany = length(index_to_check);
    theta_check = theta0;
    RateToCompare = rate(channel,theta0,hd,B,K,M,N0);
    improvment_made = false;
    for i = 1 : HowMany
        theta_try = theta_check;
        theta_try(index_to_check(i)) = theta_check(index_to_check(i)) .* -1;
        if rate(channel,theta_try,hd,B,K,M,N0) > RateToCompare
           RateToCompare = rate(channel,theta_try,hd,B,K,M,N0);
           theta_check = theta_try;
           improvment_made = true;
        end
    end
    
end

function ratio_grad = rate_grad(theta0, V, P, N0, B, K, M)
    ratio_grad = zeros(length(theta0),1);
    h_theta = V*theta0;
    for rgi = 1 : length(theta0)
        
        ratio_grad(rgi) = 2*P*B /(K+M-1) * sum(h_theta./((P*h_theta.^2+B*N0)*log(2)) .* V(:,rgi));  
    
    end
end

function Qtheta = plusminusDilema(theta, channel, index, quant_values, Quant_ones, NUM_ITER, X,hd,B,K,M,N0)
    
    plus = theta;
    plus(index) = exp(1j*quant_values(1));
    plus = grad_desc(plus, Quant_ones, NUM_ITER,channel,hd,B,K,M,N0);
    minus = theta;
    minus(index) = exp(1j*quant_values(2));
    minus = grad_desc(minus, Quant_ones, NUM_ITER,channel,hd,B,K,M,N0);
    
    if rate(channel,minus,hd,B,K,M,N0) > rate(channel,plus,hd,B,K,M,N0)
        Qtheta = minus;
    else
        Qtheta = plus;
    end
end

function ratei = rate(V,theta,h,B,K,M,N0)
    ratei=(B/(K+M-1))*sum(log2(1+((abs(V*theta+h)).^2)/N0),1);
end

function gradi = gradRate(V,theta,h,B,K,M,N_0)
    temp = N_0*ones(K,1)+(abs(V*theta+h)).^2;
    temp2 = 2*(V*theta+h)./temp;
    gradi=(B/(log(2)*(K+M-1)))*V'*temp2;
end

function signal_clean = conv_padding(signal, filter)
    % padding
    padding_size = floor(length(filter)/2);
    data_sig = [signal(1)*ones(padding_size,1);...
            signal;...
            signal(end)*ones(padding_size,1)];
    signal_padded = conv(data_sig, filter,'same');
    % remove padding
    signal_clean = signal_padded(padding_size+1:length(signal_padded)-padding_size);
end
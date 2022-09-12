clear all
close all
clc

N = 100:50:1200; %number of devices
L = 100; %number of packets
K = 400; %number of RA slots

alpha = 0.1; %learning rate
Kvec = 1:K; %aux vector for Q learning

% Initializing performance vectors
throughput_ACB_avg = zeros(1,length(N)); 
throughput_sALOHA_avg = zeros(1,length(N));
throughput_QMTC_avg = zeros(1,length(N));
throughput_cQMTC_avg = zeros(1,length(N));
throughput_ACB_QMTC_avg = zeros(1,length(N));
throughput_ACB_cQMTC_avg = zeros(1,length(N));

throughput_ACB_metrica_avg = zeros(1,length(N));
throughput_sALOHA_metrica_avg = zeros(1,length(N));
throughput_cQMTC_metrica_avg = zeros(1,length(N));
throughput_QMTC_metrica_avg = zeros(1,length(N));
throughput_ACB_cQMTC_metrica_avg = zeros(1,length(N));
throughput_ACB_QMTC_metrica_avg = zeros(1,length(N));


for indN=1:length(N) % For each number of devices
    fprintf(['\nProgresso: N = ',num2str(N(indN))]) 
    throughput_ACB = zeros(1,L); 
    throughput_sALOHA = zeros(1,L);
    throughput_QMTC = zeros(1,L); 
    throughput_cQMTC = zeros(1,L);
    throughput_ACB_QMTC = zeros(1,L); 
    throughput_ACB_cQMTC = zeros(1,L);
    
    throughput_ACB_metrica = zeros(1,L); 
    throughput_sALOHA_metrica = zeros(1,L);
    throughput_QMTC_metrica = zeros(1,L); 
    throughput_cQMTC_metrica = zeros(1,L);
    throughput_ACB_QMTC_metrica = zeros(1,L); 
    throughput_ACB_cQMTC_metrica = zeros(1,L);
    
    Qtable_QMTC = zeros(K,N(indN)); 
    Qtable_cQMTC = zeros(K,N(indN)); 
    Qtable_ACB_QMTC = zeros(K,N(indN)); 
    Qtable_ACB_cQMTC = zeros(K,N(indN));
    
    succ_transm_ACB = zeros(L, N(indN));
    succ_transm_ACB_QMTC = zeros(L, N(indN));
    succ_transm_ACB_cQMTC = zeros(L, N(indN));
    
    ACB_p = K/N(indN);
    for ind=1:L
        % slotted ALOHA
        slot_sel_sALOHA = randi([1,K],[N(indN),1]); %Each device randomly select a slot to transmit
        slot_succ_sALOHA = sum(sum(slot_sel_sALOHA==slot_sel_sALOHA')==1); %Counting how many devices didn't collide 
        throughput_sALOHA(1,ind) = slot_succ_sALOHA/K; %Computing the ratio
        throughput_sALOHA_metrica(1,ind) = get_new_metrica(slot_succ_sALOHA, N(indN), K);
        
        
         % Metodo ACB-sALOHA
        q = rand([1 N(indN)]);
        act_devices = (q < ACB_p);

        slot_sel_ACB = randi([1,K],[1 N(indN)]).*act_devices; %Each device randomly select a slot to transmit
        succ_transm_ACB(ind,:) = (sum(slot_sel_ACB==slot_sel_ACB')==1).*act_devices; % inicializar esta variável LxN
        slot_succ_ACB = sum(succ_transm_ACB(ind,:)); %Counting how many devices didn't collide
        throughput_ACB(1,ind) = slot_succ_ACB/K; %Computing the ratio
        throughput_ACB_metrica(1,ind) = get_new_metrica(slot_succ_ACB, N(indN), K);
        
        
         % Q-learning independente
        valid_mask_QMTC = (Qtable_QMTC==max(Qtable_QMTC)); %Finding best slots to transmit based on Q-table
        slot_sel_QMTC = zeros(1,N(indN)); %Initializing vector with selected slots
        for n=1:N(indN) %If multiple slots are equally good, randomly select one of them
            valid_slots = Kvec(valid_mask_QMTC(:,n));
            choice = randi([1 sum(valid_mask_QMTC(:,n))]);
            slot_sel_QMTC(1,n) = valid_slots(choice);
        end
        device_succ_QMTC = sum(slot_sel_QMTC==slot_sel_QMTC')==1; %Finding devices which didn't collide
        reward_QMTC = 2*device_succ_QMTC-1; %Mapping 0 and 1 to -1 and 1
        reward_QMTC_matrix = zeros(K,N(indN)); %Initializing reward matrix
        for n=1:N(indN)
            reward_QMTC_matrix(slot_sel_QMTC(1,n),n) = reward_QMTC(n);
        end
        Qtable_QMTC = Qtable_QMTC+alpha*(reward_QMTC_matrix-Qtable_QMTC); %Updating Q-table
        slot_succ_QMTC = sum(device_succ_QMTC); %Counting how many devices didn't collide
        throughput_QMTC(1,ind)=slot_succ_QMTC/K; %Computing the ratio
        throughput_QMTC_metrica(1,ind) = get_new_metrica(slot_succ_QMTC, N(indN), K);
        
        
        % Q-learning colaborativo
        valid_mask_cQMTC = (Qtable_cQMTC==max(Qtable_cQMTC)); %Finding best slots to transmit based on Q-table
        slot_sel_cQMTC = zeros(1,N(indN)); %Initializing vector with selected slots
        slots_time = zeros(1,K); %Counting how many devices selected the same slot
        for n=1:N(indN) %If multiple slots are equally good, randomly select one of them
            valid_slots = Kvec(valid_mask_cQMTC(:,n));
            choice = randi([1 sum(valid_mask_cQMTC(:,n))]);
            slot_sel_cQMTC(1,n) = valid_slots(choice);
            slots_time(1,slot_sel_cQMTC(1,n)) = slots_time(1,slot_sel_cQMTC(1,n))+1;
        end
        CL = slots_time/N(indN); %Computing ratio to be used as negative reward for colliding devices
        device_succ_cQMTC = sum(slot_sel_cQMTC==slot_sel_cQMTC')==1; %Finding devices which didn't collide  
        reward_cQMTC_matrix = zeros(K,N(indN)); %Initializing reward matrix
        for n=1:N(indN)
            if(device_succ_cQMTC(1,n))
                reward_cQMTC_matrix(slot_sel_cQMTC(1,n),n) = 1; %Positive Reward
            else
                reward_cQMTC_matrix(slot_sel_cQMTC(1,n),n) = -CL(1,slot_sel_cQMTC(1,n)); %Negative Reward
            end
        end
        
        Qtable_cQMTC = Qtable_cQMTC+alpha*(reward_cQMTC_matrix-Qtable_cQMTC); %Updating Q-table
        slot_succ_cQMTC = sum(device_succ_cQMTC); %Counting how many devices didn't collide
        throughput_cQMTC(1,ind)=slot_succ_cQMTC/K; %Computing the ratio
        throughput_cQMTC_metrica(1,ind) = get_new_metrica(slot_succ_cQMTC, N(indN), K);
        
        
         % Q-learning com ACB
        q = rand([1 N(indN)]);
        act_devices = (q < ACB_p);
        
        valid_mask_ACB_QMTC = (Qtable_ACB_QMTC==max(Qtable_ACB_QMTC)); %Finding best slots to transmit based on Q-table
        slot_sel_ACB_QMTC = zeros(1,N(indN)); %Initializing vector with selected slots
        for n=1:N(indN) %If multiple slots are equally good, randomly select one of them
            valid_slots = Kvec(valid_mask_ACB_QMTC(:,n));
            choice = randi([1 sum(valid_mask_ACB_QMTC(:,n))]);
            slot_sel_ACB_QMTC(1,n) = valid_slots(choice);
        end
        
        slot_sel_ACB_QMTC = slot_sel_ACB_QMTC.*act_devices;
        succ_transm_ACB_QMTC(ind,:) = (sum(slot_sel_ACB_QMTC==slot_sel_ACB_QMTC')==1).*act_devices; %Finding devices which didn't collide
        reward_ACB_QMTC = 2*succ_transm_ACB_QMTC(ind,:)-1; %Mapping 0 and 1 to -1 and 1
        reward_ACB_QMTC_matrix = zeros(K,N(indN)); %Initializing reward matrix
        for n=1:N(indN)
            if act_devices(n)
                reward_ACB_QMTC_matrix(slot_sel_ACB_QMTC(1,n),n) = reward_ACB_QMTC(n);
            end
        end
        Qtable_ACB_QMTC = Qtable_ACB_QMTC+alpha*(reward_ACB_QMTC_matrix-Qtable_ACB_QMTC); %Updating Q-table
        slot_succ_ACB_QMTC = sum(succ_transm_ACB_QMTC(ind,:)); %Counting how many devices didn't collide
        throughput_ACB_QMTC(1,ind)=slot_succ_ACB_QMTC/K; %Computing the ratio
        throughput_ACB_QMTC_metrica(1,ind) = get_new_metrica(slot_succ_ACB_QMTC, N(indN), K);
        
        
        % Q-learning colaborativo com ACB
        q = rand([1 N(indN)]);
        act_devices = (q < ACB_p);
        
        valid_mask_ACB_cQMTC = (Qtable_ACB_cQMTC==max(Qtable_ACB_cQMTC)); %Finding best slots to transmit based on Q-table
        
        slot_sel_ACB_cQMTC = zeros(1,N(indN));
        slots_time = zeros(1,K); %Counting how many devices selected the same slot
        for n=1:N(indN) %If multiple slots are equally good, randomly select one of them
            valid_slots = Kvec(valid_mask_ACB_cQMTC(:,n));
            choice = randi([1 sum(valid_mask_ACB_cQMTC(:,n))]);
            slot_sel_ACB_cQMTC(1,n) = valid_slots(choice);
            slots_time(1,slot_sel_ACB_cQMTC(1,n)) = slots_time(1,slot_sel_ACB_cQMTC(1,n))+1;
        end 
        
        CL = slots_time/sum(act_devices); %Computing ratio to be used as negative reward for colliding devices
        slot_sel_ACB_cQMTC = slot_sel_ACB_cQMTC.*act_devices;
        succ_transm_ACB_cQMTC(ind,:) = (sum(slot_sel_ACB_cQMTC==slot_sel_ACB_cQMTC')==1).*act_devices; %Finding devices which didn't collide  
        reward_ACB_cQMTC_matrix = zeros(K,N(indN)); %Initializing reward matrix
        for n=1:N(indN)
            if act_devices(n)
                if(succ_transm_ACB_cQMTC(ind,n))
                    reward_ACB_cQMTC_matrix(slot_sel_ACB_cQMTC(1,n),n) = 1; %Positive Reward
                else
                    reward_ACB_cQMTC_matrix(slot_sel_ACB_cQMTC(1,n),n) = -CL(1,slot_sel_ACB_cQMTC(1,n)); %Negative Reward
                end
            end
        end
        Qtable_ACB_cQMTC = Qtable_ACB_cQMTC+alpha*(reward_ACB_cQMTC_matrix-Qtable_ACB_cQMTC); %Updating Q-table
        slot_succ_ACB_cQMTC = sum(succ_transm_ACB_cQMTC(ind,:)); %Counting how many devices didn't collide
        throughput_ACB_cQMTC(1,ind)=slot_succ_ACB_cQMTC/K; %Computing the ratio
        throughput_ACB_cQMTC_metrica(1,ind) = get_new_metrica(slot_succ_ACB_cQMTC, N(indN), K);
        
    end
    
    throughput_sALOHA_avg(1,indN) = mean(throughput_sALOHA);
    throughput_ACB_avg(1,indN) = mean(throughput_ACB);
    throughput_QMTC_avg(1,indN) = mean(throughput_QMTC);
    throughput_cQMTC_avg(1,indN) = mean(throughput_cQMTC);
    throughput_ACB_QMTC_avg(1,indN) = mean(throughput_ACB_QMTC);
    throughput_ACB_cQMTC_avg(1,indN) = mean(throughput_ACB_cQMTC);
    
    % Computing performances for each scheme 
    throughput_sALOHA_metrica_avg(1,indN) = mean(throughput_sALOHA_metrica);
    throughput_ACB_metrica_avg(1,indN) = mean(throughput_ACB_metrica);
    throughput_QMTC_metrica_avg(1,indN) = mean(throughput_QMTC_metrica);
    throughput_cQMTC_metrica_avg(1,indN) = mean(throughput_cQMTC_metrica);
    throughput_ACB_QMTC_metrica_avg(1,indN) = mean(throughput_ACB_QMTC_metrica);
    throughput_ACB_cQMTC_metrica_avg(1,indN) = mean(throughput_ACB_cQMTC_metrica);
end

% Analytical performance
analyt_throughput_ACB = N.*(((K-1)/K).^(N-1))./K;
analyt_throughput_sALOHA = N.*(((K-1)/K).^(N-1))./K;

figure
plot(N,throughput_sALOHA_avg,'-k',N,throughput_ACB_avg,'bo:',N,throughput_QMTC_avg,'ro:',N,throughput_cQMTC_avg,'rx:',N,throughput_ACB_QMTC_avg,'go:',N,throughput_ACB_cQMTC_avg,'gx:')
xlabel('Number of Devices (\it{N})')
ylabel('Normalized Throughput')
legend('sALOHA', 'ACB-sALOHA', 'iQMTC', 'cQMTC', 'ACB-iQMTC', 'ACB-cQMTC')

figure
plot(N,throughput_sALOHA_metrica_avg,'-k',N,throughput_ACB_metrica_avg,'bo:',N,throughput_QMTC_metrica_avg,'ro:',N,throughput_cQMTC_metrica_avg,'rx:',N,throughput_ACB_QMTC_metrica_avg,'go:',N,throughput_ACB_cQMTC_metrica_avg,'gx:')
xlabel('Number of Devices (\it{N})')
ylabel('Performance')
legend('sALOHA', 'ACB-sALOHA', 'iQMTC', 'cQMTC', 'ACB-iQMTC', 'ACB-cQMTC')


function [metrica] = get_new_metrica(succ, n, K)
    if (n < K)
       metrica = succ/n; % metrica de performance
    else
       metrica = succ/K; 
    end
end

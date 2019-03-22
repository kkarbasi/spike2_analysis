%%
clear all
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B082107/B082107_1340_List.smr.mat') 
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B082207/B082207_1505_List.smr.mat') % % only two events, bs
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B110807/B110807_1632_List.smr.mat')% % only two events, bs
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B090407/B090407_1526_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B091208/B091208_1545_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B091608/B091608_1208_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B091707/B091707_1414_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B091908_1/B091908_1_1500_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B091908_2/B091908_2_1550_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B093008/B093008_1149_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B100308_2/B100308_2_1403_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B101707/B101707_1333_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B111507/B111507_1528_List.smr.mat') % % only two events, bs
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B121407_1/B121407_1_1124_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/B121807/B121807_1135_List.smr.mat') % % only two events, bs

% load('/home/kkarbasi/dmount/data/david_neurons_mat/W091008/W091008_1241_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/W091208/W091208_1337_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/W120108/W120108_1622_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/W120208_1/W120208_1_1418_List.smr.mat') % like the first
load('/home/kkarbasi/dmount/data/david_neurons_mat/W120308_2/W120308_2_1714_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/W120508_1/W120508_1_1420_List.smr.mat') % like the first
% load('/home/kkarbasi/dmount/data/david_neurons_mat/W120508_2/W120508_2_1628_List.smr.mat') % like the first

% load('/home/kkarbasi/dmount/data/david_neurons_mat/F090606/F090606_1313_List.smr.mat') 
% in
% /home/kkarbasi/dmount/data/david_neurons_mat/F090606/F090606_1313_List.smr.mat:
% 3: primary target presentation 
% 5: primary and corrective presentation
% 6: trial onset. In these files the trials are presentation in
% forward and reverse in just one direction

% load('/home/kkarbasi/dmount/data/david_neurons_mat/F091106/F091106_0949_List.smr.mat') % like the first 
% load('/home/kkarbasi/dmount/data/david_neurons_mat/F100906/F100906_1343_List.smr.mat') % like the first  
% load('/home/kkarbasi/dmount/data/david_neurons_mat/F100906/F100906_1343_List.smr.mat') % like the first  
% load('/home/kkarbasi/dmount/data/david_neurons_mat/F101106/F101106_1311_List.smr.mat') % like the first  
% load('/home/kkarbasi/dmount/data/david_neurons_mat/F101606/F101606_1308_List.smr.mat') % like the first  
% load('/home/kkarbasi/dmount/data/david_neurons_mat/F102006/F102006_0916_List.smr.mat') % like the first  
% load('/home/kkarbasi/dmount/data/david_neurons_mat/F102506/F102506_1506_List.smr.mat') % like the first  
% load('/home/kkarbasi/dmount/data/david_neurons_mat/F102706/F102706_1435_List.smr.mat') % like the first  
% load('/home/kkarbasi/dmount/data/david_neurons_mat/F110206/F110206_1400_List.smr.mat') % like the first  
% load('/home/kkarbasi/dmount/data/david_neurons_mat/F110706/F110706_1413_List.smr.mat') % like the first  
% load('/home/kkarbasi/dmount/data/david_neurons_mat/F112006/F112006_1446_List.smr.mat') % like the first  

%% get all events
labels = block.segments{1,1}.events{1,end}.labels;
unique_events = unique(labels, 'rows');
%     '1          '
%     '100        '
%     '102        '
%     '18         '
%     '36         '
%     '37         '
%     '38         '
%     '64         '
%     '78         '

label_times = block.segments{1,1}.events{1,end}.times;
% event_id = 1;
for event_id = 1:size(unique_events, 1)
    event_times{event_id} = label_times(find(ismember(labels,unique_events(event_id,:), 'rows')));
end

%%
sac_o_times = block.segments{1,1}.events{1,1}.times;
sac_e_times = block.segments{1,1}.events{1,2}.times;

%% create 2d target and eye signals
eye_sampling_rate = block.segments{1,1}.analogsignals{1,1}.sampling_rate;
eye_dt = 1.0/eye_sampling_rate;

HE = block.segments{1,1}.analogsignals{1,1}.signal;
t_HE = (0 : numel(HE)-1) * ...
    (1.0/block.segments{1,1}.analogsignals{1,1}.sampling_rate) + ...
    block.segments{1,1}.analogsignals{1,1}.t_start;

VE = block.segments{1,1}.analogsignals{1,2}.signal;
t_VE = (0 : numel(VE)-1) * ...
    (1.0/block.segments{1,1}.analogsignals{1,2}.sampling_rate) + ...
    block.segments{1,1}.analogsignals{1,2}.t_start;

HT = block.segments{1,1}.analogsignals{1,3}.signal;
t_HT = (0 : numel(HT)-1) * ...
    (1.0/block.segments{1,1}.analogsignals{1,3}.sampling_rate) + ...
    block.segments{1,1}.analogsignals{1,3}.t_start;

VT = block.segments{1,1}.analogsignals{1,4}.signal;
t_VT = (0 : numel(VT)-1) * ...
    (1.0/block.segments{1,1}.analogsignals{1,4}.sampling_rate) + ...
    block.segments{1,1}.analogsignals{1,4}.t_start;

%% ploting trials
t_min = 100; %s
t_max = 200; %s

clf
% t_range = (1+t_min*eye_sampling_rate : t_max*eye_sampling_rate);

plot(t_HT(t_HT > t_min & t_HT <= t_max), HT(t_HT > t_min & t_HT <= t_max), 'LineWidth', 2)
hold on
plot(t_VT(t_VT > t_min & t_VT <= t_max), VT(t_VT > t_min & t_VT <= t_max), 'LineWidth', 2)
hold on

plot(t_HE(t_HE > t_min & t_HE <= t_max), HE(t_HE > t_min & t_HE <= t_max), 'LineWidth', 2)
hold on
plot(t_VE(t_VE > t_min & t_VE <= t_max), VE(t_VE > t_min & t_VE <= t_max), 'LineWidth', 2)
hold on
% 
event_id = 2;
vline(event_times{event_id}(event_times{event_id} < t_max & event_times{event_id} > t_min), 'k') % trials

% vline(event_times{4}(event_times{4} < t_max & event_times{4} > t_min), 'y') % trials
% vline(event_times{5}(event_times{5} < t_max & event_times{5} > t_min), 'g') % primary target jump in adaptation trials
% vline(sac_o_times, 'g') % saccade onset
% vline(event_times{6}(event_times{6} < t_max & event_times{6} > t_min), 'g') % corrective target jump
% plot(event_times{5}(1:50), 7, 'r|')
% hold on
% plot(sac_o_times(1:50), 6, 'g*')
hold on
% plot(event_times{6}(1:50), 5, 'k*')
% hold on
legend('HT', 'VT', 'HE', 'VE')

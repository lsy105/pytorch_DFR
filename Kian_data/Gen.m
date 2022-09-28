clc;
clear all;
close all

%%%% Loading the user activity file
labs23=load('label23.mat');
labs24=load('label24.mat');
labs25=load('label25.mat');
%%%%% Simulating a MIMO-OFDM environemnt 
qpskMod = comm.QPSKModulator;
qpskDemod = comm.QPSKDemodulator;
ofdmMod = comm.OFDMModulator('FFTLength',40,...
    'NumGuardBandCarriers',[0;0],'NumTransmitAntennas',2);
ofdmDemod = comm.OFDMDemodulator(ofdmMod);
ofdmDemod.NumReceiveAntennas = 6;
showResourceMapping(ofdmMod)
ofdmModDim = info(ofdmMod);

numData = ofdmModDim.DataInputSize(1);   % Number of data subcarriers
numSym = ofdmModDim.DataInputSize(2);    % Number of OFDM symbols
numTxAnt = ofdmModDim.DataInputSize(3);     % Number of transmit antennas
numRxAnt = ofdmDemod.NumReceiveAntennas; % Number of Recieved antennas
%SNrdb=-30;
SNrdb=0;
snr = 10^(SNrdb/10);
filename = 'recieved1_0db_6_6.mat';
rng('default');

%labs=[labs23.label;labs25.label];

labs=[labs23.label;labs24.label;labs25.label];
nframes = 6102;
data = randi([0 3],nframes*numData,numSym,numTxAnt);
modData = qpskMod(data(:));
modData = reshape(modData,nframes*numData,numSym,numTxAnt);

errorRate = comm.ErrorRate;
save labsnew labs
for k = 1:nframes
    k
    %     onandoff = zeros(1,numData);
    %     labs = randperm(numData,7);
    %     onandoff(labs)= 1;    % Find row indices for kth OFDM frame
    indData = (k-1)*ofdmModDim.DataInputSize(1)+1:k*numData;
    label(k,:)=labs(k,:);
    % Generate random OFDM pilot symbols
    %pilotData = complex(rand(ofdmModDim.PilotInputSize), ...
    % rand(ofdmModDim.PilotInputSize));
    onandoff=labs(k,:);
    data1= modData(indData,:,:);
    data1(onandoff==0,:,:)=0;
    % Modulate QPSK symbols using OFDM
    data1=data1(:,:);
    %dataOFDM = ofdmMod(data1);
    % Create flat, i.i.d., Rayleigh fading channel
    
    % Pass OFDM signal through Rayleigh and AWGN channels
    %receivedSignal = awgn(data1*chGain,SNrdb);
    %rec(k,:,:)= receivedSignal;
    recieved(k,:)=data1(1,:);
    % Apply least squares solution to remove effects of fading channel
    
end
for i=1:nframes
    chGain = complex(randn(numTxAnt,numRxAnt),randn(numTxAnt,numRxAnt))/sqrt(.1); % Random 2x2 channel
    
    nn = complex(rand(1,numRxAnt));
    %data1= data1+nn;
    ss = sqrt(snr).*(recieved(i,:)*chGain); % Real valued Gaussina Primary User Signal
    %nn=nn./sum(abs(nn.^2));
    %ss=ss./sum(abs(ss.^2));
    
    receivedSignal(i,:) = (ss)+nn;
end
recieved1 = sum(abs(receivedSignal.^2)');
recieved1 = recieved1';
target= labs(1:nframes,1);
save(filename, 'recieved1')
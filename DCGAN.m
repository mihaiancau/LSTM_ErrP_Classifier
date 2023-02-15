%% Transform signals and labels
% Select signals and labels from one subject and trial
load 'C:\Users\Mihay\OneDrive - Technical University of Cluj-Napoca\Teza doctorat mama\Data\Subiecti - date echilibrate\S9\fc5_vaS9Antr_echi.mat';
% Convert categorical labels to array
labels = zeros(1,length(transpose(Labels))); % dim2 needs to be length(Labels)
for i = 1 : length(transpose(Labels))
    if (transpose(Labels(i)) == 'N') 
        labels(i) = 1; % standard
    else
        labels(i) = 2; % stimulated
    end
end
% Convert cellular signals to array of double
signals = cell2mat(Signals).'; % 231 x length(transpose(Labels)) array
% Standardize the signals to have zero mean and unit variance
meanSignals = mean(signals,2); % mean(A,dim) returns the mean along dimension dim
signalsNormalized = signals-meanSignals;
stdSignals = std(signalsNormalized(:)); % std(A) returns the standard deviation of the elements of A along the first array dimension whose size does not equal 1
signalsNormalized = signalsNormalized/stdSignals;

%% Generator Network
% Output: 231 x 1 x 1 array
numFilters = 64;
numLatentInputs = 100;
% This derives from the more general activations = input_size - (filter_size - 1),
% provided no padding and striding applied
projectionSize = [160 1 511]; % = N1 projection size is [N1 1 1]
filterSize = 5; % = f1 filter size is [1 f1]
numClasses = 2;
embeddingDimension = 100; % matches numLatentInputs

% Network layer array
layersGenerator = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    projectAndReshapeLayer(projectionSize,numLatentInputs,'proj');
    concatenationLayer(3,2,'Name','cat');
    transposedConv2dLayer([filterSize 1],4*numFilters,'Name','tconv1') 
    batchNormalizationLayer('Name','bn1','Epsilon',5e-5)
    reluLayer('Name','relu1')
    transposedConv2dLayer([2*filterSize 1],2*numFilters,'Name','tconv2')
    batchNormalizationLayer('Name','bn2','Epsilon',5e-5)
    reluLayer('Name','relu2')
    transposedConv2dLayer([4*filterSize 1],numFilters,'Name','tconv3') 
    batchNormalizationLayer('Name','bn3','Epsilon',5e-5)
    reluLayer('Name','relu3')
    transposedConv2dLayer([8*filterSize 1],1,'Name','tconv4') 
    ];

lgraphGenerator = layerGraph(layersGenerator);

layers = [
    imageInputLayer([1 1 1],'Name','labels','Normalization','none')
    % An embed and reshape layer takes as input numeric indices of categorical elements and converts them to images of the specified size. 
    % Use embed and reshape layers to input categorical data into conditional GANs.
    embedAndReshapeLayer(projectionSize(1:2),embeddingDimension,numClasses,'emb')];

lgraphGenerator = addLayers(lgraphGenerator,layers);
lgraphGenerator = connectLayers(lgraphGenerator,'emb','cat/in2');

dlnetGenerator = dlnetwork(lgraphGenerator);

%% Discriminator Network
% Input: 1 x 231 x 1 array
scale = 0.2; % for Leaky ReLU if x < 0, then f(x) = scale * x, otherwise f(x) = x
inputSize = [231 1 1];

% This derives from the more general 
% activations = ((input_size + 2*padding - filter_size)/stride) + 1
layersDiscriminator = [
    imageInputLayer(inputSize,'Normalization','none','Name','in')
    concatenationLayer(3,2,'Name','cat')
    convolution2dLayer([33 1],8*numFilters,'Stride',2,'Padding',[1 0],'Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    convolution2dLayer([16 1],4*numFilters,'Stride',2,'Padding',[1 0],'Name','conv2')
    leakyReluLayer(scale,'Name','lrelu2')
    convolution2dLayer([7 1],2*numFilters,'Stride',2,'Padding',[1 0],'Name','conv3')
    leakyReluLayer(scale,'Name','lrelu3')
    convolution2dLayer([3 1],numFilters,'Stride',2,'Padding',[1 0],'Name','conv4')
    leakyReluLayer(scale,'Name','lrelu4')
    convolution2dLayer([10 1],1,'Name','conv5')];

lgraphDiscriminator = layerGraph(layersDiscriminator);

layers = [
    imageInputLayer([1 1],'Name','labels','Normalization','none')
    embedAndReshapeLayer(inputSize,embeddingDimension,numClasses,'emb')];

lgraphDiscriminator = addLayers(lgraphDiscriminator,layers);
lgraphDiscriminator = connectLayers(lgraphDiscriminator,'emb','cat/in2');

dlnetDiscriminator = dlnetwork(lgraphDiscriminator);

%% Network training
% Training parameters
params.numLatentInputs = numLatentInputs;
params.numClasses = numClasses;
params.sizeData = [inputSize length(labels)];
params.numEpochs = 50; % 50
params.miniBatchSize = 16; % 32

% Specify the options for Adam optimizer
params.learnRate = 0.0001; % 0.0002
params.gradientDecayFactor = 0.5;
params.squaredGradientDecayFactor = 0.999;

executionEnvironment = "cpu";
params.executionEnvironment = executionEnvironment;

% Train the CGAN
[dlnetGenerator,dlnetDiscriminator] = trainGAN(dlnetGenerator, ...
        dlnetDiscriminator,signalsNormalized,labels,params); 
    
%% Setting up the synthesis of standard and stimulated epochs
% rng default

numTests = size(Labels,1); % the number of 1-by-1-by-100 arrays of random values to input into the generator network
ZNew = randn(1,1,numLatentInputs,numTests,'single');
dlZNew = dlarray(ZNew,'SSCB');

% Specify that 3/4 of arrays are standard and the rest are
% stimulated
numStd = length(labels(labels == 1));
numStim = length(labels(labels == 2));
TNew = ones(1,1,1,numTests,'single');
TNew(1,1,1,1:floor(numStim*numTests/length(labels))) = single(2);
dlTNew = dlarray(TNew,'SSCB');

%% Synthesis of standard and stimulated epochs
dlXGeneratedNew = predict(dlnetGenerator,dlZNew,dlTNew)*stdSignals+meanSignals;

idxGenerated = 1:numTests;
idxStim = idxGenerated(1:floor(numStim*numTests/length(labels))); % first 1/4 are fake stim epochs
idxStd = idxGenerated((length(idxStim) + 1):end); % last 3/4 are fake std epochs

% Extract signals
XGeneratedNew = squeeze(extractdata(gather(dlXGeneratedNew)));
% Extract labels
TGeneratedNew = zeros(1,numTests);
TGeneratedNew(idxStim) = 1;

%% Conversion of fake data to LSTM readable format
f = XGeneratedNew;
ft = TGeneratedNew;

% Convert fake signals to cell data
SignalsNew = num2cell(f,1);
SignalsNew = transpose(SignalsNew);
SignalsNew = cellfun(@transpose, SignalsNew, 'UniformOutput', false);
SignalsNew = cellfun(@double, SignalsNew, 'UniformOutput',false);
% Convert fake labels to categorical data
[m,n] = size(ft);
LabelsNew = cell(n,1);
for i = 1:n
    if ft(1,i) == 0
        LabelsNew{i,1} = 'N';
    else
        LabelsNew{i,1} = 'A';
    end
end
LabelsNew = categorical(LabelsNew);

%% Plot samples of the artificial epochs
t = 1:231;
n = 10;
splitInd = floor(numStim*numTests/length(labels));
figure;
% for k = 1 : n
%     subplot (2,2*n,k);
%     x = XGeneratedNew(:,k);
%     plot(t,x,'r');
%     title(sprintf('Subplot %d: Fake Stim',k));
% end
% for k = 1 : n
%     subplot (2,2*n,n+k);
%     x = XGeneratedNew(:,k);
%     plot(t,x,'r');
%     title(sprintf('Subplot %d: Fake Std',n+k));
% end

subplot(2,3,1);
x = XGeneratedNew(:,1);
plot(t,x,'r');
title('Subplot 1: Fake Stim');
subplot(2,3,2);
x = XGeneratedNew(:,2);
plot(t,x,'r');
title('Subplot 2: Fake Stim');
subplot(2,3,3);
x = XGeneratedNew(:,3);
plot(t,x,'r');
title('Subplot 3: Fake Stim')
subplot(2,3,4);
x = XGeneratedNew(:,splitInd + 1);
plot(t,x,'r');
title('Subplot 4: Fake Std');
subplot(2,3,5);
x = XGeneratedNew(:,splitInd + 2);
plot(t,x,'r');
title('Subplot 5: Fake Std');
subplot(2,3,6);
x = XGeneratedNew(:,splitInd + 3);
plot(t,x,'r');
title('Subplot 6: Fake Std')

% Plot samples of the original epochs
SignalsMat = cell2mat(Signals);
LabelsMat = zeros(1,size(Labels,1));
for i = 1 : size(Labels, 1)
    if Labels(i) == 'N'
        LabelsMat(i) = 0;
    else
        LabelsMat(i) = 1;
    end
end

stimInd = 0;
stdInd = 0;
XStimInd = zeros(1,3);
XStdInd = zeros(1,3);

genInd = 1;
while stimInd < 4 && genInd < length(LabelsMat)
    if LabelsMat(genInd) == 1
        stimInd = stimInd + 1;
        XStimInd(stimInd) = genInd;
    end
    genInd = genInd + 1;
end

genInd = 1;
while stdInd < 4 && genInd < length(LabelsMat)
    if LabelsMat(genInd) == 0
        stdInd = stdInd + 1;
        XStdInd(stdInd) = genInd;
    end
    genInd = genInd + 1;
end

figure;
% for k = 1 : n
%     subplot (2,2*n,k);
%     x = SignalsMat(XStimInd(k),:);
%     plot(t,x,'r');
%     title(sprintf('Subplot %d: Orig Stim',k));
% end
% 
% for k = 1 : n
%     subplot (2,2*n,n+k);
%     x = SignalsMat(XStdInd(k),:);
%     plot(t,x,'r');
%     title(sprintf('Subplot %d: Orig Std',n+k));
% end

subplot(2,3,1);
x = SignalsMat(XStimInd(1),:);
plot(t,x,'b');
title('Subplot 1: Orig Stim');
subplot(2,3,2);
x = SignalsMat(XStimInd(2),:);
plot(t,x,'b');
title('Subplot 2: Orig Stim');
subplot(2,3,3);
x = SignalsMat(XStimInd(3),:);
plot(t,x,'b');
title('Subplot 3: Orig Stim')
subplot(2,3,4);
x = SignalsMat(XStdInd(1),:);
plot(t,x,'b');
title('Subplot 4: Orig Std');
subplot(2,3,5);
x = SignalsMat(XStdInd(2),:);
plot(t,x,'b');
title('Subplot 5: Orig Std');
subplot(2,3,6);
x = SignalsMat(XStdInd(3),:);
plot(t,x,'b');
title('Subplot 6: Orig Std')


% %% Amplify original data
% f = 9; % f = (Desired Amplification Factor) - 1
% SignalsOrigAmp = Signals;
% LabelsOrigAmp = Labels;
% for i = 1:f
%     SignalsOrigAmp = [SignalsOrigAmp; Signals];
%     LabelsOrigAmp = [LabelsOrigAmp; Labels];
% end

%% Concatenate original and DCGAN fake data
SignalsAugm = [Signals; SignalsNew];
LabelsAugm = [Labels; LabelsNew];

% SignalsAugm = [SignalsOrigAmp; SignalsNew];
% LabelsAugm = [LabelsOrigAmp; LabelsNew];

Signals = SignalsAugm;
Labels = LabelsAugm;

% %% Concatenate IMF-augmented data and DGCAN fake data
% % save the original data used for DCGAN before being overwritten by IMFs
% load 'C:\Users\Mihay\OneDrive - Technical University of Cluj-Napoca\Teza doctorat mama\Data\date echilibrate plus imf 24febr2021\S11\fc6S11_echi_imf.mat';
% % now Signals and Labels contain 50% orig data and 50% IMF faked data
% SignalsAugm = [Signals; SignalsNew];
% LabelsAugm = [Labels; LabelsNew];
% 
% Signals = SignalsAugm;
% Labels = LabelsAugm;

% Save data
save 'C:\Users\Mihay\OneDrive - Technical University of Cluj-Napoca\Teza doctorat mama\Data\Subiecti - date echilibrate\S9\fc5_va_S9_Orig50p_DCGAN50p.mat' Labels Signals;

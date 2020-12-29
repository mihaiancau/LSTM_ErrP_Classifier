%% Transform signals and labels
% Select signals and labels from one subject and trial
load 'C:\Users\mihai\OneDrive - Technical University of Cluj-Napoca\Teza doctorat mama\Data_ErrP\MatLab_EpochedData\VizualActiv_DoarEpoci\fc5_va_D1800.mat';
% Convert categorical labels to 1 x 1800 vector, where 0 = std. and 1 = stim.
Labels = Labels.';
labels = zeros(1,1800); 
for i = 1 : length(Labels)
    if (Labels(i) == 'N') 
        labels(i) = 0;
    else
        labels(i) = 1;
    end
end
% Convert cellular signals to array of double
signals = cell2mat(Signals).'; % 231 x 1800 array
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
params.numEpochs = 1000;
params.miniBatchSize = 256;

% Specify the options for Adam optimizer
params.learnRate = 0.0002;
params.gradientDecayFactor = 0.5;
params.squaredGradientDecayFactor = 0.999;

executionEnvironment = "cpu";
params.executionEnvironment = executionEnvironment;

% Train the CGAN
[dlnetGenerator,dlnetDiscriminator] = trainGAN(dlnetGenerator, ...
        dlnetDiscriminator,signalsNormalized,labels,params); 
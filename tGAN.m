%% Transform signals and labels
% Select signals and labels from one subject and trial
% load EpochedData_Training\fc5_va_D1800.mat;

%% Generator Network
% Output: 231 x 1 x 1 array
numFilters = 64;
numLatentInputs = 100;
% Network architecture equation 231 = (N1+1) + 15*f1 - 4 (provided stride = 1 for all layers)
% This derives from the more general activations = input_size - (filter_size - 1),
% provided no padding and striding applied
projectionSize = [1 1 159]; % = N1 projection size is [1 1 N1]
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
% Input: 231 x 1 x 1 array
scale = 0.2; % for Leaky ReLU if x < 0, then f(x) = scale * x, otherwise f(x) = x
inputSize = [231 1 1];

% This derives from the more general 
% activations = ((input_size + 2*padding - filter_size)/stride) + 1

layersDiscriminator = [
    imageInputLayer(inputSize,'Normalization','none','Name','in')
    concatenationLayer(3,2,'Name','cat')
    convolution2dLayer([32 1],8*numFilters,'Stride',2,'Padding',[1 0],'Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    convolution2dLayer([16 1],4*numFilters,'Stride',2,'Padding',[1 0],'Name','conv2')
    leakyReluLayer(scale,'Name','lrelu2')
    convolution2dLayer([8 1],2*numFilters,'Stride',2,'Padding',[1 0],'Name','conv3')
    leakyReluLayer(scale,'Name','lrelu3')
    convolution2dLayer([4 1],numFilters,'Stride',2,'Padding',[1 0],'Name','conv4')
    leakyReluLayer(scale,'Name','lrelu4')
    convolution2dLayer([8 1],1,'Name','conv5')];

lgraphDiscriminator = layerGraph(layersDiscriminator);

layers = [
    imageInputLayer([1 1],'Name','labels','Normalization','none')
    embedAndReshapeLayer(inputSize,embeddingDimension,numClasses,'emb')];

lgraphDiscriminator = addLayers(lgraphDiscriminator,layers);
lgraphDiscriminator = connectLayers(lgraphDiscriminator,'emb','cat/in2');

dlnetDiscriminator = dlnetwork(lgraphDiscriminator);

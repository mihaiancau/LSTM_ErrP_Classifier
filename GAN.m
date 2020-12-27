%% Transform signals and labels
% Select signals and labels from one subject and trial
% load EpochedData_Training\fc5_va_D1800.mat;

%% Generator Network
% Output: 1 x 231 x 1 array
numFilters = 64;
numLatentInputs = 100;
% Network architecture equation 231 = N1 + 15*f1 - 4 (provided stride = 1 for all layers)
projectionSize = [1 160 1]; % = N1 projection size is [1 N1 1]
filterSize = 5; % = f1 filter size is [1 f1]
numClasses = 2;
embeddingDimension = 100; % matches numLatentInputs

% Network layer array
layersGenerator = [
    imageInputLayer([1 numLatentInputs 1],'Normalization','none','Name','in')
    projectAndReshapeLayer(projectionSize,numLatentInputs,'proj');
    concatenationLayer(2,2,'Name','cat');
    transposedConv2dLayer([1 filterSize],4*numFilters,'Name','tconv1') 
    batchNormalizationLayer('Name','bn1','Epsilon',5e-5)
    reluLayer('Name','relu1')
    transposedConv2dLayer([1 2*filterSize],2*numFilters,'Name','tconv2')
    batchNormalizationLayer('Name','bn2','Epsilon',5e-5)
    reluLayer('Name','relu2')
    transposedConv2dLayer([1 4*filterSize],numFilters,'Name','tconv3') 
    batchNormalizationLayer('Name','bn3','Epsilon',5e-5)
    reluLayer('Name','relu3')
    transposedConv2dLayer([1 8*filterSize],1,'Name','tconv4') 
    ];

lgraphGenerator = layerGraph(layersGenerator);

layers = [
    imageInputLayer([1 1 1],'Name','labels','Normalization','none')
    % An embed and reshape layer takes as input numeric indices of categorical elements and converts them to images of the specified size. 
    % Use embed and reshape layers to input categorical data into conditional GANs.
    embedAndReshapeLayer([1 1],embeddingDimension,numClasses,'emb')];

lgraphGenerator = addLayers(lgraphGenerator,layers);
lgraphGenerator = connectLayers(lgraphGenerator,'emb','cat/in2');
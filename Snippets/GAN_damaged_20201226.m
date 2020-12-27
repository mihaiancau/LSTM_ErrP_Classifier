%% Data structure 
% Signals = 1800 epochs of 231 signal samples of 2 categories
% Labels = 2 categories

%% Define generator network
% Input parameters
dataTensorSize = [1 231 1];

layersGen = [
    imageInputLayer(dataTensorSize,'Normalization','none','Name','signals')
    concatenationLayer(2,2,'Name','concat')
    transposedConv2dLayer([15 1],55,'Name','transconv1')
    batchNormalizationLayer('Name','batchnorm1','Epsilon',5e-5)
    reluLayer('Name','relu1')
    transposedConv2dLayer([1 232],1,'Name','transconv2')
    
    ];

lgraphGen = layerGraph(layersGen);
layersLab = [
    imageInputLayer([1 1 1],'Normalization','none','Name','labels')
    ]; 

lgraphGen = addLayers(lgraphGen,layersLab);
lgraphGen = connectLayers(lgraphGen,'labels','concat/in2');

plot(lgraphGen)
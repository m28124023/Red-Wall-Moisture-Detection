% load model
net = load("train_WATER.mat"); 

% your data
imdsTest = imageDatastore("your_data", 'IncludeSubfolders', true);

% classify
[YPred, scores] = classify(net, imdsTest);
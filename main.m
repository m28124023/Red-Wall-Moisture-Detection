% load model
net = load("train_WATER.mat"); 

% your data
imdsTest = imageDatastore("your_data", 'IncludeSubfolders', true);

% classify
[YPred, scores] = classify(net, imdsTest);


% 定義參數
K = 5;
allData = imageDatastore("C:\Users\Yun\Desktop\Topic\LI_DATA\Thermal_Image\CNN_DATA", ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

% 產生 K-fold 的 partition
cv = cvpartition(allData.Labels, 'KFold', K);

% 讀入預訓練參數
trainingSetup = load("C:\Users\Yun\Desktop\Topic\CNN_MAT\WATER\train_WATER.mat");

% 記錄每折結果

accuracyAll = zeros(K,1);
precisionAll = zeros(K,1);
recallAll = zeros(K,1);
f1All = zeros(K,1);

for fold = 1:K
    fprintf('正在執行第 %d 折交叉驗證...\n', fold);

    % 訓練與驗證資料分割
    imdsTrain = subset(allData, training(cv, fold));
    imdsValidation = subset(allData, test(cv, fold));

    % 資料增強
     imageAugmenter = imageDataAugmenter( ...
        "RandRotation",[-90 90],...
        "RandXReflection", true,...
     "RandYReflection", true,...
        "RandScale",[0.8 1.2], ...
        "RandXReflection",true);

    % 調整大小與增強
    augimdsTrain = augmentedImageDatastore([227 227 3], imdsTrain, ...
        "DataAugmentation", imageAugmenter);
    augimdsValidation = augmentedImageDatastore([227 227 3], imdsValidation);

    % 訓練選項
    opts = trainingOptions("sgdm", ...
        "ExecutionEnvironment", "gpu", ...
        "MiniBatchSize", 70, ...
        "LearnRateDropPeriod", 10, ...
        "LearnRateDropFactor", 0.001, ...
        "InitialLearnRate", 0.0001, ...
        "MaxEpochs", 100, ...
        "Shuffle", "every-epoch", ...
        "ValidationFrequency", 25, ...
        "Plots", "none", ...  % 避免每折都開圖
        "ValidationData", augimdsValidation);

    % 定義網路層（每折需重新定義以避免參數遺留）
    layers = [
        imageInputLayer([227 227 3],"Name","data","Mean",trainingSetup.data.Mean)
        convolution2dLayer([11 11],96,"Name","conv1","BiasLearnRateFactor",2,"Stride",[4 4],"Bias",trainingSetup.conv1.Bias,"Weights",trainingSetup.conv1.Weights)
        reluLayer("Name","relu1")
        crossChannelNormalizationLayer(5,"Name","norm1","K",1)
        maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
        groupedConvolution2dLayer([5 5],128,2,"Name","conv2","BiasLearnRateFactor",2,"Padding",[2 2 2 2],"Bias",trainingSetup.conv2.Bias,"Weights",trainingSetup.conv2.Weights)
        reluLayer("Name","relu2")
        crossChannelNormalizationLayer(5,"Name","norm2","K",1)
        maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])
        convolution2dLayer([3 3],384,"Name","conv3","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.conv3.Bias,"Weights",trainingSetup.conv3.Weights)
        reluLayer("Name","relu3")
        groupedConvolution2dLayer([3 3],192,2,"Name","conv4","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.conv4.Bias,"Weights",trainingSetup.conv4.Weights)
        reluLayer("Name","relu4")
        groupedConvolution2dLayer([3 3],128,2,"Name","conv5","BiasLearnRateFactor",2,"Padding",[1 1 1 1],"Bias",trainingSetup.conv5.Bias,"Weights",trainingSetup.conv5.Weights)
        reluLayer("Name","relu5")
        maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
        fullyConnectedLayer(4096,"Name","fc6","BiasLearnRateFactor",2,"Bias",trainingSetup.fc6.Bias,"Weights",trainingSetup.fc6.Weights)
        reluLayer("Name","relu6")
        dropoutLayer(0.5,"Name","drop6")
        fullyConnectedLayer(4096,"Name","fc7","BiasLearnRateFactor",2,"Bias",trainingSetup.fc7.Bias,"Weights",trainingSetup.fc7.Weights)
        reluLayer("Name","relu7")
        dropoutLayer(0.5,"Name","drop7")
        fullyConnectedLayer(2,"Name","fc","BiasLearnRateFactor",10,"WeightLearnRateFactor",10)
        softmaxLayer("Name","softmax")
        classificationLayer("Name","classoutput")
    ];

    % 模型訓練
    [net, traininfo] = trainNetwork(augimdsTrain, layers, opts);

    % 預測與準確率
    YPred = classify(net, augimdsValidation);
    YTrue = imdsValidation.Labels;
    accuracy = mean(YPred == YTrue);
    accuracyAll(fold) = accuracy;

   figure;
    plotconfusion(YTrue, YPred);
    title(sprintf("Fold %d - Confusion Matrix", fold));

    % 計算 precision / recall / F1-score
    confMat = confusionmat(YTrue, YPred);
    TP = diag(confMat);
    FP = sum(confMat,1)' - TP;
    FN = sum(confMat,2) - TP;

    precision = mean(TP ./ (TP + FP + eps));
    recall = mean(TP ./ (TP + FN + eps));
    f1 = 2 * (precision * recall) / (precision + recall + eps);

    precisionAll(fold) = precision;
    recallAll(fold) = recall;
    f1All(fold) = f1;

    % 儲存模型
    % save(fullfile(modelSavePath, sprintf("net_fold_%d.mat", fold)), 'net', 'traininfo');

    fprintf("準確率: %.2f%%, Precision: %.2f, Recall: %.2f, F1-score: %.2f\n", ...
        accuracy*100, precision, recall, f1);
end

% 顯示平均結果
fprintf("\n===== K-Fold Summary =====\n");
fprintf("平均準確率: %.2f%% ± %.2f%%\n", mean(accuracyAll)*100, std(accuracyAll)*100);
fprintf("平均Precision: %.3f\n", mean(precisionAll));
fprintf("平均Recall: %.3f\n", mean(recallAll));
fprintf("平均F1-score: %.3f\n", mean(f1All));

save('redwall_model.mat', 'net');


imdsTest = imageDatastore("C:\Users\Yun\Desktop\Topic\DATA\image\IR_5648", 'IncludeSubfolders', true);

% 建立增強資料儲存結構
augTest = augmentedImageDatastore([227 227 3], imdsTest);

% 批次分類
[YPred, scores] = classify(net, augTest);
fileNames = imdsTest.Files;

% 顯示前幾張預測結果
for i = 1:12
    I = readimage(imdsTest, i);
    % figure;
    % imshow(I);
    % title(sprintf("預測: %s", string(YPred(i))));
end


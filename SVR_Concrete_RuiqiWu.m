%% Import the data
data = readmatrix('Concrete_Data.csv');
x = normalize(data,1);
y = readmatrix('Concrete_Target.csv');

%% Split cross-calidation data 
NumTrain = round(0.7*size(x,1));
dimension = size(x,2);
NumTest = size(x,1) - (NumTrain);
px = randperm(size(x,1));

%% Store training and testing data and result
% Training
xTrain = zeros(NumTrain,dimension);
label4training = zeros(NumTrain,1);
% Testing
xTest = zeros(NumTest,dimension);
label4testing = zeros(NumTest,1);

%% Creat training and testing data
index = 0; % Initialize the index
   for k = 1:NumTrain
         index = index + 1;
         xTrain(index,:) = x(px(k),1:end);
         label4training(index,1) = y(px(k),1);
   end

   index = 0; % Initialize the index
    for k = NumTrain + 1:NumTrain + NumTest 
        index = index + 1;
                xTest(index,:) = x(px(k),1:end);
                label4testing(index,1) = y(px(k),1);
    end

 %% Train the model
 mdl = fitrsvm(xTrain, label4training,'KernelFunction','gaussian');

 %% Performance of the SVR network
 yTest = predict(mdl, xTest);
 yTrain = predict(mdl, xTrain);
 RMSE = sqrt(mean((yTest-label4testing).^2));

  % Calculate R2
 r = label4testing-yTest;
 normr = norm(r);
 SSE = normr.^2;
 SST = norm(label4testing-mean(label4testing))^2;
 R2 = 1 - SSE/SST;

%% Visualize the predictions from the ANN model
plot(label4testing,yTest,'x','Color','b'); hold on;
plot(label4training,yTrain,'o','Color','m'); hold on;
plot(0:100,0:100,'Color','k'); hold off;

xlabel('Training data') 
ylabel('Predicted strength') 
legend({'Testing','Training'},'Location','northwest')



%% %% Test RMSE vs training data ratio 
for i = 1:99
    TrainingDataRatio = i;
NumTrain = round(TrainingDataRatio*0.01*size(x,1));
dimension = size(x,2);
NumTest = size(x,1) - (NumTrain);
px = randperm(size(x,1));

xTrain = zeros(NumTrain,dimension);
label4training = zeros(NumTrain,1);
xTest = zeros(NumTest,dimension);
label4testing = zeros(NumTest,1);

index = 0; 
   for k = 1:NumTrain
         index = index + 1;
         xTrain(index,:) = x(px(k),1:end);
         label4training(index,1) = y(px(k),1);
   end

   index = 0;
    for k = NumTrain + 1:NumTrain + NumTest 
        index = index + 1;
                xTest(index,:) = x(px(k),1:end);
                label4testing(index,1) = y(px(k),1);
    end

 mdl = fitrsvm(xTrain, label4training,'KernelFunction','gaussian');

 yTest = predict(mdl, xTest);
 yTrain = predict(mdl, xTrain);
 RMSE(i) = sqrt(mean((yTest-label4testing).^2))
end 

plot(1:99,RMSE,'Color','b'); hold on;

xlabel('Training data ratio') 
ylabel('RMSE') 
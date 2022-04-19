%% Import the data
data = readmatrix('Concrete1.csv');
x = data(:,1:8);
y = data(:,9);
m = length(y);

%% Visualization of the data
histogram(x(:,3),10);
plot(x(:,3),y,'o')

%% Normalize the freatures and transform the output
y2 = log(1+y);
for i = 1:8
    x2(:,i) = (x(:,i)-min(x(:,i)))/(max(x(:,i))-min(x(:,i)));
end
    histogram(x2(:,2),10);
 plot(x2(:,1),y2,'o');

 %% Train an artificial neural network(ANN)
 xt = x2';
 yt = y2';

 %  Define Parameters 
 hiddenLayerSize = 15;
 net = fitnet(hiddenLayerSize);
 net.divideParam.trainRatio = 70/100;
 net.divideParam.valRatio = 30/100;
 net.divideParam.testRatio = 0/100;
 
 %  Train the model
 [net,tr] = train(net,xt,yt);

 %% Performance of the ANN network
 yTrain = exp(net(xt(:,tr.trainInd)))-1;
 yTrainTrue = exp(yt(tr.trainInd))-1;
 sqrt(mean((yTrain - yTrainTrue).^2))

 yVal = exp(net(xt(:,tr.valInd)))-1;
 yValTrue = exp(yt(tr.valInd))-1;
 sqrt(mean((yVal - yValTrue).^2))
 
 % Calculate R2
 r = yTrainTrue-yTrain;
 normr = norm(r);
 SSE = normr.^2;
 SST = norm(yTrainTrue-mean(yTrainTrue))^2;
 R2 = 1 - SSE/SST;

%% Visualize the predictions from the ANN model
plot(yTrainTrue,yTrain,'o','Color','m'); hold on;
plot(yValTrue,yVal,'x','Color','b');hold on;
plot(0:100,0:100,'Color','k'); hold off;

xlabel('Testing Data') 
ylabel('Predicted strength') 
legend({'Training','Testing'},'Location','northwest')


 %% Optimize the number of neurons in the hidden layer

for i = 1:60
    %Defining the architecture of the ANN
    hiddenLayerSize = i;
    net = fitnet(hiddenLayerSize);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 30/100;
    net.divideParam.testRatio = 0/100;
   
    % Training the ANN
    [net,tr] = train(net,xt,yt);

    % Determing the error
    yTrain = exp(net(xt(:,tr.trainInd)))-1;
    yTrainTrue = exp(yt(tr.trainInd))-1;
    yVal = exp(net(xt(:,tr.valInd)))-1;
    yValTrue = exp(yt(tr.valInd))-1;
    rmse_train(i) = sqrt(mean((yTrain - yTrainTrue).^2)) % RMSE of training
    rmse_val(i) = sqrt(mean((yVal - yValTrue).^2)) % RMSE of validation
end

%% Select the optimal number of neuron in hidden layer
plot(1:60,rmse_train); hold on;
plot(1:60,rmse_val); hold off;
xlabel('Hidden layer number') 
ylabel('RMSE value') 
legend({'RMSE of train','RMSE of test'},'Location','northwest')
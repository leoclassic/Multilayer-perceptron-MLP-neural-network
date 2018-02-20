function [  ] = MLP

xTrain = linspace(-4,4,100)';
tTrain = mapminmax(sin(xTrain)',0,1)';
tTrain = tTrain + rand(size(tTrain,1),size(tTrain,2))./10; % noise
xTest = linspace(-4,4,100)';
tTest = mapminmax(sin(xTest)',0,1)';

h = [10,10]; % 2 hidden layers network.
maxIter = 200;
lr = 0.05;

model = trainFcn(xTrain,tTrain,h,maxIter,lr);

Y = predict(model,xTest);

figure
plot(xTest,tTest,'b',xTest,Y,'r')
end

function model = trainFcn(X,T,layer,maxIter,eta)
model = initFcn(X,T,layer);

error = inf(1,maxIter);
for iter = 1:maxIter
    [~,output] = predict(model,X);
    
    E = T-output{end};
    error(iter) = mean(E(:)'*E(:));
    
    for l = model.nLayer-1:-1:1
        df = output{l+1}.*(1-output{l+1});
        dG = df.*E;
        dW = output{l}*dG';
        model.W{l} = model.W{l} + eta*dW;
        E = model.W{l}*dG;
    end
    
    Y = predict(model,X);
    subplot(2,1,1)
    plot(X,T,'.b',X,Y,'r')
    subplot(2,1,2)
    plot(error)
    drawnow
end

end

function model = initFcn(X,T,layer)
model.Layer = [size(X,1); layer(:); size(T,1)];
model.nLayer = numel(model.Layer);
model.W = cell(model.nLayer-1);
for i = 1:model.nLayer-1
    model.W{i} = rands(model.Layer(i),model.Layer(i+1))./10;
end
end

function [Y,output] = predict(model,X)
model.nLayer = length(model.W)+1;
output = cell(1,model.nLayer);
output{1} = X;
for l = 2:model.nLayer
    output{l} = logsig(model.W{l-1}'*output{l-1});
end
Y = output{end};
end
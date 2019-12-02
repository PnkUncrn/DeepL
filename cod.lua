

--[[ outline:
1-train 1 NN on part of MNIST data 
2-test and Guage accuracy
3-Calculate delta parmas
-Store top k data in PSServer
3-Train 2nd neural network on data guage accucracy
4-Add the parameters from PSsererve (from NN1) to second, guage accuracy

-----
1. Reference user trains on local dataset and tests accuracy 
2. Zeroes its model
3. Reference user downloads all parameters from server
4. Trains on its local dataset
Tests accuracy
--]]
--NN neural Network

require 'nn'
require 'torch'
require 'dataset-mnist'
require 'image'
require 'optim'

--Basic Parameters
--N= number of participants
batchSize= 10
learningRate=0.01
weightDecay=1e-7
N=4
theta_u_perc=0.1

-- fix seed
torch.manualSeed(1)

--model design
classes = {'1','2','3','4','5','6','7','8','9','10'}
geometry = {32,32}

-- log results to files
trainLogger = optim.Logger(paths.concat("logs", 'train.log'))
testLogger = optim.Logger(paths.concat("logs", 'test.log'))

--OG NN
inputLayer = 1024
hiddenLayer1 = 128
hiddenLayer2 = 64
outputSize = 10

mlp = nn.Sequential()
mlp:add(nn.Reshape(1024))
mlp:add(nn.Linear(inputLayer, hiddenLayer1))
mlp:add(nn.ReLU())
mlp:add(nn.Linear(hiddenLayer1, hiddenLayer2))
mlp:add(nn.ReLU())
mlp:add(nn.Linear(hiddenLayer2, outputSize))
mlp:add(nn.LogSoftMax())
print(mlp)

--Parameter Server model
mlp2 = nn.Sequential()
mlp2:add(nn.Reshape(1024))
mlp2:add(nn.Linear(inputLayer, hiddenLayer1))
mlp2:add(nn.ReLU())
mlp2:add(nn.Linear(hiddenLayer1, hiddenLayer2))
mlp2:add(nn.ReLU())
mlp2:add(nn.Linear(hiddenLayer2, outputSize))
mlp2:add(nn.LogSoftMax())
print(mlp2)

--Parameter Server model
mlp3 = nn.Sequential()
mlp3:add(nn.Reshape(1024))
mlp3:add(nn.Linear(inputLayer, hiddenLayer1))
mlp3:add(nn.ReLU())
mlp3:add(nn.Linear(hiddenLayer1, hiddenLayer2))
mlp3:add(nn.ReLU())
mlp3:add(nn.Linear(hiddenLayer2, outputSize))
mlp3:add(nn.LogSoftMax())
print(mlp2)

Server= nn.Sequential()
Server:add(nn.Reshape(1024))
Server:add(nn.Linear(inputLayer, hiddenLayer1))
Server:add(nn.ReLU())
Server:add(nn.Linear(hiddenLayer1, hiddenLayer2))
Server:add(nn.ReLU())
Server:add(nn.Linear(hiddenLayer2, outputSize))
Server:add(nn.LogSoftMax())
print(Server)

RefU= nn.Sequential()
RefU:add(nn.Reshape(1024))
RefU:add(nn.Linear(inputLayer, hiddenLayer1))
RefU:add(nn.ReLU())
RefU:add(nn.Linear(hiddenLayer1, hiddenLayer2))
RefU:add(nn.ReLU())
RefU:add(nn.Linear(hiddenLayer2, outputSize))
RefU:add(nn.LogSoftMax())
print(RefU)


Sparams, SgradParams=Server:getParameters()


--loss Criterion 
criterion=nn.ClassNLLCriterion()


--Dataset- MNIST
nbTrainingPatches = 6000/N
nbTestingPatches = 10000


-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

-- Confusion Matrix
confusion = optim.ConfusionMatrix(classes)

--calculating startIndex
function startIndex(x)
 ind = 1 +((x-1)*nbTrainingPatches)
 return ind
end


--training function
function train(dataset, model)
--initialize the epoch
  epoch = epoch or 1
  params, gradParams=model:getParameters()
  --one epoch
  print("training on train set")
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')

  print("Dataset Size:"..dataset:size())  

  --from 1 to the end of available data = 600
  for t=1,dataset:size(), batchSize do
    --create mini batch
    
    local inputs = torch.Tensor(batchSize,1,geometry[1],geometry[2])
    local targets = torch.Tensor(batchSize)
    local k = 1
    for i = t,math.min(t+batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local _,target = sample[2]:clone():max(1)
      target = target:squeeze()
      inputs[k] = input
      targets[k] = target
      k = k + 1
    end

      -- Method to evaluate f(X) and df/dX
      -- this function given the weights (x) will minimize the loss
  local feval = function(x)
    collectgarbage()

    -- if parameters not specified
    if x ~= params then
      params:copy(x)
    end

    -- reset gradients
    gradParams:zero()
      -- evaluate function for complete mini batch
      -- ouputs= predicted model
    local outputs = model:forward(inputs)
    local loss = criterion:forward(outputs, targets)

    -- estimate dloss_doutput by propagating the error backward
    local dloss_doutput = criterion:backward(outputs, targets)
    --accumulate the gradients
    model:backward(inputs, dloss_doutput)
          
        --[[
        if weightDecay > 0 then
          loss = loss + weightDecay * torch.norm(parameters,2)^2/2
          gradParameters:add(parameters:clone():mul(weightDecay))
          gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
        
        end
          
         --]]
         
    -- update confusion
    for i = 1,batchSize do
      confusion:add(outputs[i], targets[i])
    end
         
    return loss,gradParams
  end
      
  sgdState = sgdState or {
            learningRate = learningRate,
            momentum = 0,
            learningRateDecay = 5e-7
         }
  optim.sgd(feval, params, sgdState)
    -- disp progress
    --Progress bar shows what percentage of full dataset are we done training on
    --xlua.progress(t, dataset:size())
  
  -- end of epoch
  end
  
  --printing confusion every epoch
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  confusion:zero()
   
  epoch = epoch + 1

--end of training the NN on dataset
end
function test(dataset, model)


   -- test over given dataset
   print('<trainer> on testing Set:')
    for t = 1,dataset:size(),batchSize do
      -- disp progress
      --xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(batchSize)
      local k = 1
      for i = t,math.min(t+batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- test samples
      preds = model:forward(inputs)

      -- confusion:
      for i = 1,batchSize do
         confusion:add(preds[i], targets[i])
      end
    end


   -- print confusion matrix
  preds:zero()
  print(confusion)
  print("Confusion total valid")
  print(confusion.totalValid * 100)
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}

   confusion:zero()
end







--Testiing on Server before adding anything
print("<Server>Testing before addition of parameters")
test(testData, Server)

print("<RefU>Testing before addition of parameters")
test(testData, RefU)
--Driver snippet to create batches, train  NN, test individual NN and update PSparams
print("Current NN 1 under training ")



-- Training set: Creation and Normalization
----start index is 
participant =1
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry, startIndex(participant))
trainData:normalizeGlobal(mean, std)

--Train NN 1 over multiple epochs
while epoch ~= 10 do
  --initializing params
  params_init, gradParams_init=mlp:getParameters()

   -- train/test
   train(trainData, mlp)
   
   params,gradParams = mlp:getParameters()
   
   --calculating delta w= w (new) -w(old)
  delta_params=params:clone()
  delta_params:add(-params_init)

  --Finding top theta_u values and zeroing everything else in delta_params

  --finding top theta_u values
  theta_u= math.floor(theta_u_perc*params:size(1))
  print("theta_u val " .. theta_u)
  y,i=torch.topk(delta_params:clone():abs(), theta_u, 1, true)

  print("Size of  i")
  print(i:size(1))
  --initializing flag tensor so only index values are 1

  flagtensor=torch.Tensor(params:size(1)):fill(0)
  i:apply(function(x) flagtensor[x]=1 end)


  delta_params:cmul(flagtensor)
  Sparams:add(delta_params)

end

--reset epoch 
epoch=1

print("\n\n NN 1: Testing ")
test(testData, mlp)
print("<Server> Testing on Server After addition of delta parameter")
test(testData, Server)

print("ParamsL Last 20")
print(params[{{140086,140105}}])


--Repeat the same driver for NN 2
print("Current NN 2 under training ")

-- Training set: Creation and Normalization
--start index is 
participant = participant + 1

trainData = mnist.loadTrainSet(nbTrainingPatches, geometry, startIndex(participant))
trainData:normalizeGlobal(mean, std)

--define params
params:zero()
gradParams:zero()

--Train NN 2  over multiple epochs
while epoch ~= 10 do
   -- train/test
   
   params_mlp2,gradPrarms_mlp2 = mlp2:getParameters()

   train(trainData, mlp2)
   
   params, gradParams= mlp2:getParameters()
   --After participant N is trained over epochs we add parameters to server parameter

  --calculating delta w= w (new) -w(old)
  delta_params_2=params:clone()
  delta_params_2:add(-params_mlp2)

  --Finding top theta_u values and zeroing everything else in delta_params

  --finding top theta_u values
  theta_u= math.floor(theta_u_perc*params:size(1))
  print("theta_u val " .. theta_u)
  y,i=torch.topk(delta_params_2:clone():abs(), theta_u, 1, true)

  print("Size of i")
  print(i:size(1))
  --initializing flag tensor so only index values are 1
  flagtensor=torch.Tensor(params:size(1)):fill(0)
  i:apply(function(x) flagtensor[x]=1 end)


  delta_params_2:cmul(flagtensor)
  Sparams:add(delta_params_2)

end

--reset epoch 
epoch=1

--test the model, get accuracy
print("\n \n Neural Network 2")
print("ParamsL Last 20")
print(params[{{140086,140105}}])
print("Testing NN2")
test(testData, mlp2)


print("Testing on Server After addition of delta parameter of NN 2")
test(testData, Server)

---Training NN3
print("Current NN 3 under training ")

-- Training set: Creation and Normalization
--start index is 
participant = participant + 1

trainData = mnist.loadTrainSet(nbTrainingPatches, geometry, startIndex(participant))
trainData:normalizeGlobal(mean, std)

--define params
params:zero()
gradParams:zero()

--Train NN 2  over multiple epochs
while epoch ~= 10 do
   -- train/test
   
   params_mlp3,gradPrarms_mlp3 = mlp3:getParameters()

   train(trainData, mlp3)
   
   params, gradParams= mlp3:getParameters()
   --After participant N is trained over epochs we add parameters to server parameter

  --calculating delta w= w (new) -w(old)
  delta_params_3=params:clone()
  delta_params_3:add(-params_mlp3)

  --Finding top theta_u values and zeroing everything else in delta_params

  --finding top theta_u values
  theta_u= math.floor(theta_u_perc*params:size(1))
  print("theta_u val " .. theta_u)
  y,i=torch.topk(delta_params_3:clone():abs(), theta_u, 1, true)

  print("Size of i")
  print(i:size(1))
  --initializing flag tensor so only index values are 1
  flagtensor=torch.Tensor(params:size(1)):fill(0)
  i:apply(function(x) flagtensor[x]=1 end)


  delta_params_3:cmul(flagtensor)
  Sparams:add(delta_params_3)

end

--reset epoch 
epoch =1 
--test the model, get accuracy
print("\n \n Neural Network 2")
print("ParamsL Last 20")
print(params[{{140086,140105}}])
print("Testing NN3")
test(testData, mlp3)


print("Testing on Server After addition of delta parameter of NN 2")
test(testData, Server)

----RefU---
--1. Reference user trains on local dataset and tests accuracy 
print("<RefU>")
participant = participant + 1

trainData = mnist.loadTrainSet(60, geometry, startIndex(participant))
trainData:normalizeGlobal(mean, std)

for epoch =1,10 
do train(trainData, RefU)
end
epoch =1
print("<RefU> Testing on RefU before download of Server params. nbtraininPatches = 600")

test(testData, RefU)
--2. Zeroes its model
  params_RefU,gradPrarms_RefU = RefU:getParameters()
  params_RefU:zero()
  gradParams:zero()

--3. Reference user downloads all parameters from server
params_RefU:copy(Sparams)
--4. Trains on its local dataset
for epoch =1,10 do
train(trainData, RefU)
end
--5. Tests accuracy
print("<RefU> Testing on RefU After download")

test(testData, RefU)



--Initializing everything to zero
epoch=1
params_init:zero()
params:zero()
gradParams:zero()
params_mlp2:zero()
gradPrarms_mlp2:zero()
params_mlp3:zero()
gradPrarms_mlp3:zero()
params_RefU:zero()
gradPrarms_RefU:zero()
Sparams:zero()
SgradParams:zero()

   
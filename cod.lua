

--[[ outline:
1-train 1 NN on part of MNIST data 
2-test and Guage accuracy
3-Calculate delta parmas
-Store top k data in PSServer
3-Train 2nd neural network on data guage accucracy
4-Add the parameters from PSsererve (from NN1) to second, guage accuracy

--]]
--NN neural Network

require 'nn'
require 'torch'
require 'dataset-mnist'
require 'image'
require 'optim'

--Basic Parameters
batchSize= 1
learningRate=0.01
weightDecay=1e-7
N=2 --N= number of participants
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
PServer = nn.Sequential()
PServer:add(nn.Reshape(1024))
PServer:add(nn.Linear(inputLayer, hiddenLayer1))
PServer:add(nn.ReLU())
PServer:add(nn.Linear(hiddenLayer1, hiddenLayer2))
PServer:add(nn.ReLU())
PServer:add(nn.Linear(hiddenLayer2, outputSize))
PServer:add(nn.LogSoftMax())
print(PServer)

PSparams_init,PSgradParameters_init = PServer:getParameters()

params_init, gradParams_init=mlp:getParameters()
--loss Criterion 
criterion=nn.ClassNLLCriterion()


--Dataset- MNIST
nbTrainingPatches = 60000/N
nbTestingPatches = 10000


-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

-- Confusion Matrix
confusion = optim.ConfusionMatrix(classes)

--training 
function train(dataset, model)
--initialize the epoch
  epoch = epoch or 1
  
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

      -- MEthod to evaluate f(X) and df/dX
      -- this function given the weight (x) will minimize the loss
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
      xlua.progress(t, dataset:size())

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
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,batchSize do
         confusion:add(preds[i], targets[i])
      end
    end


   -- print confusion matrix
  print(confusion)
  print("Confusion total valid")
  print(confusion.totalValid * 100)
  testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}

   confusion:zero()
end





--Driver snippet to create batches, train  NN, test individual NN and update PSparams
print("Current NN 1 under training ")

-- Training set: Creation and Normalization
--start index is 1
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry, 1)
trainData:normalizeGlobal(mean, std)

--initializing params
params,gradParams = mlp:getParameters()


--Train NN over multiple epochs
while epoch ~= 4 do
   -- train/test
   train(trainData, mlp)
end


--calculating delta w= w (new) -w(old)
delta_params=params:clone()
delta_params:add(-params_init)

--Finding top theta_u values and zeroing everything else in delta_params

--finding top theta_u values
theta_u= math.floor(theta_u_perc*params:size(1))
print("theta_u val " .. theta_u)
y,i=torch.topk(delta_params:clone():abs(), theta_u, 1, true)

print("y and i")
print(i:size(1))
--initializing flag tensor so only index values are 1
flagtensor=torch.Tensor(params:size(1)):fill(0)
i:apply(function(x) flagtensor[x]=1 end)


delta_params:cmul(flagtensor)



--reset epoch 
epoch=1

--test the model, get accuracy
print("Neural Network 1")
print("ParamsL Last 20")
print(params[{{140086,140105}}])
test(testData, mlp)




--Repeat the same driver for NN 2
print("Current NN 2 under training ")
-- Training set: Creation and Normalization

--start index is 
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry, 30001)
trainData:normalizeGlobal(mean, std)

--define params
params:zero()
params,gradParams = PServer:getParameters()

--Train NN over multiple epochs
while epoch ~= 4 do
   -- train/test
   train(trainData, PServer)
end

--After participant N is trained over epochs we add parameters to server parameter

--calculating delta w= w (new) -w(old)
delta_params_2=params:clone()
delta_params_2:add(-PSparams_init)

--Finding top theta_u values and zeroing everything else in delta_params

--finding top theta_u values
theta_u= math.floor(theta_u_perc*params:size(1))
print("theta_u val " .. theta_u)
y,i=torch.topk(delta_params_2:clone():abs(), theta_u, 1, true)

print("y and i")
print(i:size(1))
--initializing flag tensor so only index values are 1
flagtensor=torch.Tensor(params:size(1)):fill(0)
i:apply(function(x) flagtensor[x]=1 end)


delta_params_2:cmul(flagtensor)



--reset epoch ladela
epoch=1

--test the model, get accuracy
print("Neural Network ")
print("ParamsL Last 20")
print(params[{{140086,140105}}])
test(testData, PServer)



--[[After guaging the accuracy of NN2
we add the delta parameters of NN1 to NN2 and guage accuracy
--]]

params:add(delta_params)
test(testData, PServer)


--Initializing everything to zero
params_init:zero()
params:zero()
gradParams:zero()
PSparams_init:zero()
PSgradParameters_init:zero()
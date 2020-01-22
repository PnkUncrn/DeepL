require 'nn'
require 'torch'
require 'dataset-mnist'
require 'image'
require 'optim'

helper= {}
torch.manualSeed(50)

inputLayer = 1024
hiddenLayer1 = 128
hiddenLayer2 = 64
outputSize = 10


function CreateNN(self)

self = nn.Sequential()
self:add(nn.Reshape(1024))
self:add(nn.Linear(inputLayer, hiddenLayer1))
self:add(nn.ReLU())
self:add(nn.Linear(hiddenLayer1, hiddenLayer2))
self:add(nn.ReLU())
self:add(nn.Linear(hiddenLayer2, outputSize))
self:add(nn.LogSoftMax())

return self
end



function startIndex(x)
 local nbTrainP=nbTrainingPatches
 ind = 1 +((x-1)*nbTrainP)
 return ind
end
function helper.train(dataset, model)
--initialize the epoch
  params, gradParams=model:getParameters()
  --one epoch
  
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
      params:clone(x)
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
   
--end of training the NN on dataset
end
function helper.test(dataset, model, mode)
  -- test over given dataset
   print('<Tester> on testing Set:')
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
  
  if mode=='ServerMode' then
    ServerLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  elseif mode=='RefUserMode' then
    RefUserLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  else
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    
  end
  
  confusion:zero()

end

--iterParticipant: the exact index corresponding to the NeuralNet based on which the same train data is carved out for NN everytime.
function helper.DriverTrain(model, iterParticipant)
  
  local epochL = epoch
  
  if not TotalParticipantInteraction then
    TotalParticipantInteraction = 1
  else TotalParticipantInteraction= TotalParticipantInteraction +1
  end
  
  trainData = mnist.loadTrainSet(nbTrainingPatches, geometry, startIndex(iterParticipant))
  trainData:normalizeGlobal(mean, std)
  
--Train NN over multiple epochs
  for iter=1,epochL do
   -- train/test
  print("<Drivertrain> online epoch # " .. iter .. ' [batchSize = ' .. batchSize .. ']')
  
  params_old,gradPrarms_old = model:getParameters()

  if iter > 1 then
    params_old:copy(Sparams)
  end
  
   helper.train(trainData, model)
   
   params_new, gradParams_new= model:getParameters()
   --After participant N is trained over epochs we add parameters to server parameter

    --calculating delta w= w (new) -w(old)
    delta_params=params_new:clone()
    delta_params:add(-params_old)

    --Finding top theta_u values and zeroing everything else in delta_params

    --finding top theta_u values
  theta_u= math.floor(theta_u_perc*params_new:size(1))
  y,i=torch.topk(delta_params:clone():abs(), theta_u, 1, true)

  --initializing flag tensor so only index values are 1
  flagtensor=torch.Tensor(params_new:size(1)):fill(0)
  i:apply(function(x) flagtensor[x]=1 end)


  delta_params:cmul(flagtensor)
  Sparams:add(delta_params)

end

  print("<Model>Testing "..iterParticipant)
  helper.test(testData, model)


  print("<Server>Testing post addition of delta parameter of Total participant "..TotalParticipantInteraction)
  helper.test(testData, Server, 'ServerMode')
end

function helper.ReferenceUserDriver(model)
   if not IterNumRefUser then
    IterNumRefUser = 1
  else IterNumRefUser= IterNumRefUser +1
  end

  
  RefereceUserParams, RefereceUserGParams=model:getParameters()
  RefereceUserParams:copy(Sparams)
  

  
  print("<RefU" ..IterNumRefUser.."> Training after downloading params")
  for iterator =1,10 do
    helper.train(ReferenceUserTrainData, model)
  end
  --]]
  --Tests accuracy
  print("------------------------------------------------------")
  print("<RefUser: " ..IterNumRefUser.." Testing")
  helper.test(testData, model, 'RefUserMode')
  end
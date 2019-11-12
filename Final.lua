--[[ outline:
1-train 1 NN on part of MNIST data 
2-test and Guage accuracy
3-Select random parameters to copy onto seperate running total (w or theta)
3-Zero the parameters
4-Repeat for a different set of MNISt data untill all data 
5-Creat NN based on Parameter server
6-guage accuracy

--]]
--NN neural Network

require 'nn'
require 'torch'
require 'dataset-mnist'
require 'image'
require 'optim'

--N= number of participants
batchSize= 30
learningRate=0.01
weightDecay=1e-7
N=4
theta_u_perc=0.01

--load MNIST


--model design
classes = {'1','2','3','4','5','6','7','8','9','10'}
geometry = {32,32}

--Trial model
PServer = nn.Sequential()
PServer:add(nn.Reshape(1024))
      PServer:add(nn.Linear(1024, 2048))
      PServer:add(nn.Tanh())
      PServer:add(nn.Linear(2048,#classes))
      PServer:add(nn.LogSoftMax())
      

print (PServer)

--NN model
model = nn.Sequential()
model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024, 2048))
      model:add(nn.Tanh())
      model:add(nn.Linear(2048,#classes))
      model:add(nn.LogSoftMax())
      
      
print (model)

--OG Model
--[[
inputLayer = 1024
hiddenLayer1 = 128
hiddenLayer2 = 64
outputSize = 10

--OG NN

model = nn.Sequential()
model:add(nn.Reshape(1024))
model:add(nn.Linear(inputLayer, hiddenLayer1))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenLayer1, hiddenLayer2))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenLayer2, outputSize))
model:add(nn.LogSoftMax())
print(model)

--]]



PSparams_init,PSgradParameters_init = PServer:getParameters()
PSparams_update=PSparams_init:clone()

params, gradParams=model:getParameters()
--loss Criterion 
criterion=nn.ClassNLLCriterion()


--Dataset- MNIST
nbTrainingPatches = 600
nbTestingPatches = 10000


--training 
function train(dataset)
--initialize the epoch
  epoch = epoch or 1
  
  --one epoch
  print("training on train set")
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')

  print("Dataset Size:"..dataset:size())  

  --from 1 to th end of available data = 600
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
      xlua.progress(t, dataset:size())
      
  -- end of epoch
  end
   
  epoch = epoch + 1

--end of training the NN on dataset
end


--iterate over every NN

for participant=1,N do

print("Current NN under training ".. participant)

-- create training set and normalize
startIndex = 1 +((participant-1)*600)
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry, startIndex)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

--equating the current NN's params to PSserver params
params:copy(PSparams_init)

--Train NN over multiple epochs
while epoch ~= 3 do
   -- train/test
   train(trainData)
end

--After participant N is trained over epochs we add parameters to server parameter

--calculating delta w= w (new) -w(old)
delta_params=params:clone()
delta_params:add(-PSparams_init)


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

--copy delta_params to params
params:copy(delta_params)


--Adding the delta params from model to PSServer
PSparams_update:add(params)
print(torch.eq(PSparams_update, PSparams_init):min(1))

--reset epoch
epoch=1

--test the model, get accuracy

end



--Test the server and get accuracy

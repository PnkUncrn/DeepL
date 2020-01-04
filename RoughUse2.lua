--[[
trying to mimic cod.lua

Gameplan : 1-
--]]
require 'nn'
require 'torch'
require 'dataset-mnist'
require 'image'
require 'optim'
require 'Helper'

--Basic Parameters
--N= number of participants
batchSize= 10
learningRate=0.01
weightDecay=1e-7
N=25
theta_u_perc=0.1
torch.manualSeed(45)
epoch=20

-- log results to files
trainLogger = optim.Logger(paths.concat("logs", 'train.log'))
testLogger = optim.Logger(paths.concat("logs", 'test.log'))
RefUserLogger=optim.Logger(paths.concat("logs",'RefUserLogger.log'))
ServerLogger=optim.Logger(paths.concat("logs",'ServerLogger.log'))

--model design
classes = {'1','2','3','4','5','6','7','8','9','10'}
geometry = {32,32}

-- log results to files
trainLogger = optim.Logger(paths.concat("logs", 'train.log'))
testLogger = optim.Logger(paths.concat("logs", 'test.log'))


--loss Criterion 
criterion=nn.ClassNLLCriterion()

--Dataset- MNIST
nbTrainingPatches = 60000/N
nbTestingPatches = 10000

-- Confusion Matrix
confusion = optim.ConfusionMatrix(classes)


-- create test sets and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

ReferenceUserTrainData = mnist.loadTrainSet(60, geometry, 4550)
ReferenceUserTrainData:normalizeGlobal(mean, std)

--Creating NN table 
--creating multiple NN using for loop
Server=CreateNN()
Sparams, SgradParams=Server:getParameters()

NNTable={}
RefereceUserTable={}
for i = 1,6  do
  RefereceUserTable[i]=CreateNN()
  --[[paramsL, gradParamsL=RefereceUserTable[i]:getParameters()
  paramsL:zero()
  gradParamsL:zero()
  --]]
end

for i = 1,N  do
  NNTable[i]=CreateNN()
end

helper.DriverTrain(NNTable[1])
helper.ReferenceUserDriver(RefereceUserTable[1])

helper.DriverTrain(NNTable[2])
helper.DriverTrain(NNTable[3])
helper.ReferenceUserDriver(RefereceUserTable[2])

helper.DriverTrain(NNTable[4])
helper.DriverTrain(NNTable[5])
helper.ReferenceUserDriver(RefereceUserTable[3])

helper.DriverTrain(NNTable[6])
helper.DriverTrain(NNTable[1])

helper.DriverTrain(NNTable[7])
helper.DriverTrain(NNTable[2])

helper.DriverTrain(NNTable[8])
helper.DriverTrain(NNTable[3])

helper.DriverTrain(NNTable[9])
helper.DriverTrain(NNTable[10])
helper.ReferenceUserDriver(RefereceUserTable[4])

helper.DriverTrain(NNTable[11])
helper.DriverTrain(NNTable[12])
helper.DriverTrain(NNTable[4])

helper.DriverTrain(NNTable[13])
helper.DriverTrain(NNTable[14])
helper.DriverTrain(NNTable[1])
helper.DriverTrain(NNTable[5])

helper.DriverTrain(NNTable[15])
helper.ReferenceUserDriver(RefereceUserTable[5])

--plotting to file
gnuplot.setterm('png')
gnuplot.pngfigure("test.png")
RefUserLogger:style{'+-'}
RefUserLogger:plot()

gnuplot.plotflush()
gnuplot.closeall()

--Initializing everything to zero
epoch=1

Sparams:zero()
SgradParams:zero()

   

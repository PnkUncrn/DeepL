--[[BaselineOneUSer

In this file we try to create a baseline scenario. There will be only one user that will train on all the samples in the entire Training Dataset.

The accuracy obtained will represent the baseline as highest accuracy.

Epoch =50

-Create One user
-Train data is entire dataset
for every epoch:
  -Train
  -Test Accuracy
  
--]]

require 'nn'
require 'torch'
require 'dataset-mnist'
require 'image'
require 'optim'
require 'Helper'

--Basic Parameters
batchSize= 10
learningRate=0.01
weightDecay=1e-7
NumParticipant=1
epoch=50
torch.manualSeed(46)

classes = {'1','2','3','4','5','6','7','8','9','10'}
geometry = {32,32}

--loss Criterion 
criterion=nn.ClassNLLCriterion()

--Dataset- MNIST
nbTrainingPatches = 60
nbTestingPatches = 10000

--Adding helpers to avoid tripping errors in Helper
trainLogger = optim.Logger(paths.concat("logs", 'Base60train.log'))
testLogger = optim.Logger(paths.concat("logs", 'Base60test.log'))

-- Confusion Matrix
confusion = optim.ConfusionMatrix(classes)


-- create test sets and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)


trainData = mnist.loadTrainSet(nbTrainingPatches, geometry, 1)
trainData:normalizeGlobal(mean, std)

LoneUser=CreateNN()

for iter=1, epoch do
print("\n Epoch: " .. iter)
helper.train(trainData, LoneUser)
helper.test(testData, LoneUser)
end

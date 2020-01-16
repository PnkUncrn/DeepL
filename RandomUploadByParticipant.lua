--[[RoughUSe 3
1- has multiple rounds
2- each round iterates over all the participants and randomly chooses which of the participants will interact with the server.
3-At the end of the round, we will print the result of the confusion matrix of the Reference User and how many participants interacted with ther Server.

--]]

--[[

Gameplan : 1-
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
NumParticipant=20
NumRefUser =5
theta_u_perc=0.1
epoch=20

torch.manualSeed(46)

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
nbTrainingPatches = 600
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

NeuralNet={}
ReferenceUser={}
for i = 1,NumRefUser  do
  ReferenceUser[i]=CreateNN()
end

for i = 1,NumParticipant  do
  NeuralNet[i]=CreateNN()
end

for round=1, NumRefUser do
  ParticipantInteraction=0
  for k=1, NumParticipant do
    randNum=torch.uniform()
    if randNum>0.5 then
    ParticipantInteraction=ParticipantInteraction+1
    print("=====Participant: "..k .." Selected. RandNum: "..randNum)
    helper.DriverTrain(NeuralNet[k])
  else print("=====Participant: "..k .." Rejected. RandNum: "..randNum)
  end
  
  end
  print("=======Completed Round: " ..round)
  print("Participant Interaction: "..ParticipantInteraction)
  helper.ReferenceUserDriver(ReferenceUser[round])
end
  

--plotting to file

--Initializing everything to zero
epoch=1

Sparams:zero()
SgradParams:zero()

   

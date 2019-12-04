require 'nn'
require 'torch'

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
print(mlp3)

RefU= nn.Sequential()
RefU:add(nn.Reshape(1024))
RefU:add(nn.Linear(inputLayer, hiddenLayer1))
RefU:add(nn.ReLU())
RefU:add(nn.Linear(hiddenLayer1, hiddenLayer2))
RefU:add(nn.ReLU())
RefU:add(nn.Linear(hiddenLayer2, outputSize))
RefU:add(nn.LogSoftMax())
print(RefU)


RefU2= nn.Sequential()
RefU2:add(nn.Reshape(1024))
RefU2:add(nn.Linear(inputLayer, hiddenLayer1))
RefU2:add(nn.ReLU())
RefU2:add(nn.Linear(hiddenLayer1, hiddenLayer2))
RefU2:add(nn.ReLU())
RefU2:add(nn.Linear(hiddenLayer2, outputSize))
RefU2:add(nn.LogSoftMax())
print(RefU2)


Server= nn.Sequential()
Server:add(nn.Reshape(1024))
Server:add(nn.Linear(inputLayer, hiddenLayer1))
Server:add(nn.ReLU())
Server:add(nn.Linear(hiddenLayer1, hiddenLayer2))
Server:add(nn.ReLU())
Server:add(nn.Linear(hiddenLayer2, outputSize))
Server:add(nn.LogSoftMax())
print(Server)





require 'torch'
require 'paths'
require 'image'

mnist = {}

mnist.path_remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
mnist.path_dataset = 'mnist.t7'
mnist.path_trainset = paths.concat(mnist.path_dataset, 'train_32x32.t7')
mnist.path_testset = paths.concat(mnist.path_dataset, 'test_32x32.t7')

function mnist.download()
   if not paths.filep(mnist.path_trainset) or not paths.filep(mnist.path_testset) then
      local remote = mnist.path_remote
      local tar = paths.basename(remote)
      os.execute('wget ' .. remote .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
   end
end

function mnist.loadTrainSet(maxLoad, geometry, startIndex)
 
   return mnist.loadDataset(mnist.path_trainset, maxLoad, geometry, startIndex)
end

function mnist.loadTestSet(maxLoad, geometry)
   return mnist.loadDataset(mnist.path_testset, maxLoad, geometry)
end

function mnist.loadDataset(fileName, maxLoad, geometry, startIndex)
   mnist.download()
   local f = torch.load(fileName, 'ascii')
   local data = f.data:type(torch.getdefaulttensortype())
   local labels = f.labels

   --NEaxmple = no. of rows in data = 60,000
   local nExample = f.data:size(1)
   
   --if Maxload is specified and is less than total data available, then 
   --nExample = maxload
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
   end
  
   if not startIndex then 
     startIndex =1
    end
  
   endIndex=startIndex+nExample-1

   print("StartIndex"..startIndex)
   print("endIndex"..endIndex)
   --carving out the datasets
   data = data[{{startIndex,endIndex},{},{},{}}]
   labels = labels[{{startIndex, endIndex}}]
   print("Dimension of dataset")
   print(labels:size())
   print('<mnist> done')
   
   local dataset = {}
   dataset.data = data
   dataset.labels = labels
  
   
   function dataset:normalize(mean_, std_)
      local mean = mean_ or data:view(data:size(1), -1):mean(1)
      local std = std_ or data:view(data:size(1), -1):std(1, true)
      for i=1,data:size(1) do
         data[i]:add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(10)

   setmetatable(dataset, {__index = function(self, index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local label = labelvector:zero()
			     label[class] = 1
			     local example = {input, label}
           return example
    end})

   return dataset
end


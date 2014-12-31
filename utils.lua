--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 05/12/14
-- Time: 13:32
-- To change this template use File | Settings | File Templates.
--
local utils = {}


function utils.get_accuracy(outputs, targets)
   local correct, total = 0, 0
   for i = 1, targets:size(1) do
      total = total + 1
      _, index = outputs[i]:max(1)
      if index[1] == targets[i] then
         correct = correct + 1
      end
   end
   return (correct / total)*100
end

function utils.create_minibatch(dataset, index, batchSize, n, geometry)
   local inputs = torch.Tensor(batchSize,1,geometry[1],geometry[2])
   local targets = torch.Tensor(batchSize)
   local k = 1
   for i = index, math.min(index+batchSize-1, n) do
      -- load new sample
      inputs[k] = dataset.inputs:select(1,i)
      targets[k] = dataset.targets[i]+1
      k = k + 1
   end
   return inputs, targets
end

function utils.normalize(data)
   local std = data:std()
   local mean = data:mean()
   data:add(-mean)
   data:mul(1/std)
   return mean, std
end

return utils

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

function utils.create_minibatch(dataset, index, batchSize, geometry)
   local inputs = torch.Tensor(batchSize,1,geometry[1],geometry[2])
   local targets = torch.Tensor(batchSize)
   local k = 1
   for i = index, math.min(index+batchSize-1,dataset:size()) do
      -- load new sample
      local sample = dataset[i]
      local input = sample[1]:clone()
      local _,target = sample[2]:clone():max(1)
      target = target:squeeze()
      inputs[k] = input
      targets[k] = target
      k = k + 1
   end
   return inputs, targets
end

return utils

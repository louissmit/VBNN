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

function utils.num_grad(to_check, func)
    local epsilon = 2*torch.sqrt(1e-12)*(1+torch.norm(to_check))
    print(epsilon)
    to_check:add(epsilon)
    local F1 = func()
    to_check:add(-2*epsilon)
    local F2 = func()
    to_check:add(epsilon)
    local numgrad = torch.add(F1, -F2):mul(1 / (2*epsilon))
    return numgrad
end

return utils

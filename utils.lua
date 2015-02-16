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
      local _, index = outputs[i]:max(1)
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

function utils.select_data(trainData, indices)
    local new_trainData = {}
    local ds_size = trainData.inputs:size()
    new_trainData.inputs = torch.Tensor(#indices, ds_size[2], ds_size[3])
    new_trainData.targets = torch.Tensor(#indices)
    for k, v in pairs(indices) do
        new_trainData.inputs[k] = trainData.inputs[v]
        new_trainData.targets[k] = trainData.targets[v]
    end
    return new_trainData
end

function utils.num_grad(to_check, func)
    local epsilon = 2*torch.sqrt(1e-12)*(1+torch.norm(to_check))
    print("epsilon: ", epsilon)
    to_check:add(epsilon)
    local F1 = func()
    print("F1: ", torch.min(F1), torch.max(F1))
    to_check:add(-2*epsilon)
    local F2 = func()
    print("F2: ", torch.min(F2), torch.max(F2))
    to_check:add(epsilon)
    print("DISTF1F2: ", torch.dist(F1, F2))
    local numgrad = torch.add(F1, -F2):mul(1 / (2*epsilon))
    return numgrad
end

function utils.isnan(x) return x ~= x end

function utils.norm_pdf(x, mu, sigma)
    print(sigma:mean())
    local sigmasq = torch.pow(sigma,2)
    print(x:mean())
    print(mu:mean())
    print(sigmasq:mean())
    local exp = torch.exp(torch.cdiv(torch.pow((x-mu), 2), sigmasq):mul(-0.5))
    exp:cdiv(torch.sqrt(sigmasq:mul(2*math.pi)))
    return exp

--    return torch.exp(-.5 * (x-mu)*(x-mu)/(sigma*sigma)) / torch.sqrt(2.0*math.pi*sigma*sigma)
end

function utils.safe_save(object, folder, name)
    local filename = paths.concat(folder, name)
    print('Saving '..name..filename)
    if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    torch.save(filename, object)
end

return utils

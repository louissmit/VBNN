local mnist = require('mnist')
local u = require('utils')

local data = {}

function data.getMnist()
    local trainData = mnist.traindataset()
    local testData = mnist.testdataset()
    local create_minibatch = function(self, index, batchSize, n, geometry)
        local inputs = torch.Tensor(batchSize,1,geometry[1],geometry[2])
        local targets = torch.Tensor(batchSize)
        local k = 1
        for i = index, math.min(index+batchSize-1, n) do
            -- load new sample
            inputs[k] = self.inputs:select(1,i)
            targets[k] = self.targets[i]+1
            k = k + 1
        end
        return inputs, targets
    end
    trainData = {
        inputs=trainData.data:type('torch.FloatTensor'),
        targets=trainData.label,
        create_minibatch = create_minibatch}
    u.normalize(trainData.inputs)
    testData = {
        inputs=testData.data:type('torch.FloatTensor'),
        targets=testData.label,
        create_minibatch = create_minibatch}
    u.normalize(testData.inputs)
    return trainData, testData
end


function data.getBacteriaFold(i, k)
    local d = torch.load('gutbacteria_shuffled.torch')
    local nr_of_samples = d.inputs:size(1)

    local create_minibatch = function(self, index, batchSize, n)
        local inputs = torch.Tensor(batchSize,1, self.inputs:size(2))
        local targets = torch.Tensor(batchSize)
        local k = 1
        for i = index, math.min(index+batchSize-1, n) do
            -- load new sample
            inputs[k] = self.inputs:select(1,i)
            targets[k] = self.targets[i]+1
            k = k + 1
        end
        return inputs, targets
    end
    local fold_size = torch.round(nr_of_samples / k)
    local train_size = (k-1)*fold_size
    local test_size = nr_of_samples-train_size

    local traininputs = torch.Tensor(train_size, d.inputs:size(2)):fill(24)
    local testinputs = torch.Tensor(test_size, d.inputs:size(2)):fill(24)
    local traintargets = torch.Tensor(train_size):fill(24)
    local testtargets = torch.Tensor(test_size):fill(24)

    local lower = (i-1)*fold_size+1
    local upper = lower + test_size -1
    local e = 1
    local r = 1
    for i = 1, nr_of_samples do
        if i >= lower and i <= math.min(upper, nr_of_samples) then
            testinputs[e] = d.inputs[i]
            testtargets[e] = d.targets[i]
            e = e + 1
        else
            traininputs[r] = d.inputs[i]
            traintargets[r] = d.targets[i]
            r = r + 1
        end
    end
    local train = {
        inputs = traininputs,
        targets = traintargets,
        create_minibatch = create_minibatch
    }
    local test = {
        inputs = testinputs,
        targets = testtargets,
        create_minibatch = create_minibatch
    }
    return train, test
end

return data


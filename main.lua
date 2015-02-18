require 'cunn'
require 'optim'
local mnist = require('mnist')
local u = require('utils')
local main = {}

function main:buildModel(opt)
    local input_size = opt.geometry[1]*opt.geometry[2]
    local model = nn.Sequential()
    model:add(nn.Reshape(input_size))
    model:add(nn.Linear(input_size, opt.hidden[1]))
    --model:get(2).bias:zero()
    --model:get(2):reset(opt.var_init)
    model:add(nn.ReLU())
    opt.W = input_size*opt.hidden[1]+opt.hidden[1]--parameters:size(1)
    for i = 2, #opt.hidden do
        model:add(nn.Linear(opt.hidden[i-1], opt.hidden[i]))
        --    model:get(i*2).bias:zero()
        model:add(nn.ReLU())
        opt.W = opt.W + opt.hidden[i-1]*opt.hidden[i]+opt.hidden[i]
    end
    model:add(nn.Linear(opt.hidden[#opt.hidden], #opt.classes))
    model:add(nn.LogSoftMax())

    local criterion = nn.ClassNLLCriterion()
    if opt.cuda then
        model:cuda()
        criterion:cuda()
    end
    local parameters, gradParameters = model:getParameters()

    return model, parameters, gradParameters, criterion
end

function main:buildDataset(opt)
    local trainData = mnist.traindataset()
    local testData = mnist.testdataset()
    trainData = {inputs=trainData.data:type('torch.FloatTensor'), targets=trainData.label}
    u.normalize(trainData.inputs)
    testData = {inputs=testData.data:type('torch.FloatTensor'), targets=testData.label}
    u.normalize(testData.inputs)
    return trainData, testData
end

function main:train(model, parameters, gradParameters, criterion,  dataset, opt)
    local accuracy = 0
    local error = 0

    for t = 1, opt.trainSize, opt.batchSize do
        --      local batchtime = sys.clock()
        local inputs, targets = u.create_minibatch(dataset, t, opt.batchSize, opt.trainSize, opt.geometry)
        if opt.cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end
        collectgarbage()
        -- reset gradients
        gradParameters:zero()

        local outputs = model:forward(inputs)
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do)
        error = error + criterion:forward(outputs, targets)
        accuracy = accuracy + u.get_accuracy(outputs, targets)
        local x, _, update = optim.adam(function(_) return error, gradParameters:mul(1/opt.batchSize) end, parameters, opt.meanState)
        xlua.progress(t, opt.trainSize)
    end
    return accuracy/opt.B, error/opt.B
end

function main:run()
    local opt = require('config')
    torch.manualSeed(3)
    torch.setnumthreads(opt.threads)
    print('<torch> set nb of threads to ' .. torch.getnumthreads())
    torch.setdefaulttensortype('torch.FloatTensor')
    local model, parameters, gradParameters, criterion = self:buildModel(opt)
    local trainSet, testSet = self:buildDataset(opt)
    while true do
        local acc, err = self:train(model, parameters, gradParameters, criterion, trainSet, opt)
        print(acc, err)
    end

end
main:run()


return main
require 'nn'
require 'optim'
local mnist = require('mnist')
local u = require('utils')
local Convnet = require('convnet')
local MLP = require('mlp')
local main = {}
include('VBLinear.lua')

function main:buildDataset(opt)
    local trainData = mnist.traindataset()
    local testData = mnist.testdataset()
    trainData = {inputs=trainData.data:type('torch.FloatTensor'), targets=trainData.label}
    u.normalize(trainData.inputs)
    testData = {inputs=testData.data:type('torch.FloatTensor'), targets=testData.label}
    u.normalize(testData.inputs)
    return trainData, testData
end

function main:train(net, dataset, opt)
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
        net:resetGradients()
        if opt.type == 'vb' then
            local sample_err = 0
            local sample_acc = 0
            for i = 1, opt.S do
                net:sample()
                local err, acc = net:run(inputs, targets)
                sample_err = sample_err + err
                sample_acc = sample_acc + acc
            end
            accuracy = accuracy + sample_acc/opt.S
            error = error + sample_err/opt.S
            net:update(opt)
        else
            local err, acc = net:run(inputs, targets)
            accuracy = accuracy + acc
            error = error + err
            net:update(opt)
        end

        xlua.progress(t, opt.trainSize)
    end
    return accuracy/opt.B, error/opt.B
end

function main:test(net, dataset, opt)
    local error = 0
    local accuracy = 0
    local B = opt.testSize/opt.batchSize
    for t = 1,opt.testSize,opt.batchSize do
        -- disp progress
        xlua.progress(t, opt.testSize)

        local inputs, targets = u.create_minibatch(dataset, t, opt.batchSize, opt.testSize, opt.geometry)
        if opt.cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end
        local err, acc = net:test(inputs, targets)
        accuracy = accuracy + acc
        error = error + err
    end
    return accuracy/B, error/B
end

function main:run()
    local opt = require('config')
    accLogger = optim.Logger(paths.concat(opt.network_name, 'acc.log'))
    errLogger = optim.Logger(paths.concat(opt.network_name, 'error.log'))
--    torch.manualSeed(1)
    torch.setnumthreads(opt.threads)
    print('<torch> set nb of threads to ' .. torch.getnumthreads())
    torch.setdefaulttensortype('torch.FloatTensor')
--    local net = Convnet:buildModel(opt)
    local net = MLP:buildModel(opt)
    local trainSet, testSet = self:buildDataset(opt)

    while true do
        local trainAccuracy, trainError = self:train(net, trainSet, opt)
        print(trainAccuracy, trainError)
        local testAccuracy, testError = self:test(net, testSet, opt)
        print(testAccuracy, testError)
        accLogger:add{['train acc.'] = trainAccuracy, ['test acc.'] = testAccuracy}
        accLogger:style{['train acc.'] = '-', ['test acc.'] = '-'}
        accLogger:plot()
        errLogger:add{['train err.'] = trainError, ['test err.'] = testError}
        errLogger:style{['train err.'] = '-', ['test err.'] = '-'}
        errLogger:plot()
        u.safe_save(net, opt.network_name, 'model')
    end

end
main:run()


return main
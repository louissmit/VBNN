require 'cunn'
require 'optim'
local inspect = require('inspect')
local u = require('utils')
local data = require('data')
local Convnet = require('convnet')
local MLP = require('mlp')
local main = {}
include('VBLinear.lua')
torch.setdefaulttensortype('torch.FloatTensor')


function main:train(net, dataset, opt)
    local accuracy = 0
    local error = 0
    local t = 1
    local B = opt.trainSize/opt.batchSize
    self.indices = self.indices or torch.range(1, opt.trainSize, opt.batchSize)
    u.shuffle(self.indices):apply(function(batch_index)
        --      local batchtime = sys.clock()
        local inputs, targets = dataset:create_minibatch(batch_index, opt.batchSize, opt.trainSize, opt.geometry)
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
--            opt.S = torch.ceil(torch.pow(opt.S, 0.95))
        else
            local err, acc = net:run(inputs, targets)
            accuracy = accuracy + acc
            error = error + err
            net:update(opt)
        end

        xlua.progress(t, opt.trainSize)
        t = t + 1
    end)
    return accuracy/B, error/B
end

function main:test(net, dataset, opt)
    local error = 0
    local accuracy = 0
    local B = opt.testSize/opt.testBatchSize
    for t = 1,opt.testSize,opt.testBatchSize do
        -- disp progress
        xlua.progress(t, opt.testSize)

        local inputs, targets = dataset:create_minibatch(t, opt.testBatchSize, opt.testSize, opt.geometry)
        if opt.cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end
        collectgarbage()
        local err, acc = net:test(inputs, targets)
        accuracy = accuracy + acc
        error = error + err
    end
    return accuracy/B, error/B
end

function main:crossvalidate()
    local opt = require('config')
    local max_epoch = 50
    local totalacc = 0
    local totalerr = 0
    local i = 1
    local k = 10
    Log = require('logger'):init(opt.network_name)
    while i <= k do
        local net = MLP:buildModel(opt)
        local trainAccuracy, trainError
        local testAccuracy, testError
        local t = 1
        while t <= max_epoch do
            local trainSet, testSet = data.getBacteriaFold(i, k)
            testAccuracy, testError = self:test(net, testSet, opt)
            print(testAccuracy, testError)
            trainAccuracy, trainError = self:train(net, trainSet, opt)
            print(trainAccuracy, trainError)
            if opt.log then
                Log:add('devacc-fold='..i, testAccuracy)
                Log:add('trainacc-fold='..i, trainAccuracy)
                Log:add('deverr-fold='..i, testError)
                Log:add('trainerr-fold='..i, trainError)
                if opt.type == 'vb' then
                    local lc = net:calc_lc(opt)
                    Log:add('lc-fold='..i, lc)
                end
                Log:flush()
            end

            t = t + 1
        end
        u.safe_save(net, opt.network_name, 'model-fold='..i)
        totalacc = totalacc + testAccuracy
        totalerr = totalerr + testError
        i = i + 1
    end
    return totalacc/k, totalerr/k
end

function main:run()
    local opt = require('config')
    -- global logger
    Log = require('logger'):init(opt.network_name)
    torch.setnumthreads(opt.threads)
    print('<torch> set nb of threads to ' .. torch.getnumthreads())
--    local net = Convnet:buildModel(opt)
    local net = MLP:buildModel(opt)
    local trainSet, testSet = data.getMnist()
--    local trainSet, testSet = data.getBacteriaFold(2, 10)

    while true do
        local trainAccuracy, trainError = self:train(net, trainSet, opt)
        print(trainAccuracy, trainError)
        local testAccuracy, testError = self:test(net, testSet, opt)
        print(testAccuracy, testError)
        if opt.log then
            Log:add('devacc', testAccuracy)
            Log:add('trainacc', trainAccuracy)
            Log:add('deverr', testError)
            Log:add('trainerr', trainError)
            if opt.type == 'vb' then
                local lc = net:calc_lc(opt)
                Log:add('lc', lc)
                print('LC: ', lc)
            end
            Log:flush()
        end
        u.safe_save(net, opt.network_name, 'model')
--          net:save()
    end

end
main:run()
--print(main:crossvalidate())
--local net = MLP:load('vsadf2')
--local net = torch.load('vsadf2/model')
--local opt = net.opt
--opt.testSamples = 5
--opt.quicktest = false
--local opt = require('config')
--local trainSet, testSet = data.getMnist()
--local trainSet, testSet = data.getBacteriaFold(1, 10)
--print(net.model)
--print(testSet)
--print(main:test(net, testSet, opt))
--
return main
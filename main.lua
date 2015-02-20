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
    self.indices = self.indices or torch.range(1, opt.trainSize, opt.batchSize)
    u.shuffle(self.indices):apply(function(batch_index)
        --      local batchtime = sys.clock()
        local inputs, targets = u.create_minibatch(dataset, batch_index, opt.batchSize, opt.trainSize, opt.geometry)
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
        if opt.type == 'vb' then
            local lc = net:calc_lc(opt)
            print('LC: ', lc)
        end
        xlua.progress(t, opt.trainSize)
        t = t + 1
    end)
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
        collectgarbage()
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
    lcLogger = optim.Logger(paths.concat(opt.network_name, 'lc.log'))
    torch.manualSeed(3)
    torch.setnumthreads(opt.threads)
    print('<torch> set nb of threads to ' .. torch.getnumthreads())
--    local net = Convnet:buildModel(opt)
    local net = MLP:buildModel(opt)
    local trainSet, testSet = data.getMnist()

    while true do
        local trainAccuracy, trainError = self:train(net, trainSet, opt)
        print(trainAccuracy, trainError)
        local testAccuracy, testError = self:test(net, testSet, opt)
        print(testAccuracy, testError)
        accLogger:add{['train acc.'] = trainAccuracy, ['test acc.'] = testAccuracy}
        accLogger:style{['train acc.'] = '-', ['test acc.'] = '-'}
        errLogger:add{['train err.'] = trainError, ['test err.'] = testError}
        errLogger:style{['train err.'] = '-', ['test err.'] = '-' }
        if opt.type == 'vb' then
            local lc = net:calc_lc(opt)
            print('LC: ', lc)
            lcLogger:add{['LC'] = lc}
            lcLogger:style{['LC'] = '-' }
        end
        if opt.plot then
            accLogger:plot()
            errLogger:plot()
            lcLogger:plot()
        end
        u.safe_save(net, opt.network_name, 'model')
--          net:save()
    end

end
main:run()
--local net = MLP:load('vsadf2')
local net = torch.load('vsadf2/model')
local opt = net.opt
opt.testSamples = 5
opt.quicktest = false
--local opt = require('config')
local trainSet, testSet = data.getMnist()
print(net.model)
print(testSet)
print(main:test(net, testSet, opt))
--
return main
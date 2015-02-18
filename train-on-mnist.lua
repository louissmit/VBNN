require 'nn'
require 'optim'
require 'gfx.js'
local inspect = require 'inspect'
local u = require('utils')
local viz = require('visualize')
NNparams = require('NNparams')
VBparams = require('VBparams')
VBSSparams = require('VBSSparams')
require 'torch'
require 'cunn'
--torch.setdefaulttensortype('torch.CudaTensor')
--print( inspect(cutorch.getDeviceProperties(cutorch.getDevice()) ))
local mnist = require('mnist')
opt = {}
opt.threads = 1
opt.network_to_load = "gravess"
opt.network_name = "ttttttt"
opt.type = "vb"
--opt.cuda = true
opt.trainSize = 100
opt.testSize = 1000

opt.plot = true
opt.batchSize = 1
opt.B = (opt.trainSize/opt.batchSize)--*100
opt.hidden = {100}
opt.S = 10
opt.testSamples = 5
opt.alpha = 0.8 -- NVIL
--opt.normcheck = true
--opt.plotlc = true
--opt.viz = true
-- fix seed
--torch.manualSeed(3)

opt.mu_init = 0.1
opt.var_init = torch.pow(0.075, 2)--torch.sqrt(2/opt.hidden[1])--0.01
opt.pi_init = {
    mu = 5,
    var = 0.00001
}
-- optimisation params
opt.levarState = {
    learningRate = 0.000001,
--    learningRateDecay = 0.01
}
--opt.lcvarState = {
--    learningRate = 0.0000001,
--    learningRateDecay = 0.001
--}
opt.lemeanState = {
    learningRate = 0.00000001,
--    learningRateDecay = 0.01
}
--opt.lcmeanState = {
--    learningRate = 0.000000001,
--    learningRateDecay = 0.01
--}
opt.lepiState = {
    learningRate = 0.000001,
}
--opt.lcpiState = {
--    learningRate = 0.000001,
--}
opt.smState = {
    learningRate = 0.00000002,
}

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

torch.setdefaulttensortype('torch.FloatTensor')


----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'0','1','2','3','4','5','6','7','8','9'}

-- geometry: width and height of input images
geometry = {28,28}
local input_size = geometry[1]*geometry[2]

-- define model to train
model = nn.Sequential()
------------------------------------------------------------
-- regular 2-layer MLP
------------------------------------------------------------
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
model:add(nn.Linear(opt.hidden[#opt.hidden], #classes))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
if opt.cuda then
    model:cuda()
    criterion:cuda()
end

-- retrieve parameters and gradients
parameters, gradParameters = model:getParameters()

print("nr. of parameters: ", parameters:size(1))

if opt.network_to_load == '' then
    if opt.type == 'vb' then
        beta = VBparams:init(opt.W, opt)
    elseif opt.type == 'ssvb' then
        beta = VBSSparams:init(opt.W, opt)
    else
        beta = NNparams:init(parameters, opt)
    end
else
   print('<trainer> reloading previously trained network')
    if opt.type == 'ssvb' then
        beta = VBSSparams:load(opt.network_to_load)
    elseif opt.type == 'vb' then
        beta = VBparams:load(opt.network_to_load)
    else
        beta = NNparams:load(opt.network_to_load)
    end
end

-- verbose
print('<mnist> using model:')


----------------------------------------------------------------------
-- preprocess dataset
--

trainData = mnist.traindataset()
testData = mnist.testdataset()
trainData = {inputs=trainData.data:type('torch.FloatTensor'), targets=trainData.label}
u.normalize(trainData.inputs)
--trainData = u.select_data(trainData, {1,2,3,13,15,16,17,18,19,20})
testData = {inputs=testData.data:type('torch.FloatTensor'), targets=testData.label}
u.normalize(testData.inputs)

----------------------------------------------------------------------
-- define training and testing functions
--

-- training function
function train(dataset, type)
    print("Training!")
    -- epoch tracker
    epoch = epoch or 1
    local B = opt.trainSize/opt.batchSize
    local accuracy = 0
    local error = 0
    local avg_lc = 0
    local avg_le = 0

    -- local vars
    local time = sys.clock()

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1, opt.trainSize,opt.batchSize do
        --      local batchtime = sys.clock()
        local inputs, targets = u.create_minibatch(dataset, t, opt.batchSize, opt.trainSize, geometry)
        if opt.cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end

        collectgarbage()

        -- reset gradients
        gradParameters:zero()

        if type == 'vb' then
            -- sample W
            local lc, le, acc = beta:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
            accuracy = accuracy + acc
            avg_lc = avg_lc + lc
            avg_le = avg_le + le

        elseif type == 'ssvb' then

            local le, lc, acc = beta:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
            accuracy = accuracy + acc
            avg_lc = avg_lc + lc
            avg_le = avg_le + le
        else
            local err, acc = beta:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
            error = error + err
            accuracy = accuracy + acc
--            print("params:min(): ", torch.min(parameters))
--            print("params:max(): ", torch.max(parameters))
        end
        --      print("batchtime: ", sys.clock() - batchtime)

        -- disp progress
        xlua.progress(t, opt.trainSize)
    end

    -- time taken
    time = sys.clock() - time
    time = time / opt.testSize
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    if type == 'vb' or type == 'ssvb' then
        print("beta.means:min(): ", torch.min(beta.means))
        print("beta.means:max(): ", torch.max(beta.means))
        print("beta.vars:min(): ", torch.min(torch.exp(beta.lvars)))
        print("beta.vars:max(): ", torch.max(torch.exp(beta.lvars)))
    end
    if type == 'ssvb' then
        print("beta.pi:min(): ", torch.min(beta.pi))
        print("beta.pi:max(): ", torch.max(beta.pi))
        print("beta.pi:avg(): ", torch.mean(beta.pi))
    end

    -- save/log current net
    beta:save(opt)

    -- next epoch
    epoch = epoch + 1
    if type == 'vb' or type == 'ssvb' then
        accuracy = accuracy/B
        avg_le = avg_le /B
        avg_lc = avg_lc /B
        error = avg_le --+ avg_lc
    else
        accuracy = accuracy/B
        error = error/B
    end
    return accuracy, error, avg_lc, avg_le
end

-- test function
function test(dataset, type)
    local B = opt.testSize/opt.batchSize
    local accuracy = 0
    local mu_acc = 0
    local avg_error = 0
    print("Testing!")
    -- local vars
    local time = sys.clock()
    if type == 'vb' then
        local p = parameters:narrow(1,1, opt.W)
        p:copy(beta.means)
    elseif type == 'ssvb' then
        local p = parameters:narrow(1,1, opt.W)

        local evalm = u.norm_pdf(beta.means, beta.means, torch.exp(beta.lvars))
        local evalz = u.norm_pdf(torch.Tensor(opt.W):zero(), beta.means, torch.exp(beta.lvars))
        local pi = torch.pow(torch.add(torch.exp(-beta.p),1),-1)-- nn.Sigmoid(beta.p)
        evalm:cmul(pi)
        evalz:cmul(pi)
        local mapss = torch.gt(evalm, torch.add(pi, -1):add(evalz)):float()
        p:copy(torch.cmul(mapss, torch.cmul(beta.means, pi)))
--        p:copy(torch.cmul(beta.means, pi))
    end

    -- test over given dataset
    print('<trainer> on testing Set:')
    for t = 1,opt.testSize,opt.batchSize do
        -- disp progress
        xlua.progress(t, opt.testSize)

        local inputs, targets = u.create_minibatch(dataset, t, opt.batchSize, opt.testSize, geometry)
        if opt.cuda then
            inputs = inputs:cuda()
            targets = targets:cuda()
        end
        if type == 'vb' then

            local err, acc = beta:test(inputs, targets, model, parameters, criterion, opt)
            accuracy = accuracy + acc
            avg_error = avg_error + err
        else
            -- test samples
            local preds = model:forward(inputs)
            --        print(torch.gt(model:get(3).output, 0):sum())
            accuracy = accuracy + u.get_accuracy(preds, targets)
            local err = criterion:forward(preds, targets)

            avg_error = avg_error + err
        end


    end

    -- timing
    time = sys.clock() - time
    time = time / opt.testSize
    print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

    return accuracy/B, avg_error/B
end

----------------------------------------------------------------------
-- and train!
--
-- log results to files
accLogger = optim.Logger(paths.concat(opt.network_name, 'acc.log'))
errorLogger = optim.Logger(paths.concat(opt.network_name, 'error.log'))
leLogger = optim.Logger(paths.concat(opt.network_name, 'le.log'))
lcLogger = optim.Logger(paths.concat(opt.network_name, 'lc.log'))
nrlogger = optim.Logger(paths.concat(opt.network_name, 'nr.log'))

while true do
    -- train/test
--    local trainaccuracy, trainerror, lc, le = train(trainData, opt.type)
    if opt.viz then
--        viz.show_input_parameters(parameters, parameters:size(), opt)
        viz.show_input_parameters(beta.means, beta.means:size(), 'means', opt)
        viz.show_input_parameters(beta.lvars, beta.lvars:size(), 'vars', opt)
        if type == 'ssvb' then
            viz.show_input_parameters(beta.p, beta.p:size(), 'pi', opt)
        end
--        viz.show_parameters(beta.means, beta.vars, beta.pi, opt.hidden, opt.cuda)
    end
    print("TRAINACCURACY: ", trainaccuracy, trainerror)
    local testaccuracy, testerror = test(testData, opt.type)
    print("TESTACCURACY: ", testaccuracy, testerror)
exit()

--    viz.graph_things(accuracies)
    accLogger:add{['% accuracy (train set)'] = trainaccuracy, ['% accuracy (test set)'] = testaccuracy }
    errorLogger:add{['LL (train set)'] = trainerror, ['LL (test set)'] = testerror }
    if opt.type == 'ssvb' or opt.type == 'vb' then
        leLogger:add({['LE'] = le})
        lcLogger:add({['LC'] = lc})
    end

    -- plot errors
    if opt.plot then
        accLogger:style{['% accuracy (train set)'] = '-', ['% accuracy (test set)'] = '-'}
        errorLogger:style{['LL (train set)'] = '-', ['LL (test set)'] = '-' }
        if opt.type == 'ssvb' or opt.type=='vb' then
            leLogger:style({['LE'] = '-'})
            lcLogger:style({['LC'] = '-'})
            leLogger:plot()
            lcLogger:plot()
        end
        accLogger:plot()
        errorLogger:plot()

    end
end

--require 'nn'
--require 'nnx'
require 'optim'
require 'gfx.js'
--local _ = require ("moses")
local inspect = require 'inspect'
local u = require('utils')
local viz = require('visualize')
--require 'pl'
--require 'paths'
NNparams = require('NNparams')
VBparams = require('VBparams')
VBSSparams = require('VBSSparams')
--require 'rmsprop'
require 'adam'
require 'cutorch'
require 'cunn'
--torch.setdefaulttensortype('torch.CudaTensor')
print(  cutorch.getDeviceProperties(cutorch.getDevice()) )
torch.setdefaulttensortype('torch.FloatTensor')
local mnist = require('mnist')

opt = {}
opt.threads = 8
opt.network_to_load = ""
opt.network_name = "picorrection"
opt.type = "ssvb"
opt.trainSize = 60000
opt.testSize = 10000
opt.plot = true
opt.batchSize = 100
opt.B = (opt.trainSize/opt.batchSize)--*1000
opt.hidden = {12}
opt.S = 25
opt.alpha = 0.9 -- NVIL
opt.normcheck = true
-- fix seed
torch.manualSeed(1)

-- optimisation params
opt.varState = {
    learningRate = 0.0000001,
    momentumDecay = 0.1,
    updateDecay = 0.9
}
opt.meanState = {
    learningRate = 0.00000001,
    momentumDecay = 0.1,
    updateDecay = 0.9
}
opt.piState = {
    learningRate = 0.00000000001,
    momentumDecay = 0.1,
    updateDecay = 0.9
}

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())



----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'0','1','2','3','4','5','6','7','8','9'}

-- geometry: width and height of input images
geometry = {28,28}
local input_size = geometry[1]*geometry[2]

-- define model to train
unwrapped_model = nn.Sequential()
------------------------------------------------------------
-- regular 2-layer MLP
------------------------------------------------------------
unwrapped_model:add(nn.View(input_size))
unwrapped_model:add(nn.Linear(input_size, opt.hidden[1]))
unwrapped_model:add(nn.ReLU())
for i = 2, #opt.hidden do
    unwrapped_model:add(nn.Linear(opt.hidden[i-1], opt.hidden[i]))
    unwrapped_model:add(nn.ReLU())
end
unwrapped_model:add(nn.Linear(opt.hidden[#opt.hidden], #classes))

-- retrieve parameters and gradients
parameters, gradParameters = unwrapped_model:getParameters()
W = parameters:size(1)
print("nr. of parameters: ", W)

if opt.network_to_load == '' then
    if opt.type == 'vb' then
        beta = VBparams:init(W, opt)
    elseif opt.type == 'ssvb' then
        beta = VBSSparams:init(W, opt)
    else
        beta = NNparams:init(W, opt)
    end
else
   print('<trainer> reloading previously trained network')
   local filename = paths.concat(opt.network_to_load, 'network')
   beta = torch.load(filename)
end

-- verbose
print('<mnist> using model:')
print(unwrapped_model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
unwrapped_model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

model = nn.Sequential()
model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
model:add(unwrapped_model)
model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor'))

----------------------------------------------------------------------
-- preprocess dataset
--

trainData = mnist.traindataset()
testData = mnist.testdataset()
trainData = {inputs=trainData.data:type('torch.FloatTensor'), targets=trainData.label}
u.normalize(trainData.inputs)
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
        collectgarbage()

        -- reset gradients
        gradParameters:zero()

        if type == 'vb' then
            -- sample W
            local lc, le, acc = beta:train(inputs, targets, unwrapped_model, criterion, parameters, gradParameters, opt)
            accuracy = accuracy + acc
            avg_lc = avg_lc + lc
            avg_le = avg_le + le
        elseif type == 'ssvb' then
            local le, lc, acc = beta:train(inputs, targets, unwrapped_model, criterion, parameters, gradParameters, opt)
            accuracy = accuracy + acc
            avg_lc = avg_lc + lc
            avg_le = avg_le + le
        else
            local err, acc = beta:train(inputs, targets, unwrapped_model, criterion, parameters, gradParameters, opt)
            error = error + err
            accuracy = accuracy + acc
        end
        --      print("batchtime: ", sys.clock() - batchtime)

        -- disp progress
        xlua.progress(t, opt.trainSize)
    end

    -- time taken
    time = sys.clock() - time
    time = time / opt.testSize
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    if type == 'vb' then
        print("beta.means:min(): ", torch.min(beta.means))
        print("beta.means:max(): ", torch.max(beta.means))
        print("beta.vars:min(): ", torch.min(torch.exp(beta.lvars)))
        print("beta.vars:max(): ", torch.max(torch.exp(beta.lvars)))
    elseif type == 'ssvb' then
        print("beta.pi:min(): ", torch.min(beta.pi))
        print("beta.pi:max(): ", torch.max(beta.pi))
        print("beta.pi:avg(): ", torch.mean(beta.pi))
    end

    -- save/log current net
    local filename = paths.concat(opt.network_name, 'network')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    if type == 'vb' or type == 'ssvb' then
        torch.save(filename, beta)
    else
        torch.save(filename, parameters)
    end

    -- next epoch
    epoch = epoch + 1
    if type == 'vb' or type == 'ssvb' then
        accuracy = accuracy/B
        avg_le = avg_le /B
        avg_lc = avg_lc /B
        error = avg_le + avg_lc
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
        parameters:copy(beta.means)
    elseif type == 'ssvb' then
--        parameters:copy(torch.cmul(beta.means, beta.pi))
--    local evalm = torch.Tensor(W):map2(beta.means, beta.vars, function(_, mu, var)
--        return u.norm_pdf(mu,mu,var)
--    end)
--    local evalz = torch.Tensor(W):map2(beta.means, beta.vars, function(_, mu, var)
--        return u.norm_pdf(0,mu,var)
--    end)
--print(evalm)
--        parameters:copy(torch.max(evalm, evalz))
        parameters:copy(torch.cmul(beta.means, torch.pow(beta.pi,2):cmul(torch.pow(beta.vars, -1))))
    end

    -- test over given dataset
    print('<trainer> on testing Set:')
    for t = 1,opt.testSize,opt.batchSize do
        -- disp progress
        xlua.progress(t, opt.testSize)

        local inputs, targets = u.create_minibatch(dataset, t, opt.batchSize, opt.testSize, geometry)

        -- test samples
        local preds = unwrapped_model:forward(inputs)
        accuracy = accuracy + u.get_accuracy(preds, targets)
        local err = criterion:forward(preds, targets)
        avg_error = avg_error + err

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
--    viz.show_uncertainties(model, parameters, testData, beta.means, beta.vars, opt.hidden)
--    break
    -- train/test
--viz.show_parameters(beta.means, beta.lvars, beta.p, opt.hidden)
    local trainaccuracy, trainerror, lc, le = train(trainData, opt.type)
--    local trainaccuracy, lccc, leee = train(trainData, opt.type)
    print("TRAINACCURACY: ", trainaccuracy, trainerror)
    local testaccuracy, testerror = test(testData, opt.type)
    print("TESTACCURACY: ", testaccuracy, testerror)

--    viz.graph_things(accuracies)
    accLogger:add{['% accuracy (train set)'] = trainaccuracy, ['% accuracy (test set)'] = testaccuracy }
    errorLogger:add{['LL (train set)'] = trainerror, ['LL (test set)'] = testerror }
    if opt.type == 'ssvb' then
        leLogger:add({['LE'] = le})
        lcLogger:add({['LC'] = lc})
    end


    -- plot errors
    if opt.plot then
        accLogger:style{['% accuracy (train set)'] = '-', ['% accuracy (test set)'] = '-'}
        errorLogger:style{['LL (train set)'] = '-', ['LL (test set)'] = '-' }
        if opt.type == 'ssvb' then
            leLogger:style({['LE'] = '-'})
            lcLogger:style({['LC'] = '-'})
            leLogger:plot()
            lcLogger:plot()
        end
        accLogger:plot()
        errorLogger:plot()

    end
end

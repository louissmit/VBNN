require 'nn'
--require 'nnx'
require 'optim'
print("HELLO WORLD")
--require 'image'
require 'dataset-mnist'
require 'gfx.js'
local _ = require ("moses")
local inspect = require 'inspect'
local u = require('utils')
local viz = require('visualize')
--require 'pl'
--require 'paths'
VBparams = require('VBparams')

opt = {}
opt.threads = 8
opt.network_to_load = ""
opt.network_name = "adagrad2"
opt.type = "vb"
opt.plot = true
opt.batchSize = 1
opt.hidden = 10
opt.S = 1
--opt.full = true
-- fix seed
--torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
torch.setdefaulttensortype('torch.FloatTensor')


----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'0','1','2','3','4','5','6','7','8','9'}

-- geometry: width and height of input images
geometry = {32,32}

-- define model to train
model = nn.Sequential()
------------------------------------------------------------
-- regular 2-layer MLP
------------------------------------------------------------
model:add(nn.Reshape(1024))
model:add(nn.Linear(1024, opt.hidden))
model:add(nn.ReLU())
model:add(nn.Linear(opt.hidden, #classes))

-- retrieve parameters and gradients
parameters, gradParameters = model:getParameters()
W = parameters:size(1)
print("nr. of parameters: ", W)

if opt.network_to_load == '' then
    beta = VBparams:init(W)
else
   print('<trainer> reloading previously trained network')
   local filename = paths.concat(opt.network_to_load, 'network')
   beta = torch.load(filename)
end

-- verbose
print('<mnist> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
    nbTrainingPatches = 60000
    nbTestingPatches = 10000
else
    nbTrainingPatches = 2000
    nbTestingPatches = 1000
    print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.network_name, 'train.log'))
accLogger = optim.Logger(paths.concat(opt.network_name, 'acc.log'))
testLogger = optim.Logger(paths.concat(opt.network_name, 'test.log'))


-- training function
function train(dataset, type)
    print("Training!")
    -- epoch tracker
    epoch = epoch or 1
    local accuracy = 0
    local avg_error = 0

    -- local vars
    local time = sys.clock()
    local B = dataset:size()/opt.batchSize

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1,dataset:size(),opt.batchSize do
        --      local batchtime = sys.clock()
        local inputs, targets = u.create_minibatch(dataset, t, opt.batchSize, geometry)
        collectgarbage()

        -- reset gradients
        gradParameters:zero()

        if type == 'vb' then
            -- sample W
            local LN_squared = torch.Tensor(W):zero()
            local gradsum = torch.Tensor(W):zero()
            local outputs
            local LE = 0
            for i = 1, opt.S do
                parameters:copy(beta:sampleW())
                outputs = model:forward(inputs)
                accuracy = accuracy + u.get_accuracy(outputs, targets)
                local df_do = criterion:backward(outputs, targets)
                model:backward(inputs, df_do)
                LE = LE + criterion:forward(outputs, targets)
                LN_squared:add(torch.pow(gradParameters, 2))
                gradsum:add(gradParameters)
                gradParameters:zero()
            end
            LE = LE/opt.S

            -- update optimal prior alpha
            local mu_hat, var_hat = beta:compute_prior()
            local vb_mugrads, mlcg = beta:compute_mugrads(gradsum, B, opt.S)

            local vb_vargrads, vlcg = beta:compute_vargrads(LN_squared, B, opt.S)

            local LC = beta:calc_LC(B)
            local LD = LE + torch.sum(LC)

            avg_error = avg_error + LE

            meansgdState = meansgdState or {
                learningRate = 0.001,
                momentum = 0.9,
                learningRateDecay = 5e-7
            }
            varsgdState = varsgdState or {
                learningRate = 0.0001,
                momentum = 0.9
            }
--            print("vb_vargrads: ",torch.min(vb_vargrads), torch.max(vb_vargrads))
--            print("vb_mugrads: ", torch.min(vb_mugrads), torch.max(vb_mugrads))
            optim.adagrad(function(_) return LD, vb_mugrads end, beta.means, meansgdState)
            optim.adagrad(function(_) return LD, vb_vargrads end, beta.vars, varsgdState)
        else
            -- evaluate function for complete mini batch
            local outputs = model:forward(inputs)

            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)
            local f = criterion:forward(outputs, targets)
            accuracy = accuracy + u.get_accuracy(outputs, targets)
            avg_error = avg_error + f
            sgdState = sgdState or {
                learningRate = 0.1,
                momentum = 0.0,
                learningRateDecay = 5e-7
            }

            optim.sgd(function(_) return f, gradParameters end, parameters, sgdState)
        end
        --      print("batchtime: ", sys.clock() - batchtime)

        -- disp progress
        xlua.progress(t, dataset:size())
    end

    -- time taken
    time = sys.clock() - time
    time = time / dataset:size()
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    avg_error = avg_error / B
    trainLogger:add{['LL (train set)'] = avg_error }
    local weights, vars
    if type == 'vb' then
        weights = torch.Tensor(W):copy(beta.means):resize(opt.hidden, 32, 32)
        vars = torch.Tensor(W):copy(beta.vars):resize(opt.hidden, 32, 32)
        print("beta.means:min(): ", torch.min(beta.means))
        print("beta.means:max(): ", torch.max(beta.means))
        print("beta.vars:min(): ", torch.min(beta.vars))
        print("beta.vars:max(): ", torch.max(beta.vars))
    else
        weights = torch.Tensor(W):copy(parameters):resize(opt.hidden, 32, 32)
    end
    viz.show_parameters(weights, vars)

    -- save/log current net
    local filename = paths.concat(opt.network_name, 'network')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    if type == 'vb' then
        torch.save(filename, beta)
    else
        torch.save(filename, parameters)
    end


    -- next epoch
    epoch = epoch + 1
    if type == 'vb' then
        accuracy = accuracy/(B*opt.S)
    else
        accuracy = accuracy/B
    end
    return accuracy
end

-- test function
function test(dataset)
    local B = dataset:size()/opt.batchSize
    local accuracy = 0
    print("Testing!")
    -- local vars
    local time = sys.clock()

    -- test over given dataset
    print('<trainer> on testing Set:')
    local avg_error = 0
    for t = 1,dataset:size(),opt.batchSize do
        -- disp progress
        xlua.progress(t, dataset:size())

        local inputs, targets = u.create_minibatch(dataset, t, opt.batchSize, geometry)

        if type == 'vb' then
            parameters:copy(beta.means)
        end

        -- test samples
        local preds = model:forward(inputs)
        accuracy = accuracy + u.get_accuracy(preds, targets)
        local err = criterion:forward(preds, targets)
        avg_error = avg_error + err

    end
    avg_error = avg_error / B

    testLogger:add{['LL (test set)'] = avg_error}
    -- timing
    time = sys.clock() - time
    time = time / dataset:size()
    print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

    return accuracy/B
end

----------------------------------------------------------------------
-- and train!
--
while true do
    -- train/test
    local trainaccuracy = train(trainData, opt.type)
    print("TRAINACCURACY: ", trainaccuracy)
    local testaccuracy = test(testData)
    accLogger:add{['% accuracy (train set)'] = trainaccuracy, ['% accuracy (test set)'] = testaccuracy}
    -- plot errors
    if opt.plot then
        trainLogger:style{['LL (train set)'] = '-'}
        accLogger:style{['% accuracy (train set)'] = '-', ['% accuracy (test set)'] = '-'}
        testLogger:style{['LL (test set)'] = '-'}
        trainLogger:plot()
        testLogger:plot()
        accLogger:plot()
    end
end

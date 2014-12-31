require 'nn'
--require 'nnx'
require 'optim'
require 'gfx.js'
--local _ = require ("moses")
--local inspect = require 'inspect'
local u = require('utils')
local viz = require('visualize')
--require 'pl'
--require 'paths'
VBparams = require('VBparams')
require 'rmsprop'
local mnist = require('mnist')

opt = {}
opt.threads = 8
opt.network_to_load = "rmsprop12"
opt.network_name = "asdf"
opt.type = ""
opt.plot = true
opt.batchSize = 1
opt.hidden = {12}
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
geometry = {28,28}
input_size = geometry[1]*geometry[2]

-- define model to train
model = nn.Sequential()
------------------------------------------------------------
-- regular 2-layer MLP
------------------------------------------------------------
model:add(nn.Reshape(input_size))
model:add(nn.Linear(input_size, opt.hidden[1]))
model:add(nn.ReLU())
for i = 2, #opt.hidden do
    model:add(nn.Linear(opt.hidden[i-1], opt.hidden[i]))
    model:add(nn.ReLU())
end
model:add(nn.Linear(opt.hidden[#opt.hidden], #classes))

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
print(beta.means:size(1))


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

trainData = mnist.traindataset()
testData = mnist.testdataset()
trainData = {inputs=trainData.data:type('torch.FloatTensor'), targets=trainData.label}
u.normalize(trainData.inputs)
testData = {inputs=testData.data:type('torch.FloatTensor'), targets=testData.label}
u.normalize(testData.inputs)

----------------------------------------------------------------------
-- define training and testing functions
--

-- log results to files
accLogger = optim.Logger(paths.concat(opt.network_name, 'acc.log'))
errorLogger = optim.Logger(paths.concat(opt.network_name, 'error.log'))


-- training function
function train(dataset, type)
    print("Training!")
    -- epoch tracker
    epoch = epoch or 1
    local accuracy = 0
    local avg_error = 0

    -- local vars
    local time = sys.clock()
    local B = nbTrainingPatches/opt.batchSize

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1, nbTrainingPatches,opt.batchSize do
        --      local batchtime = sys.clock()
        local inputs, targets = u.create_minibatch(dataset, t, opt.batchSize, nbTrainingPatches, geometry)
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

--            meansgdState = meansgdState or {
--                learningRate = 0.00001
--                momentum = 0.9
--                learningRateDecay = 5e-7
--            }
--            varsgdState = varsgdState or {
--                learningRate = 0.000001
--                momentum = 0.9
--            }
            varsgdState = varsgdState or {
                learningRate = 0.000001,
                momentumDecay = 0.1,
                updateDecay = 0.01
            }
            meansgdState = meansgdState or {
                learningRate = 0.0001,
                momentumDecay = 0.1,
                updateDecay = 0.01
            }

            --            print("vb_vargrads: ",torch.min(vb_vargrads), torch.max(vb_vargrads))
--            print("vb_mugrads: ", torch.min(vb_mugrads), torch.max(vb_mugrads))
            rmsprop(function(_) return LD, vb_mugrads end, beta.means, meansgdState)
            rmsprop(function(_) return LD, vb_vargrads end, beta.vars, varsgdState)
        else
            -- evaluate function for complete mini batch
            local outputs = model:forward(inputs)

            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)
            local f = criterion:forward(outputs, targets)
            accuracy = accuracy + u.get_accuracy(outputs, targets)
            avg_error = avg_error + f
            meansgdState = state or {
                learningRate = 0.0001,
                momentumDecay = 0.1,
                updateDecay = 0.01
            }

            rmsprop(function(_) return f, gradParameters end, parameters, state)
        end
        --      print("batchtime: ", sys.clock() - batchtime)

        -- disp progress
        xlua.progress(t, nbTrainingPatches)
    end

    -- time taken
    time = sys.clock() - time
    time = time / nbTestingPatches
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    avg_error = avg_error / B
    local weights, vars
    if type == 'vb' then
        print("beta.means:min(): ", torch.min(beta.means))
        print("beta.means:max(): ", torch.max(beta.means))
        print("beta.vars:min(): ", torch.min(beta.vars))
        print("beta.vars:max(): ", torch.max(beta.vars))
    else
--        weights = torch.Tensor(W):copy(parameters):resize(opt.hidden, 32, 32)
    end

    -- save/log current net
    local filename = paths.concat(opt.network_name, 'network')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    if type == 'vb' then
        torch.save(filename, beta)
        torch.save(filename..'mustate', meansgdState)
        torch.save(filename..'varstate', varsgdState)
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
    return accuracy, avg_error
end

-- test function
function test(dataset, type)
    local B = nbTestingPatches/opt.batchSize
    local accuracy = 0
    print("Testing!")
    -- local vars
    local time = sys.clock()
    if type == 'vb' then
        parameters:copy(beta.means)
    end

    -- test over given dataset
    print('<trainer> on testing Set:')
    local avg_error = 0
    for t = 1,nbTestingPatches,opt.batchSize do
        -- disp progress
        xlua.progress(t, nbTestingPatches)

        local inputs, targets = u.create_minibatch(dataset, t, opt.batchSize, nbTestingPatches, geometry)

        -- test samples
        local preds = model:forward(inputs)
        accuracy = accuracy + u.get_accuracy(preds, targets)
        local err = criterion:forward(preds, targets)
        avg_error = avg_error + err

    end
    avg_error = avg_error / B

    -- timing
    time = sys.clock() - time
    time = time / nbTestingPatches
    print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

    return accuracy/B, avg_error
end

----------------------------------------------------------------------
-- and train!
--
while true do
    viz.show_uncertainties(model, parameters, testData, beta.means, beta.vars, opt.hidden)
--    viz.show_parameters(beta.means, beta.vars, opt.hidden)
    break
    -- train/test
    local trainaccuracy, trainerror = train(trainData, opt.type)
    print("TRAINACCURACY: ", trainaccuracy)
    local testaccuracy, testerror = test(testData, opt.type)
    accLogger:add{['% accuracy (train set)'] = trainaccuracy, ['% accuracy (test set)'] = testaccuracy }
    errorLogger:add{['LL (train set)'] = trainerror, ['LL (test set)'] = testerror}

    -- plot errors
    if opt.plot then
        accLogger:style{['% accuracy (train set)'] = '-', ['% accuracy (test set)'] = '-'}
        errorLogger:style{['LL (train set)'] = '-', ['LL (test set)'] = '-'}
        accLogger:plot()
        errorLogger:plot()
    end
end

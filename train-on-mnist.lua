require 'nn'
--require 'nnx'
require 'optim'
print("HELLO WORLD")
--require 'image'
require 'dataset-mnist'
require 'gfx.js'
--require 'pl'
--require 'paths'
VBparams = require('VBparams')

opt = {}
opt.threads = 8
opt.network_to_load = ""
opt.network_name = "sgdS1"
opt.plot = true
opt.learningRate = 0.1
opt.batchSize = 10
opt.momentum = 0.0
opt.hidden = 28
opt.S = 10
--opt.full = true
-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is a recommended')
end

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
   beta = VBparams:init(parameters)
else
   print('<trainer> reloading previously trained network')
   local filename = paths.concat(opt.network_name, 'model')
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
function get_accuracy(outputs, targets)
   local correct, total = 0, 0
   for i = 1, targets:size(1) do
      total = total + 1
      _, index = outputs[i]:max(1)
      if index[1] == targets[i] then
         correct = correct + 1
      end
   end
   return (correct / total)*100
end



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
      -- create mini batch
--      local batchtime = sys.clock()
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end
      collectgarbage()

      -- reset gradients
      gradParameters:zero()
      -- create closure to evaluate f(X) and df/dX
      -- Perform SGD step:
      sgdState = sgdState or {
         learningRate = opt.learningRate,
         momentum = opt.momentum,
         learningRateDecay = 5e-7
      }
      if type == 'vb' then
         -- sample W
         local LN_squared = torch.Tensor(W):zero()
         local gradsum = torch.Tensor(W):zero()
         local outputs
         local LE = 0
         for i = 1, opt.S do
            parameters:copy(beta:sampleW())
            outputs = model:forward(inputs)
            accuracy = accuracy + get_accuracy(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)
            LE = LE + criterion:forward(outputs, targets)
            LN_squared:add(torch.pow(gradParameters, 2))
            gradsum:add(gradParameters)
            gradParameters:zero()
         end
         LE = LE/opt.S
         accuracy = accuracy/opt.S

         -- update optimal prior alpha
         local mu_hat, var_hat = beta:compute_prior()
         local vb_mugrads, mlcg = beta:compute_mugrads(gradsum, B, opt.S)

         local vb_vargrads, vlcg = beta:compute_vargrads(LN_squared, B, opt.S)

         local LC = beta:calc_LC(B)
         local LD = LE + torch.sum(LC)

         avg_error = avg_error + LE

--         meanrpropState = {
--            stepsize = 0.001
--         }
--         varrpropState = {
--            stepsize = 0.0001
--         }
         meansgdState = {
            learningRate = 0.01,
            momentum = 0.9,
            learningRateDecay = 5e-7
         }
         varsgdState = {
            learningRate = 0.0001,
            momentum = 0.9,
            learningRateDecay = 5e-7
         }
         print("vb_vargrads: ",torch.min(vb_vargrads), torch.max(vb_vargrads))
         print("vb_mugrads: ", torch.min(vb_mugrads), torch.max(vb_mugrads))
         optim.sgd(function(_) return LD, vb_mugrads end, beta.means, meansgdState)
         optim.sgd(function(_) return LD, vb_vargrads end, beta.vars, varsgdState)
      else
          -- evaluate function for complete mini batch
          local outputs = model:forward(inputs)

          -- estimate df/dW
          local df_do = criterion:backward(outputs, targets)
          model:backward(inputs, df_do)
          local f = criterion:forward(outputs, targets)
          accuracy = accuracy + get_accuracy(outputs, targets)
          avg_error = avg_error + f

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
   local weights = torch.Tensor(W):copy(beta.means):resize(opt.hidden, 32, 32)
   local vars = torch.Tensor(W):copy(beta.vars):resize(opt.hidden, 32, 32)
   local meanimgs = {}
   local varimgs = {}
   for i = 1, opt.hidden do
      table.insert(meanimgs, weights[i])
      table.insert(varimgs, vars[i])
   end
   print(torch.min(beta.means))
   print(torch.max(beta.means))
   print(torch.min(beta.vars))
   print(torch.max(beta.vars))
   gfx.image(meanimgs,{zoom=3.5, legend='means'})--, min=-0.4, max=0.4})
   gfx.image(varimgs,{zoom=3.5, legend='vars'})--, min=0.0, max=0.5})
--   gfx.chart(data, {
--      chart = 'scatter', -- or: bar, stacked, multibar, scatter
--      width = 600,
--      height = 450,
--   })

   -- save/log current net
   local filename = paths.concat(opt.network_name, 'network')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, beta)

   -- next epoch
   epoch = epoch + 1
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
--      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      if type == 'vb' then
         parameters:copy(beta.means)
      end

      -- test samples
      local preds = model:forward(inputs)
      accuracy = accuracy + get_accuracy(preds, targets)
      local err = criterion:forward(preds, targets)
      avg_error = avg_error + err

      -- confusion:
--      for i = 1,opt.batchSize do
--         confusion:add(preds[i], targets[i])
--      end
   end
   avg_error = avg_error / B

   testLogger:add{['LL (test set)'] = avg_error}
   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
--   print(confusion)
--   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
--   confusion:zero()
    return accuracy/B
end

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   local trainaccuracy = train(trainData, 'vb')
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

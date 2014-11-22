----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST 
-- dataset, using multiple optimization techniques (SGD, LBFGS)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

--require 'torch'
require 'nn'
--require 'nnx'
require 'optim'
--require 'image'
require 'dataset-mnist'
--require 'pl'
--require 'paths'
VBparams = require('VBparams')

----------------------------------------------------------------------
-- parse command-line options
--
--local opt = lapp[[
--   -s,--save          (default "logs")      subdirectory to save logs
--   -n,--network       (default "")          reload pretrained network
--   -f,--full                                use the full dataset
--   -p,--plot                                plot while training
--   -r,--learningRate  (default 0.05)        learning rate, for SGD only
--   -b,--batchSize     (default 10)          batch size
--   -m,--momentum      (default 0)           momentum, for SGD only
--   -t,--threads       (default 4)           number of threads
--]]
opt = {}
opt.threads = 8
opt.network = ""
opt.save = "logs"
opt.network_name = "vbrprop"
opt.model = "mlp"
opt.plot = true
opt.learningRate = 0.1
opt.batchSize = 100
opt.momentum = 0.09
opt.hidden = 128
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

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

   ------------------------------------------------------------
   -- regular 2-layer MLP
   ------------------------------------------------------------
   model:add(nn.Reshape(1024))
   model:add(nn.Linear(1024, opt.hidden))
   model:add(nn.Tanh())
   model:add(nn.Linear(opt.hidden, #classes))
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
end

-- retrieve parameters and gradients
parameters, gradParameters = model:getParameters()
W = parameters:size(1)

-- verbose
print('<mnist> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
beta = VBparams:init(parameters)

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
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- training function
function train(dataset, type)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   local avg_error = 0

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
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
      collectgarbage()

      -- reset gradients
      gradParameters:zero()

      if type == 'vb' then
         parameters:copy(beta:sampleW())
      end

      -- evaluate function for complete mini batch
      local outputs = model:forward(inputs)

      -- estimate df/dW
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      -- create closure to evaluate f(X) and df/dX
      -- Perform SGD step:
      sgdState = sgdState or {
         learningRate = opt.learningRate,
         momentum = opt.momentum,
         learningRateDecay = 5e-7
      }
      if type == 'vb' then

         -- update optimal prior alpha
         local mu_hat = (1/W)*torch.sum(beta.means)
         local muhats = torch.Tensor(W):fill(mu_hat)
         local mu_sqe = torch.add(beta.means, torch.mul(muhats,-1)):pow(2)

         local var_hat = torch.sum(torch.add(beta.vars, mu_sqe))
         var_hat = (1/W)*var_hat

         local vb_mugrads = torch.add(beta.means, -muhats):mul(1/(var_hat*opt.batchSize)):add(gradParameters)

         local varhats = torch.Tensor(W):fill(var_hat)
         local vb_vargrads = torch.add(torch.pow(varhats,-1), -torch.pow(beta.vars, -1)):mul(1/opt.batchSize)
         vb_vargrads:add(torch.pow(gradParameters, 2))
         vb_vargrads:mul(1/2)

         local LC = torch.add(torch.Tensor(W):fill(torch.log(var_hat)), -torch.log(beta.vars))
         LC:add(mu_sqe, torch.add(beta.vars, varhats)):mul(1/2*var_hat)
         local LE = criterion:forward(outputs, targets)
         local LD = LE + torch.sum(LC)

         avg_error = avg_error + LE

         -- optimize variational posterior
--         optim.sgd(function(_) return f, vb_mugrads end, beta.means, sgdState)
--         optim.sgd(function(_) return f, vb_vargrads end, beta.vars, sgdState)
         local meanrpropState = {
            stepsize = 0.001
         }
         local varrpropState = {
            stepsize = 0.001
         }
         optim.rprop(function(_) return LD, vb_mugrads end, beta.means, meanrpropState)
         optim.rprop(function(_) return LD, vb_vargrads end, beta.vars, varrpropState)
      else
         local f = criterion:forward(outputs, targets)

         local feval = function(x) return f, gradParameters end
         avg_error = avg_error + f
         optim.sgd(feval, parameters, sgdState)
      end


      -- disp progress
      xlua.progress(t, dataset:size())
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   avg_error = avg_error / (dataset:size() / opt.batchSize)
   trainLogger:add{['% error (train set)'] = avg_error * 100}

   -- save/log current net
   local filename = paths.concat(opt.save, opt.network_name)
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   local avg_error = 0
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

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

      -- test samples
      local preds = model:forward(inputs)
      local err = criterion:forward(preds, targets)
      avg_error = avg_error + err

      -- confusion:
--      for i = 1,opt.batchSize do
--         confusion:add(preds[i], targets[i])
--      end
   end
   avg_error = avg_error / (dataset:size() / opt.batchSize)
   testLogger:add{['% error (test set)'] = avg_error * 100}
   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
--   print(confusion)
--   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
--   confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   train(trainData, 'vb')
   test(testData)

   -- plot errors
   if opt.plot then
      trainLogger:style{['% error (train set)'] = '-'}
      testLogger:style{['% error (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end

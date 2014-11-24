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
print("HELLO WORLD")
--require 'image'
require 'dataset-mnist'
require 'gfx.js'
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
opt.network = "asdf"
opt.save = "logs"
opt.network_name = "asdf"
opt.model = "mlp"
opt.plot = true
opt.learningRate = 0.1
opt.batchSize = 100
opt.momentum = 0.09
opt.hidden = 11
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

if opt.network == '' then
   beta = VBparams:init(parameters)
else
   print('<trainer> reloading previously trained network')
   local filename = paths.concat(opt.save, opt.network_name)
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
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- training function
function train(dataset, type)
   print("Training!")
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   local avg_error = 0
   local B = dataset:size()/opt.batchSize

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
--         local epsilon = 2*sqrt(1e-12)*(1+torch.norm(beta.means))
         for i = 1, opt.S do
            parameters:copy(beta:sampleW())
            outputs = model:forward(inputs)
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)
            LE = LE + criterion:forward(outputs, targets)
            LN_squared:add(torch.pow(gradParameters, 2))
            gradsum:add(gradParameters)
            gradParameters:zero()
         end
         LE = LE/opt.S

         -- update optimal prior alpha
         local mu_hat = (1/W)*torch.sum(beta.means)
         local muhats = torch.Tensor(W):fill(mu_hat)
         local mu_sqe = torch.add(beta.means, torch.mul(muhats,-1)):pow(2)

         local var_hat = torch.sum(torch.add(beta.vars, mu_sqe))
         var_hat = (1/W)*var_hat

         local vb_mugrads = torch.add(beta.means, -muhats):mul(1/(var_hat*B)):add(torch.mul(gradsum, 1/opt.S))

         local varhats = torch.Tensor(W):fill(var_hat)
         local vb_vargrads = torch.add(torch.pow(varhats,-1), -torch.pow(beta.vars, -1)):mul(1/B)
         vb_vargrads:add(LN_squared:mul(1/opt.S))
         vb_vargrads:mul(1/2)

         local LC = torch.add(torch.Tensor(W):fill(torch.log(var_hat)), -torch.log(beta.vars))
         LC:add(mu_sqe, torch.add(beta.vars, -varhats)):mul(1/2*var_hat)
         local LD = LE + torch.sum(LC)

         avg_error = avg_error + LE

         -- optimize variational posterior
--         optim.sgd(function(_) return f, vb_mugrads end, beta.means, sgdState)
--         optim.sgd(function(_) return f, vb_vargrads end, beta.vars, sgdState)
--         meanrpropState = {
--            stepsize = 0.001
--         }
--         varrpropState = {
--            stepsize = 0.0001
--         }
         meansgdState = {
            learningRate = 0.001,
            momentum = 0.9,
            learningRateDecay = 5e-7
         }
         varsgdState = {
            learningRate = 0.001,
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
   trainLogger:add{['% error (train set)'] = avg_error * 100 }
   local weights = torch.Tensor(W):copy(beta.means):resize(opt.hidden, 32, 32)
   local vars = torch.Tensor(W):copy(beta.vars):resize(opt.hidden, 32, 32)
   local meanimgs = {}
   local varimgs = {}
   for i = 1, opt.hidden do
      table.insert(meanimgs, weights[i])
      table.insert(varimgs, vars[i])
   end
   print(torch.min(weights))
   print(torch.max(weights))
   print(torch.min(vars))
   print(torch.max(vars))
   gfx.image(meanimgs,{zoom=3.5, legend='means'})--, min=-0.4, max=0.4})
   gfx.image(varimgs,{zoom=3.5, legend='vars'})--, min=0.0, max=0.5})
--   gfx.chart(data, {
--      chart = 'scatter', -- or: bar, stacked, multibar, scatter
--      width = 600,
--      height = 450,
--   })

   -- save/log current net
   local filename = paths.concat(opt.save, opt.network_name)
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, beta)

   -- next epoch
   epoch = epoch + 1
end

total = 0
correct = 0
-- test function
function test(dataset)
   print("Testing!")
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

      if type == 'vb' then
         parameters:copy(beta.means)
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
--   train(trainData, 'vb')
   test(testData)
   print(correct)
   print(total)
   print(correct/total)
   break

   -- plot errors
--   if opt.plot then
--      trainLogger:style{['% error (train set)'] = '-'}
--      testLogger:style{['% error (test set)'] = '-'}
--      trainLogger:plot()
--      testLogger:plot()
--   end
end

local u = require('utils')
require 'nn'

local convnet = {}

function convnet:buildModel(opt)
    self.opt = opt
    self.model = nn.Sequential()
    self.model:add(nn.SpatialConvolutionMM(1, 28, 5, 5))
    self.model:add(nn.ReLU())
    self.model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
    -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
    self.model:add(nn.SpatialConvolutionMM(28, 64, 5, 5))
    self.model:add(nn.ReLU())
    self.model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    -- stage 3 : standard 2-layer MLP:
    self.model:add(nn.Reshape(64*2*2))
    if opt.type == 'vb' then
        self.model:add(nn.VBLinear(64*2*2, 200, opt))
    else
        self.model:add(nn.Linear(64*2*2, 200))
    end
    self.model:add(nn.ReLU())
    if opt.type == 'vb' then
        self.model:add(nn.VBLinear(200, #opt.classes, opt))
    else
        self.model:add(nn.Linear(200, #opt.classes))
    end
    self.model:add(nn.LogSoftMax())

    self.criterion = nn.ClassNLLCriterion()
    if opt.cuda then
        self.model:cuda()
        self.criterion:cuda()
    end
    local parameters, gradParameters = self.model:getParameters()
    self.parameters = parameters
    self.gradParameters = gradParameters
    print("nr. of parameters: ", parameters:size(1))

    self.state = u.shallow_copy(opt.state)
    return self
end

function convnet:resetGradients()
    self.gradParameters:zero()
end

function convnet:sample()
    self.model:get(8):sample(self.opt)
    self.model:get(10):sample(self.opt)
end

function convnet:run(inputs, targets)
    local outputs = self.model:forward(inputs)
    local df_do = self.criterion:backward(outputs, targets)
    self.model:backward(inputs, df_do)
    local error = self.criterion:forward(outputs, targets)
    local accuracy = u.get_accuracy(outputs, targets)
    return error, accuracy
end

function convnet:update(opt)
    local x, _, update = optim.adam(
        function(_) return _, self.gradParameters:mul(1/opt.batchSize) end,
        self.parameters,
        self.state)
    if opt.type == 'vb' then
        self.model:get(8):update(opt)
        self.model:get(10):update(opt)
    end
end

return convnet
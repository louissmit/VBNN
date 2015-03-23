local u = require('utils')
require 'nn'

local convnet = {}

function convnet:buildModel(opt)
    self.vb_indices = {}
    if opt.type == 'vb' then
        self.vb_indices = {8,10 }
    end

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
        self.model:add(nn.VBLinear(64*2*2, 100, opt))
    else
        self.model:add(nn.Linear(64*2*2, 100))
    end
    self.model:add(nn.ReLU())
    if opt.type == 'vb' then
        self.model:add(nn.VBLinear(100, #opt.classes, opt))
    else
        self.model:add(nn.Linear(100, #opt.classes))
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
    for _, i in pairs(self.vb_indices) do
        self.model:get(i):resetAcc(self.opt)
    end

end

function convnet:sample()
    for _, i in pairs(self.vb_indices) do
        self.model:get(i):sample(self.opt)
    end
end

function convnet:run(inputs, targets)
    local outputs = self.model:forward(inputs)
    local df_do = self.criterion:backward(outputs, targets)
    self.model:backward(inputs, df_do)
    local error = self.criterion:forward(outputs, targets)
    local accuracy = u.get_accuracy(outputs, targets)
    return error, accuracy
end

function convnet:test(input, target)
    if self.opt.type == 'vb' then
        if self.opt.quicktest then
            for _, i in pairs(self.vb_indices) do
                self.model:get(i):clamp_to_map()
            end
            return self:run(input, target)
        else
            local error = 0
            local accuracy = 0
            for _ = 1, self.opt.testSamples do
                self:sample()
                local err, acc = self:run(input, target)
                error = error + err
                accuracy = accuracy + acc
            end
            return error/self.opt.testSamples, accuracy/self.opt.testSamples
        end
    else
        return self:run(input, target)
    end
end

function convnet:calc_lc(opt)
    local lc = 0
    for _, i in pairs(self.vb_indices) do
        lc = lc + self.model:get(i):calc_lc(opt):sum()
    end
    return lc
end


function convnet:update(opt)
    local x, _, update = optim.adam(
        function(_) return _, self.gradParameters:mul(1/opt.batchSize) end,
        self.parameters,
        self.state)
    local normratio = torch.norm(update)/torch.norm(x)
    print("normratio:", normratio)
    if opt.type == 'vb' then
        for _, i in pairs(self.vb_indices) do
            self.model:get(i):update(opt)
        end
    end
end

return convnet
local u = require('utils')
local inspect = require('inspect')
require 'nn'

local mlp = torch.class("MLP")

function mlp:buildModel(opt)
    self.opt = opt

    self.vb_indices = {}
    self.model = nn.Sequential()
    self.model:add(nn.View(opt.input_size))
    self.model:add(nn.Linear(opt.input_size, opt.hidden[1]))
    self.model:add(nn.ReLU())
    for i = 2, #opt.hidden do
        self.model:add(nn.Linear(opt.hidden[i-1], opt.hidden[i]))
        self.model:add(nn.ReLU())
    end
    self.model:add(nn.Linear(opt.hidden[#opt.hidden], #opt.classes))
    self.model:add(nn.LogSoftMax())

    self.criterion = nn.ClassNLLCriterion()
    if opt.cuda then
        self.model:cuda()
        self.criterion:cuda()
    end
    local parameters, gradParameters = self.model:getParameters()
    self.parameters = parameters
    self.gradParameters = gradParameters
    self.W = parameters:size(1)
    print(self.model)
    print("nr. of parameters: ", self.W)


    if opt.msr_init then
            for i = 1, 2 do
                local weight = self.model:get(i*2).weight
                local bias = self.model:get(i*2).bias
                bias:zero()
                newp = torch.Tensor(weight:size()):zero()
                local weight_init = torch.sqrt(2/weight:size(2))
                randomkit.normal(newp, 0, weight_init)
                weight:copy(newp)
            end
    else
        local newp = torch.Tensor(self.W)
        randomkit.normal(newp, 0, opt.weight_init)
        parameters:copy(newp)
    end


    self.state = u.shallow_copy(opt.state)
    self:reset(opt)
    return self
end

function mlp:resetGradients()
    self.model:zeroGradParameters()
end

function mlp:reset(opt)
    for _, i in pairs(self.vb_indices) do
        local weight = self.model:get(i).weight
        local W = weight:size(1)*weight:size(2)
        local sample = randomkit.normal(
            torch.Tensor(W):zero(),
            torch.Tensor(W):fill(opt.weight_init)):float():resizeAs(weight:float())
        weight:copy(sample)
    end
end

function mlp:run(inputs, targets)
    local outputs = self.model:forward(inputs)
    local df_do = self.criterion:backward(outputs, targets)
    self.model:backward(inputs, df_do)
    local error = self.criterion:forward(outputs, targets)
    local accuracy = u.get_accuracy(outputs, targets)
    return error, accuracy
end

function mlp:train(input, target)
    local err, acc = self:run(input, target)
    self:update(opt)
    return err, acc
end

function mlp:test(input, target)
    return self:run(input, target)
end

function mlp:update()
    local x, _, update = optim.adam(
        function(_) return _, self.gradParameters:mul(1/self.opt.batchSize) end,
        self.parameters,
        self.state)
    local normratio = torch.norm(update)/torch.norm(x)
    print("normratio:", normratio)
    if self.opt.log then
        Log:add('mean means', self.parameters:mean())
        Log:add('std means', self.parameters:std())
        Log:add('min. means', self.parameters:min())
        Log:add('max. means', self.parameters:max())
        Log:add('mu normratio', normratio)
    end

end

return mlp
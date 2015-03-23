local u = require('utils')
local inspect = require('inspect')
require 'nn'

local mlp = {}

function mlp:buildModel(opt)
    self.opt = opt

    self.vb_indices = {}
    self.model = nn.Sequential()
    self.model:add(nn.Reshape(opt.input_size))
    if opt.type == 'vb' then
        self.model:add(nn.VBLinear(opt.input_size, opt.hidden[1], opt))
        table.insert(self.vb_indices, 2)
    else
        self.model:add(nn.Linear(opt.input_size, opt.hidden[1]))
    end
    self.model:add(nn.ReLU())
    for i = 2, #opt.hidden do
        if opt.type == 'vb' then
            self.model:add(nn.VBLinear(opt.hidden[i-1], opt.hidden[i], opt))
            table.insert(self.vb_indices, 2*i)
        else
            self.model:add(nn.Linear(opt.hidden[i-1], opt.hidden[i]))
        end
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
    self.p = parameters:narrow(1, self.W-109, 110)
    self.g = gradParameters:narrow(1, self.W-109, 110)
    print(self.model)
    print("nr. of parameters: ", self.W)
    local newp

    for i = 1, #self.opt.hidden+1 do
        local weight = self.model:get(i*2).weight
        local bias = self.model:get(i*2).bias
        bias:zero()
        newp = torch.Tensor(weight:size()):zero()
        local weight_init = torch.sqrt(2/weight:size(2))
        randomkit.normal(newp, 0, weight_init)
        weight:copy(newp)
    end

    self.state = u.shallow_copy(opt.state)
--    self:reset(opt)
    return self
end

function mlp:resetGradients()
    self.gradParameters:zero()
    for _, i in pairs(self.vb_indices) do
        self.model:get(i):resetAcc(self.opt)
    end
end

function mlp:sample()
    for _, i in pairs(self.vb_indices) do
        self.model:get(i):sample(self.opt)
    end

end

function mlp:run(inputs, targets)
    local outputs = self.model:forward(inputs)
    local df_do = self.criterion:backward(outputs, targets)
    self.model:backward(inputs, df_do)
    local error = self.criterion:forward(outputs, targets)
--    local error = 1-torch.exp(-nll)
    local accuracy = u.get_accuracy(outputs, targets)
    return error, accuracy
end

function mlp:test(input, target)
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

function mlp:calc_lc(opt)
    local lc = 0
    for _, i in pairs(self.vb_indices) do
        lc = lc + self.model:get(i):calc_lc(opt):sum()
    end
    return lc
end

function mlp:update(opt)
    local x, _, update
    if opt.type == 'vb' then
        x, _, update = optim.sgd(
            function(_) return _, self.g end,
            self.p,
            self.state)
    else
        x, _, update = optim.sgd(
            function(_) return _, self.gradParameters end,
            self.parameters,
            self.state)
        Log:add('mean means', self.parameters:mean())
        Log:add('std means', self.parameters:std())
        Log:add('min. means', self.parameters:min())
        Log:add('max. means', self.parameters:max())
    end

--    local normratio = torch.norm(update)/torch.norm(x)
--    print("normratio:", normratio)
    if opt.type == 'vb' then
        for _, i in pairs(self.vb_indices) do
            self.model:get(i):update(opt)
        end
    end
end

return mlp
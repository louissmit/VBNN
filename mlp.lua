local u = require('utils')
local inspect = require('inspect')
require 'cunn'

local mlp = {}

function mlp:buildModel(opt)
    self.opt = opt

    local input_size = opt.geometry[1]*opt.geometry[2]
    self.model = nn.Sequential()
    self.model:add(nn.Reshape(input_size))
    if opt.type == 'vb' then
        self.model:add(nn.VBLinear(input_size, opt.hidden[1], opt))
    else
        self.model:add(nn.Linear(input_size, opt.hidden[1]))
    end
    self.model:add(nn.ReLU())
    for i = 2, #opt.hidden do
        if opt.type == 'vb' then
            self.model:add(nn.VBLinear(opt.hidden[i-1], opt.hidden[i], opt))
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
    print(self.model)
    print("nr. of parameters: ", parameters:size(1))

    self.state = u.shallow_copy(opt.state)
    return self
end

function mlp:resetGradients()
    self.gradParameters:zero()
end

function mlp:sample()
    self.model:get(2):sample(self.opt)
end

function mlp:run(inputs, targets)
    local outputs = self.model:forward(inputs)
    local df_do = self.criterion:backward(outputs, targets)
    self.model:backward(inputs, df_do)
    local error = self.criterion:forward(outputs, targets)
    local accuracy = u.get_accuracy(outputs, targets)
    return error, accuracy
end

function mlp:test(input, target)
    if self.opt.type == 'vb' then
        self.model:get(2):clamp_to_map()
    end
    return self:run(input, target)
end

function mlp:update(opt)
    local x, _, update = optim.adam(
        function(_) return _, self.gradParameters:mul(1/opt.batchSize) end,
        self.parameters,
        self.state)
    local normratio = torch.norm(update)/torch.norm(x)
--    print("normartio:", normratio)
    if opt.type == 'vb' then
        self.model:get(2):update(opt)
    end
end

return mlp
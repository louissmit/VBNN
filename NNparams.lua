--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 11/21/14
-- Time: 1:34 PM
-- To change this template use File | Settings | File Templates.
--
require 'randomkit'
local u = require('utils')
local inspect = require 'inspect'

local NNparams = {}

function NNparams:init(parameters, opt)
    local size = parameters:size(1)
    local newp = torch.Tensor(opt.W)
    randomkit.normal(newp, 0, torch.sqrt(opt.var_init))
    parameters:narrow(1, 1, opt.W):copy(newp)
    self.optimState = opt.smState
    self.update_counter = 0
    self.parameters = parameters

    return self
end

function NNparams:load(network_to_load)
    self.update_counter = 0
    parameters:copy(torch.load(paths.concat(network_to_load, 'parameters')))
    self.optimState = torch.load(paths.concat(network_to_load, 'optimstate'))
    return self
end

function NNparams:save(opt)
    u.safe_save(parameters, opt.network_name, 'parameters')
    u.safe_save(self.optimState, opt.network_name, 'optimstate')
end

function NNparams:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
    local outputs = model:forward(inputs)
--    print(model:get(4).output)
--    local p = parameters:narrow(1, parameters:size(1)-110,110)
--    local g = gradParameters:narrow(1, parameters:size(1)-110,110)
--    exit()

    -- estimate df/dW
    local df_do = criterion:backward(outputs, targets)
    model:backward(inputs, df_do)
    local error = criterion:forward(outputs, targets)
    local accuracy = u.get_accuracy(outputs, targets)
    local x, _, update = optim.adam(function(_) return error, gradParameters:mul(1/opt.batchSize) end, parameters, self.optimState)
    local normratio = torch.norm(update)/torch.norm(x)

    if opt.normcheck and (self.update_counter % 10)== 0 then
        nrlogger:add{['w'] = normratio}
        nrlogger:style({['w'] = '-'})
        nrlogger:plot()
    end


    self.update_counter = self.update_counter + 1
    return error, accuracy
end

return NNparams
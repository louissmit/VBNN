--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 11/21/14
-- Time: 1:34 PM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'randomkit'
local u = require('utils')
local inspect = require 'inspect'

local NNparams = {}
torch.setdefaulttensortype('torch.FloatTensor')

function NNparams:init(W, opt)
    self.optimState = opt.meanState
    self.update_counter = 0
    return self
end

function NNparams:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
    local outputs = model:forward(inputs)

    -- estimate df/dW
    local df_do = criterion:backward(outputs, targets)
    model:backward(inputs, df_do)
    local error = criterion:forward(outputs, targets)
    local accuracy = u.get_accuracy(outputs, targets)
    local x, _, update = adam(function(_) return error, gradParameters:mul(1/opt.batchSize) end, parameters, self.optimState)

    if opt.normcheck and (self.update_counter % 10)== 0 then
        print('hiiiiiiiiii')
        --        print("MU: ", mu_normratio)
        --        print("VAR: ", var_normratio)
        local normratio = torch.norm(update)/torch.norm(x)
        nrlogger:add{['w'] = normratio}
        nrlogger:style({['w'] = '-'})
        nrlogger:plot()
    end


    self.update_counter = self.update_counter + 1
    return error, accuracy
end

return NNparams
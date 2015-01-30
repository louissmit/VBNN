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

local VBparams = {}

function VBparams:init(W, opt)
    self.W = W
    self.lvars = torch.Tensor(W):fill(-16)
--    self.lvars = torch.Tensor(W):apply(function(_)
--        return randomkit.uniform(-10, -5)
--        return torch.log(randomkit.normal(0.001, 0.01))
--    end)
    self.means = torch.Tensor(W):apply(function(_)
        return randomkit.normal(0, 0.1)
    end)
--    self.mu_hat = 0.0
--    self.var_hat = 0.005625
    -- optimisation state
    self.varState = opt.varState
    self.meanState = opt.meanState
    self.update_counter = 0
    return self
end

function VBparams:sampleW()
    return randomkit.normal(self.means, torch.sqrt(torch.exp(self.lvars)))
end

function VBparams:compute_prior()
    self.mu_hat = (1/self.W)*torch.sum(self.means)
    local vars = torch.exp(self.lvars)
    self.mu_sqe = torch.add(self.means, -self.mu_hat):pow(2)

    self.var_hat = (1/W)*torch.sum(torch.add(vars, self.mu_sqe))
    return self.mu_hat, self.var_hat
end

function VBparams:compute_mugrads(gradsum, opt)
    local lcg = torch.add(self.means, -self.mu_hat):mul(1/(opt.B*self.var_hat))
    return torch.add(lcg, torch.mul(gradsum, 1/opt.S)), lcg
end

function VBparams:compute_vargrads(LN_squared, opt)
    local vars = torch.exp(self.lvars)
    local lcg = torch.add(-torch.pow(vars, -1), 1/self.var_hat):mul(1/opt.B)
    return torch.add(lcg, LN_squared:mul(1/opt.S)):mul(1/2):cmul(vars), lcg:mul(1/2):cmul(vars)
end

function VBparams:calc_LC(B)
    local vars = torch.exp(self.lvars)
    local LCfirst = torch.add(-torch.log(torch.sqrt(vars)), torch.log(torch.sqrt(self.var_hat)))
    local LCsecond = torch.add(self.mu_sqe, torch.add(vars, -self.var_hat)):mul(1/(2*self.var_hat))
    return torch.add(LCfirst, LCsecond):mul(1/B)
end

function VBparams:check_mugrads(gradsum, opt)
    local numgrad = u.num_grad(self.means, function()
        self:compute_prior()
        return self:calc_LC(opt.B) end)
    local le_grad, lc_grad = self:compute_mugrads(gradsum, opt)
    return lc_grad, numgrad
end

function VBparams:check_vargrads(LN_squared, opt)
    local numgrad = u.num_grad(self.lvars, function() return self:calc_LC(opt.B) end)
    local gle, glc = self:compute_vargrads(LN_squared, opt)
    return glc, numgrad
end

function VBparams:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
    local LN_squared = torch.Tensor(self.W):zero()
    local gradsum = torch.Tensor(self.W):zero()
    local outputs
    local LE = 0
    local accuracy = 0.0
    for i = 1, opt.S do
        parameters:copy(self:sampleW())
        outputs = model:forward(inputs)
        accuracy = accuracy + u.get_accuracy(outputs, targets)
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
    local mu_hat, var_hat = self:compute_prior()
    local vb_mugrads, mlcg = self:compute_mugrads(gradsum, opt)

    local vb_vargrads, vlcg = self:compute_vargrads(LN_squared, opt)

    local LC = self:calc_LC(opt.B)
--    local grad, numgrad = self:check_vargrads(LN_squared, opt)
--    print('minvargradcheck: ', torch.min(grad), torch.min(numgrad))
--    print('maxvargradcheck: ', torch.max(grad), torch.max(numgrad))
--    print('vargradcheck: ', torch.max(torch.abs(torch.add(grad, -numgrad))))
--    exit()

    local LD = LE + torch.sum(LC)

    --            print("vb_vargrads: ",torch.min(vb_vargrads), torch.max(vb_vargrads))
--                print("vb_mugrads: ", torch.min(vb_mugrads), torch.max(vb_mugrads))
    local x, _, update = adam(function(_) return LD, vb_mugrads:mul(1/opt.batchSize) end, self.means, self.meanState)

    local mu_normratio = torch.norm(update)/torch.norm(x)

--    local x, _, update = adam(function(_) return LD, vb_vargrads:mul(1/opt.batchSize) end, self.lvars, self.varState)
--    local var_normratio = torch.norm(update)/torch.norm(x)
    local var_normratio = 0
    if opt.normcheck and (self.update_counter % 1)==0 then
--        print("MU: ", mu_normratio)
--        print("VAR: ", var_normratio)
        nrlogger:style({['mu'] = '-',['var'] = '-'})
        nrlogger:add{['mu'] = mu_normratio, ['var'] = var_normratio}
        nrlogger:plot()
    end


    self.update_counter = self.update_counter + 1
    return torch.sum(LC), LE, accuracy
end

return VBparams
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

local VBparams = {}
torch.setdefaulttensortype('torch.FloatTensor')

function VBparams:init(W, opt)
    self.W = W
    self.vars = torch.Tensor(W):fill(0.005625)
    self.means = torch.Tensor(W):apply(function(_)
        return randomkit.normal(0, 0.1)
    end)
    self.mu_hat = 0.0
    self.var_hat = 0.005625
    -- optimisation state
    self.varState = opt.varState
    self.meanState = opt.meanState
    return self
end

function VBparams:sampleW()
    return randomkit.normal(self.means, torch.sqrt(self.vars))
end

function VBparams:compute_prior()
    self.mu_hat = (1/self.W)*torch.sum(self.means)
    self.mu_sqe = torch.add(self.means, -self.mu_hat):pow(2)

    self.var_hat = (1/W)*torch.sum(torch.add(self.vars, self.mu_sqe))
    return self.mu_hat, self.var_hat
end

function VBparams:compute_mugrads(gradsum, B, S)
    local lcg = torch.add(self.means, -self.mu_hat):mul(1/(B*self.var_hat))
    return torch.add(lcg, torch.mul(gradsum, 1/S)), lcg
end

function VBparams:compute_vargrads(LN_squared, B, S)
    local lcg = torch.add(-torch.pow(self.vars, -1), 1/self.var_hat):mul(1/B)
    return torch.add(lcg, LN_squared:mul(1/S)):mul(1/2), lcg
end

function VBparams:calc_LC(B)
    local LCfirst = torch.add(-torch.log(torch.sqrt(self.vars)), torch.log(torch.sqrt(self.var_hat)))
    local LCsecond = torch.add(self.mu_sqe, torch.add(self.vars, -self.var_hat)):mul(1/(2*self.var_hat))
    return torch.add(LCfirst, LCsecond):mul(1/B)
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

    -- update optimal prior alpha
    local mu_hat, var_hat = self:compute_prior()
    local vb_mugrads, mlcg = self:compute_mugrads(gradsum, opt.B, opt.S)

    local vb_vargrads, vlcg = self:compute_vargrads(LN_squared, opt.B, opt.S)

    local LC = self:calc_LC(opt.B)
    local LD = LE + torch.sum(LC)


    varsgdState = varsgdState or {
        learningRate = 0.000001,
        momentumDecay = 0.1,
        updateDecay = 0.01
    }
    meansgdState = meansgdState or {
        learningRate = 0.0001,
        momentumDecay = 0.1,
        updateDecay = 0.01
    }
    --            print("vb_vargrads: ",torch.min(vb_vargrads), torch.max(vb_vargrads))
    --            print("vb_mugrads: ", torch.min(vb_mugrads), torch.max(vb_mugrads))
    rmsprop(function(_) return LD, vb_mugrads end, self.means, meansgdState)
    rmsprop(function(_) return LD, vb_vargrads end, self.vars, varsgdState)
    return LE, accuracy
end

return VBparams
--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 11/21/14
-- Time: 1:34 PM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'randomkit'
require 'nn'
local u = require('utils')
local inspect = require 'inspect'


local VBSSparams = {}
torch.setdefaulttensortype('torch.FloatTensor')

function VBSSparams:init(W)
    self.W = W
    self.vars = torch.Tensor(W):fill(0.005625)
    self.means = torch.Tensor(W):apply(function(_)
        return randomkit.normal(0, 0.1)
    end)
--    self.p = torch.Tensor(W):fill(0.1)
    self.p = torch.Tensor(W):apply(function(_)
        return randomkit.normal(0, 1)
    end)
    self.mu_hat = (1/self.W)*torch.sum(self.means)
    self.var_hat = 0.005625
    self.pi_hat = 0.5
    -- optimisation state
    self.varState = opt.varState
    self.meanState = opt.meanState
    return self
end

function VBSSparams:sampleTheta()
    local w = randomkit.normal(torch.Tensor(self.W):zero(), torch.Tensor(self.W):fill(1.0))
    w = w:type('torch.FloatTensor')
--    local z = randomkit.binomial(torch.Tensor(self.W):fill(1.0), self.pi)
    local pi = nn.Sigmoid():forward(self.p)
    local z = torch.Tensor(self.W):map(pi, function(_, p)
        return randomkit.binomial(1.0, p)
    end)
    return w, z
end

function VBSSparams:compute_prior()
--    self.mu_hat = (1/self.W)*torch.sum(self.means)
    self.mu_sqe = torch.add(self.means, -self.mu_hat):pow(2)

    self.var_hat = (1/W)*torch.sum(torch.add(self.vars, self.mu_sqe))
    self.pi_hat = (1/self.W)*torch.sum(nn.Sigmoid():forward(self.p))
    print('p_hat', self.pi_hat)
    return self.mu_hat, self.var_hat
end

function VBSSparams:compute_mugrads(gradsum, opt)
    local lcg = torch.add(self.means, -self.mu_hat):mul(1/(opt.B*self.var_hat))
    return torch.add(lcg, torch.mul(gradsum, 1/opt.S)), lcg
end

function VBSSparams:compute_vargrads(LN_squared, opt)
    local lcg = torch.add(-torch.pow(self.vars, -1), 1/self.var_hat):mul(1/opt.B)
    return torch.add(lcg, LN_squared:mul(1/opt.S)):mul(1/2), lcg
end

function VBSSparams:compute_pgrads(opt)
    local pi = nn.Sigmoid():forward(self.p)
    local minpi = torch.add(-pi, 1)
    local grad = torch.add(-torch.log(minpi), -torch.log(1-self.pi_hat))
    grad:add(torch.log(pi))
    grad:add(torch.log(self.pi_hat))
    grad:cmul(torch.cmul(pi, minpi))
    return grad:mul(1/opt.B)
end

function VBSSparams:calc_LC(B)
    local LCfirst = torch.add(-torch.log(torch.sqrt(self.vars)), torch.log(torch.sqrt(self.var_hat)))
    local LCsecond = torch.add(self.mu_sqe, torch.add(self.vars, -self.var_hat)):mul(1/(2*self.var_hat))

    local pi = nn.Sigmoid():forward(self.p)
    local minpi = torch.add(-pi, 1)
    local LCthird = torch.add(torch.cmul(minpi, torch.log(minpi)), torch.cmul(pi, torch.log(pi)))
    LCthird:add(torch.add(torch.mul(minpi, torch.log(1-self.pi_hat)), torch.mul(pi, torch.log(self.pi_hat))))
    return torch.add(LCfirst, LCsecond, LCthird):mul(1/B)
end

function VBSSparams:check_pgrads(opt)
    local numgrad = u.num_grad(self.p, function() return self:calc_LC(opt.B) end)
    local grad = self:compute_pgrads(opt)
    return grad, numgrad
end

function VBSSparams:check_mugrads(gradsum, opt)
    local numgrad = u.num_grad(self.means, function()
        self:compute_prior()
        return self:calc_LC(opt.B) end)
    local le_grad, lc_grad = self:compute_mugrads(gradsum, opt)
    return lc_grad, numgrad
end


function VBSSparams:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
    local LN_squared = torch.Tensor(self.W):zero()
    local gradsum = torch.Tensor(self.W):zero()
    local outputs
    local LE = 0
    local accuracy = 0.0
    for i = 1, opt.S do
        local w, z = self:sampleTheta()
        parameters:copy(torch.cmul(w, z))
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
    local vb_mugrads, mlcg = self:compute_mugrads(gradsum, opt)

    local vb_vargrads, vlcg = self:compute_vargrads(LN_squared, opt)

    local LC = self:calc_LC(opt.B)
    local grad, numgrad = self:check_pgrads(opt)
    print('minpgradcheck: ', torch.min(grad), torch.min(numgrad))
    print('maxpgradcheck: ', torch.max(grad), torch.max(numgrad))
    print('pgradcheck: ', torch.max(torch.add(grad, -numgrad)))
    grad, numgrad = self:check_mugrads(gradsum, opt)
    print('minmugradcheck: ', torch.min(grad), torch.min(numgrad))
    print('maxmugradcheck: ', torch.max(grad), torch.max(numgrad))
    print('mugradcheck: ', torch.max(torch.add(grad, -numgrad)))
    exit()
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

return VBSSparams
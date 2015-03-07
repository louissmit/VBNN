require 'randomkit'
require 'cunn'
require 'optim'
require 'torch'
local u = require('utils')

local VBLinear, parent = torch.class('nn.VBLinear', 'nn.Linear')

function VBLinear:__init(inputSize, outputSize, opt)
    parent.__init(self, inputSize, outputSize)
    self.lvars = torch.Tensor(outputSize, inputSize):fill(torch.log(opt.var_init))
    self.accGradSquared = torch.Tensor(outputSize, inputSize):float()
    self.W = outputSize*inputSize
    self.means = randomkit.normal(
        torch.Tensor(self.W):zero(),
        torch.Tensor(self.W):fill(opt.mu_init)):float():resizeAs(self.weight)
    print(self.means:std())
--    self.means = torch.Tensor(outputSize, inputSize):zero()
    self.meanState = u.shallow_copy(opt.meanState)
    self.varState = u.shallow_copy(opt.varState)
end

function VBLinear:sample(opt)
    local sample = randomkit.normal(
        self.means:float(),
        torch.sqrt(torch.exp(self.lvars:float()))
    ):float()
    if opt.cuda then
        sample = sample:cuda()
    end
    self.weight:copy(sample:resizeAs(self.weight))
end

function VBLinear:compute_prior()
    self.mu_hat = (1/self.W)*torch.sum(self.means)
--    self.mu_hat = 0
    local vars = torch.exp(self.lvars)
    self.mu_sqe = torch.add(self.means, -self.mu_hat):pow(2)

--    self.var_hat = torch.pow(0.075, 2)
    self.var_hat = (1/self.W)*torch.sum(torch.add(vars, self.mu_sqe))
    return self.mu_hat, self.var_hat
end

function VBLinear:compute_mugrads(opt)
    local lcg = torch.add(self.means, -self.mu_hat):div(opt.B*self.var_hat)
    return self.gradWeight:div(opt.S), lcg
end

function VBLinear:compute_vargrads(opt)
    local vars = torch.exp(self.lvars)
    local lcg = torch.add(-torch.pow(vars, -1), 1/self.var_hat):div(2*opt.B)
    return self.accGradSquared:div(2*opt.S):cmul(vars), lcg:cmul(vars)
end
function VBLinear:calc_lc(opt)
    local vars = torch.exp(self.lvars)
    local LCfirst = torch.add(-torch.log(torch.sqrt(vars)), torch.log(torch.sqrt(self.var_hat)))
    local LCsecond = torch.add(self.mu_sqe, torch.add(vars, -self.var_hat)):div(2*self.var_hat)
    return torch.add(LCfirst, LCsecond):mul(1/opt.B)
end

function VBLinear:clamp_to_map()
    self.weight:copy(self.means)
end

--function VBLinear:updateGradInput(input, gradOutput)
--end

function VBLinear:accGradParameters(input, gradOutput, scale)
    parent.accGradParameters(self, input, gradOutput, scale)
    self.accGradSquared:add(torch.pow(torch.mm(gradOutput:t(), input), 2))
end

function VBLinear:resetAcc()
    self.accGradSquared:zero()
end

function VBLinear:update(opt)
    self:compute_prior()
    local mleg, mlcg = self:compute_mugrads(opt)
    local mugrad = torch.add(mleg, mlcg)
    local vleg, vlcg = self:compute_vargrads(opt)
    local vgrad = torch.add(vleg, vlcg)
    local x, _, update = optim.adam(
        function(_) return LD, mugrad:mul(1/opt.batchSize) end,
        self.means,
        self.meanState)
    local mu_normratio = torch.norm(update)/torch.norm(x)
    local x, _, update = optim.adam(
        function(_) return LD, vgrad:mul(1/opt.batchSize) end,
        self.lvars,
        self.varState)
    local var_normratio = torch.norm(update)/torch.norm(x)
    local vars = torch.exp(self.lvars)
    print("var: ", vars:min(), vars:mean(), vars:max())
    print("means: ", self.means:min(), self.means:mean(), self.means:max())
    print('mu/var nr: ', mu_normratio, var_normratio)
    if opt.log then
--        Log:add('vlc grad', vlcg:norm())
--        Log:add('vle grad', vleg:norm())
--        Log:add('mlc grad', mlcg:norm())
--        Log:add('mle grad', mleg:norm())
        Log:add('mean variance', vars:mean())
        Log:add('min. variance', vars:min())
        Log:add('max. variance', vars:max())
        Log:add('mean means', self.means:mean())
        Log:add('min. means', self.means:min())
        Log:add('max. means', self.means:max())
        Log:add('mu normratio', mu_normratio)
        Log:add('var normratio', var_normratio)
    end

end
-- we do not need to accumulate parameters when sharing
VBLinear.sharedAccUpdateGradParameters = VBLinear.accUpdateGradParameters


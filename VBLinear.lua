require 'randomkit'
require 'cunn'
require 'optim'
require 'torch'
local u = require('utils')

local VBLinear, parent = torch.class('nn.VBLinear', 'nn.Linear')

function VBLinear:__init(inputSize, outputSize, opt)
    parent.__init(self, inputSize, outputSize)
    self.opt = opt
    self.var_init = self.opt.var_init
    self.bias:zero()
    if opt.msr_init then
        self.var_init = 2/self.weight:size(2)
    end
    print("var: ", self.var_init)
    self.lvars = torch.Tensor(outputSize, inputSize):fill(torch.log(self.var_init))
    --    self.accGradSquared = torch.Tensor(outputSize, inputSize):zero()
    self.gradSum = torch.Tensor(outputSize, inputSize):zero()
    self.W = outputSize*inputSize
    if opt.mu_init == 0 then
        self.means = torch.Tensor(outputSize, inputSize):zero()
    else
        local std_init = torch.sqrt(self.var_init)
        self.means = randomkit.normal(
            torch.Tensor(self.W):zero(),
            torch.Tensor(self.W):fill(std_init)):float():resizeAs(self.weight)
    end

    self.biasState = u.shallow_copy(opt.state)
    self.meanState = u.shallow_copy(opt.meanState)
    self.varState = u.shallow_copy(opt.varState)

    self.zeros = torch.Tensor(outputSize, inputSize):zero()
    self.ones = torch.Tensor(outputSize, inputSize):fill(1)
    self.e = torch.Tensor(outputSize, inputSize):zero()
    self.w = torch.Tensor(outputSize, inputSize):zero()

    if opt.cuda then
        self.gradSum = self.gradSum:cuda()
        self.means = self.means:cuda()
        self.lvars = self.lvars:cuda()
    end

    self:compute_prior()
end

function VBLinear:sample(opt)
    if self.e:type() == 'torch.CudaTensor' then
    self.e = self.e:float()
    self.zeros = self.zeros:float()
    self.ones = self.ones:float()
    end
    randomkit.normal(self.e:float(), self.zeros, self.ones)
    if self.opt.cuda then
        self.e = self.e:cuda()
    end
    local w = torch.add(self.means, torch.cmul(self.stdv, self.e))
    if self.opt.cuda then
        w = w:cuda()
    end
    self.weight:copy(w)
end

--function VBLinear:type(type)
--    assert(type, 'Module: must provide a type to convert to')
--    self.weight = self.weight:type(type)
--    self.bias = self.bias:type(type)
--    self.gradWeight = self.gradWeight:type(type)
--    self.gradBias = self.gradBias:type(type)
--    self.gradInput = self.gradInput:type(type)
--    self.output = self.output:type(type)
--    return self
--end
--
function VBLinear:compute_prior()
    self.vars = torch.exp(self.lvars)
    self.stdv = torch.sqrt(self.vars)
--    self.mu_hat = (1/self.W)*torch.sum(self.means)
    self.mu_hat = 0
    self.mu_sqe = torch.add(self.means, -self.mu_hat):pow(2)

--    self.var_hat = torch.pow(0.075, 2)
--    self.var_hat = self.var_init--(1/self.W)*torch.sum(torch.add(self.vars, self.mu_sqe))
    self.var_hat = (1/self.W)*torch.sum(torch.add(self.vars, self.mu_sqe))
    return self.mu_hat, self.var_hat
end

function VBLinear:compute_mugrads(opt)
    local lcg = torch.add(self.means, -self.mu_hat):div(opt.B*self.var_hat)
    return self.gradWeight:div(opt.S), lcg
end

function VBLinear:compute_vargrads(opt)
    local lcg = torch.add(-torch.pow(self.vars, -1), 1/self.var_hat):div(2*opt.B)
    return self.gradSum:div(2*opt.S):cmul(self.stdv), lcg:cmul(self.vars)
end
function VBLinear:calc_lc(opt)
    local LCfirst = torch.add(-torch.log(torch.sqrt(self.vars)), torch.log(torch.sqrt(self.var_hat)))
    local LCsecond = torch.add(self.mu_sqe, torch.add(self.vars, -self.var_hat)):div(2*self.var_hat)
    return torch.add(LCfirst, LCsecond):mul(1/opt.B)
end

function VBLinear:clamp_to_map()
    self.weight:copy(self.means)
end

--function VBLinear:updateGradInput(input, gradOutput)
--end

function VBLinear:accGradParameters(input, gradOutput, scale)
    parent.accGradParameters(self, input, gradOutput, scale)
    local grad = torch.mm(gradOutput:t(), input)
    self.gradSum:add(torch.cmul(grad, self.e))
--    self.gradSum:add(torch.pow(grad, 2))
--    self.accGradSquared:add(torch.pow(torch.mm(gradOutput:t(), input), 2))
end

function VBLinear:resetAcc()
    self.gradSum:zero()
end

function VBLinear:update(opt)
    local x, _, update = optim.sgd(
        function(_) return LD, self.gradBias end,
        self.bias,
        self.biasState)
--    local bias_normratio = torch.norm(update)/torch.norm(x)
    self:compute_prior()
    local mleg, mlcg = self:compute_mugrads(opt)
    local mugrad = torch.add(mleg, mlcg)
    local vleg, vlcg = self:compute_vargrads(opt)
    local vgrad = torch.add(vleg, vlcg)
    local x, _, update = optim.adam(
        function(_) return LD, mugrad end,
        self.means,
        self.meanState)
    local mu_normratio = torch.norm(update)/torch.norm(x)
    local x, _, update = optim.adam(
        function(_) return LD, vgrad end,
        self.lvars,
        self.varState)
    local var_normratio = torch.norm(update)/torch.norm(x)
    local vars = torch.exp(self.lvars)
--    print("var: ", vars:min(), vars:mean(), vars:max())
--    print("means: ", self.means:min(), self.means:mean(), self.means:max())
--    print('mu/var nr: ', mu_normratio, var_normratio)
    if opt.log then
        Log:add('vlc grad', vlcg:norm()/self.lvars:norm())
        Log:add('vle grad', vleg:norm()/self.lvars:norm())
        Log:add('mlc grad', mlcg:norm()/self.means:norm())
        Log:add('mle grad', mleg:norm()/self.means:norm())
        Log:add('min variance', vars:min())
        Log:add('max variance', vars:max())
        Log:add('mean variance', vars:mean())
        Log:add('var hat', self.var_hat)
        Log:add('mean means', self.means:mean())
        Log:add('std means', self.means:std())
        Log:add('min. means', self.means:min())
        Log:add('max. means', self.means:max())
        Log:add('mu normratio', mu_normratio)
        Log:add('var normratio', var_normratio)
    end

end
-- we do not need to accumulate parameters when sharing
--VBLinear.sharedAccUpdateGradParameters = VBLinear.accUpdateGradParameters


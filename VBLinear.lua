local VBLinear, parent = torch.class('nn.VBLinear', 'nn.Linear')
require 'randomkit'

function VBLinear:__init(inputSize, outputSize)
    parent.__init(self)
    self.means = torch.Tensor(outputSize, inputSize)
    self.lvars = torch.Tensor(outputSize, inputSize)
    self.W = outputSize*inputSize
end

function VBLinear:sample()
    self.weights = randomkit.normal(self.w, self.means, torch.sqrt(torch.exp(self.lvars)))
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

function VBLinear:compute_mugrads(gradsum, opt)
    local lcg = torch.add(self.means, -self.mu_hat):div(opt.B*self.var_hat)
    return gradsum:div(opt.S), lcg
end

function VBLinear:compute_vargrads(LN_squared, opt)
    local vars = torch.exp(self.lvars)
    local lcg = torch.add(-torch.pow(vars, -1), 1/self.var_hat):div(2*opt.B)
    return LN_squared:div(2*opt.S):cmul(vars), lcg:cmul(vars)
end

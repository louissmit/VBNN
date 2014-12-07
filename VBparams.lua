--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 11/21/14
-- Time: 1:34 PM
-- To change this template use File | Settings | File Templates.
--
require 'torch'
require 'randomkit'

local VBparams = {}
torch.setdefaulttensortype('torch.FloatTensor')

function VBparams:init(W)
    self.W = W
    self.vars = torch.Tensor(W):fill(0.005625)
    self.means = torch.Tensor(W):apply(function(_)
        return randomkit.normal(0, 0.1)
    end)
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

return VBparams
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
    self.lvars = torch.Tensor(W):fill(-18)
--    self.lvars = torch.Tensor(W):apply(function(_)
--        return randomkit.uniform(-10, 0)
--    end)
    self.means = torch.Tensor(W):apply(function(_)
        return randomkit.normal(0, 0.1)
    end)
--    self.means = torch.Tensor(W):fill(0.0)
    self.p = torch.Tensor(W):apply(function(_)
        return randomkit.normal(2, 0.001)
    end)
--    self.p = torch.Tensor(W):fill(15.0)
--    self.mu_hat = (1/self.W)*torch.sum(self.means)
--    self.var_hat = 0.005625
--    self.pi_hat = 0.5
    -- optimisation state
    self.varState = opt.varState
    self.meanState = opt.meanState
    self.piState = opt.piState

    self.update_counter = 0
    return self
end

function VBSSparams:sampleTheta()
    local e = randomkit.normal(torch.Tensor(self.W):zero(), torch.Tensor(self.W):fill(1.0))
    e = e:type('torch.FloatTensor')
--    local z = randomkit.binomial(torch.Tensor(self.W):fill(1.0), self.pi)
    local z = torch.Tensor(self.W):map(self.pi, function(_, p)
        return randomkit.binomial(1.0, p)
    end)
    return e, z
end

function VBSSparams:compute_prior()
    self.vars = torch.exp(self.lvars)
    self.stdv = torch.sqrt(self.vars)
    self.pi = nn.Sigmoid():forward(self.p)

    self.mu_hat = (1/self.W)*torch.sum(self.means) -- comment out for grad check
    self.mu_sqe = torch.add(self.means, -self.mu_hat):pow(2)

--    print('musqe', torch.min(self.mu_sqe))
--    print('musqe', torch.max(self.mu_sqe))
    self.var_hat = (1/self.W)*torch.sum(torch.add(self.vars, self.mu_sqe))
--    print('vars', torch.min(self.vars))
--    print('vars', torch.max(self.vars))
--    print('varhat', self.var_hat)
    self.pi_hat = (1/self.W)*torch.sum(self.pi)
    return self.mu_hat, self.var_hat
end

function VBSSparams:compute_mugrads(gradsum, opt)
    local lcg = torch.add(self.means, -self.mu_hat):mul(1/(opt.B*self.var_hat))
    return torch.add(lcg, torch.mul(gradsum, 1/opt.S)), lcg
end

function VBSSparams:compute_vargrads(gradsum, opt)
    local lcg = torch.add(-torch.pow(self.vars, -1), 1/self.var_hat):mul(1/opt.B)
--    return torch.add(lcg, gradsum:mul(1/opt.S)):mul(1/2):cmul(self.vars), lcg:mul(1/2):cmul(self.vars)
    return torch.add(lcg, gradsum:mul(1/(2*opt.S)):cmul(torch.pow(self.stdv, -1))), lcg:mul(1/2):cmul(self.vars)
end

function VBSSparams:compute_pgrads(gradsum, opt)
    local minpi = torch.add(-self.pi, 1)
    local grad = torch.log(torch.mul(torch.pow(minpi, -1), 1-self.pi_hat))
    grad:add(torch.log(torch.mul(self.pi, 1/self.pi_hat)))
    grad:add(torch.mul(minpi, -1/(1-self.pi_hat)))
    grad:add(torch.mul(self.pi, -1/self.pi_hat))
--    local grad = torch.add(-torch.log(minpi), -torch.log(1-self.pi_hat))
--    grad:add(torch.log(pi))
--    grad:add(torch.log(self.pi_hat))
--    grad:cmul(torch.cmul(pi, minpi))

    return torch.add(gradsum:mul(1/opt.S), grad:mul(1/opt.B))
end

function VBSSparams:calc_LC(B)
    local LCfirst = torch.add(-torch.log(self.stdv), torch.log(torch.sqrt(self.var_hat)))
    local LCsecond = torch.add(self.mu_sqe, torch.add(self.vars, -self.var_hat)):mul(1/(2*self.var_hat))
--    print(torch.min(torch.log(self.stdv)))
--    print(torch.max(torch.log(self.stdv)))
--    print(torch.log(torch.sqrt(self.var_hat)))

    local minpi = torch.add(-self.pi, 1)
    local LCthird = torch.add(torch.cmul(minpi, torch.log(minpi)), torch.cmul(self.pi, torch.log(self.pi)))
    LCthird:add(-torch.add(torch.mul(minpi, torch.log(1-self.pi_hat)), torch.mul(self.pi, torch.log(self.pi_hat))))
--    print("LCFirst: ", torch.sum(LCfirst))
--    print("LCsecond: ", torch.sum(LCsecond))
--    print("LCthird: ", torch.sum(LCthird))
--    exit()
    return torch.add(LCfirst, torch.add(LCsecond, LCthird)):mul(1/B)
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

function VBSSparams:check_vargrads(LN_squared, opt)
    local numgrad = u.num_grad(self.lvars, function() return self:calc_LC(opt.B) end)
    local gle, glc = self:compute_vargrads(LN_squared, opt)
    return glc, numgrad
end


function VBSSparams:runModel(inputs, targets, model, criterion, parameters, gradParameters, opt)
    local var_gradsum = torch.Tensor(self.W):zero()
    local mu_gradsum = torch.Tensor(self.W):zero()
    local pi_gradsum = torch.Tensor(self.W):zero()
    local outputs
    local LE = 0
    local accuracy = 0.0


    for i = 1, opt.S do
        local e, z = self:sampleTheta()
        local w = torch.cmul(torch.add(self.means, torch.cmul(self.stdv, e)), z)

        parameters:copy(w)
        outputs = model:forward(inputs)
        accuracy = accuracy + u.get_accuracy(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do)
        local LL = criterion:forward(outputs, targets)
        LE = LE + LL
        mu_gradsum:add(torch.cmul(z, gradParameters))
        var_gradsum:add(torch.cmul(self.stdv, torch.cmul(e, torch.cmul(z, gradParameters))))
        pi_gradsum:add(torch.add(z, -self.pi):mul(LL))
        gradParameters:zero()
    end

    LE = LE/(opt.S)
    accuracy  = accuracy/opt.S
--    print("LE: ", LE)
    local LC = self:calc_LC(opt.B)
--    print("LCmin: ",torch.min(LC))
--    print("LCmax: ",torch.max(LC))
--    accuracy = accuracy/opt.S
    return torch.add(LC, LE), LE, LC, accuracy, mu_gradsum, var_gradsum, pi_gradsum
end


function VBSSparams:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
    -- update optimal prior alpha
    local mu_hat, var_hat = self:compute_prior()
    local L,LE, LC, accuracy, mu_gradsum, var_gradsum, pi_gradsum = self:runModel(inputs, targets, model, criterion, parameters, gradParameters, opt)

    local vb_mugrads, mlcg = self:compute_mugrads(mu_gradsum, opt)
--    print("vb_mugrads: ", torch.min(vb_mugrads), torch.max(vb_mugrads))
--    local numgrad = u.num_grad(self.means, function() return self:runModel(inputs, targets, model, parameters, gradParameters, opt) end)
--    print(torch.min(mlcg), torch.max(mlcg))
--    print('minmugradcheck: ', torch.min(vb_mugrads), torch.min(numgrad))
--    print('maxmugradcheck: ', torch.max(vb_mugrads), torch.max(numgrad))
--    print('mugradcheck: ', torch.max(torch.abs(torch.add(vb_mugrads, -numgrad))))
--    print("-----------------------")
--    local grad, numgrad = self:check_mugrads(mu_gradsum, opt)
--    print('minmugradcheck: ', torch.min(grad), torch.min(numgrad))
--    print('maxmugradcheck: ', torch.max(grad), torch.max(numgrad))
--    print('mugradcheck: ', torch.max(torch.abs(torch.add(grad, -numgrad))))
--    exit()

--    local vb_vargrads, vlcg = self:compute_vargrads(var_gradsum, opt)
--    local numgrad = u.num_grad(self.lvars, function() return self:runModel(inputs, targets, model, parameters, gradParameters, opt) end)
--    print('minmugradcheck: ', torch.min(vb_vargrads), torch.min(numgrad))
--    print('maxmugradcheck: ', torch.max(vb_vargrads), torch.max(numgrad))
--    print('mugradcheck: ', torch.max(torch.abs(torch.add(vb_vargrads, -numgrad))))

    local pi_grads = self:compute_pgrads(pi_gradsum, opt)

--    local LC = self:calc_LC(opt.B)
--    local grad, numgrad = self:check_pgrads(opt)
--    print('minpgradcheck: ', torch.min(grad), torch.min(numgrad))
--    print('maxpgradcheck: ', torch.max(grad), torch.max(numgrad))
--    print('pgradcheck: ', torch.max(torch.add(grad, -numgrad)))
--    grad, numgrad = self:check_mugrads(gradsum, opt)
--    print('minmugradcheck: ', torch.min(grad), torch.min(numgrad))
--    print('maxmugradcheck: ', torch.max(grad), torch.max(numgrad))
--    print('mugradcheck: ', torch.max(torch.add(grad, -numgrad)))
--    local grad, numgrad = self:check_vargrads(LN_squared, opt)
--    print('minvargradcheck: ', torch.min(grad), torch.min(numgrad))
--    print('maxvargradcheck: ', torch.max(grad), torch.max(numgrad))
--    print('vargradcheck: ', torch.max(torch.abs(torch.add(grad, -numgrad))))
--    exit()
    local LD = LE + torch.sum(LC)

    --            print("vb_vargrads: ",torch.min(vb_vargrads), torch.max(vb_vargrads))
    --            print("vb_mugrads: ", torch.min(vb_mugrads), torch.max(vb_mugrads))


    local x, _, update = adam(function(_) return LD, vb_mugrads:mul(1/opt.batchSize) end, self.means, self.meanState)
    local mu_normratio = torch.norm(update)/torch.norm(x)

--    local x, _, update = adam(function(_) return LD, vb_vargrads:mul(1/opt.batchSize) end, self.lvars, self.varState)
--    local var_normratio = torch.norm(update)/torch.norm(x)
    local var_normratio = 0

    local x, _, update = adam(function(_) return LD, pi_grads:mul(1/opt.batchSize) end, self.p, self.piState)
    local pi_normratio = torch.norm(update)/torch.norm(x)

--    local pi_normratio = 0
    if opt.normcheck and (self.update_counter % 1)==0 then
        nrlogger:style({['mu'] = '-',['var'] = '-',['pi'] = '-'})
        nrlogger:add{['mu'] = mu_normratio, ['var'] = var_normratio, ['pi'] = pi_normratio }
        nrlogger:plot()
    end


    self.update_counter = self.update_counter + 1
    return LE, torch.sum(LC), accuracy
end

return VBSSparams
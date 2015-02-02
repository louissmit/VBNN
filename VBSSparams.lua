--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 11/21/14
-- Time: 1:34 PM
-- To change this template use File | Settings | File Templates.
--
require 'randomkit'
local viz = require('visualize')
local u = require('utils')
local inspect = require 'inspect'


local VBSSparams = {}

function VBSSparams:init(opt)
    self.thirdlclog = viz.graph_things(opt, 'thirdlc')
    self.secondlc = viz.graph_things(opt, 'secondlc')
    self.firstlc = viz.graph_things(opt, 'firstlc')
    self.W = opt.W

    self.means = randomkit.normal(
        torch.Tensor(self.W):zero(),
        torch.Tensor(self.W):fill(opt.mu_init)):float()
    self.lvars = torch.Tensor(self.W):fill(opt.var_init)
    self.p = randomkit.normal(
        torch.Tensor(self.W):fill(opt.pi_init.mu),
        torch.Tensor(self.W):fill(opt.pi_init.var)):float()
    self.smp = parameters:narrow(1, self.W, parameters:size(1)-self.W) -- softmax layer params

    if opt.cuda then
        self.means = self.means:cuda()
        self.lvars = self.lvars:cuda()
        self.p = self.p:cuda()
    end

    -- optimisation state
    self.varState = opt.varState
    self.meanState = opt.meanState
    self.smState = opt.smState
    self.piState = opt.piState
    self.c = 0

    return self
end

function VBSSparams:sampleTheta()
    local e = randomkit.normal(torch.Tensor(self.W):zero(), torch.Tensor(self.W):fill(1.0))
    local z = randomkit.binomial(torch.Tensor(self.W):fill(1),self.pi:float())
    if opt.cuda then
        return e:cuda(), z:cuda()
    else
        return e:float(),z:float()
    end
--    return torch.Tensor(self.W):zero():cuda(), torch.Tensor(self.W):fill(1):cuda()
end

function VBSSparams:compute_prior(B)
    self.vars = torch.exp(self.lvars)
    self.stdv = torch.sqrt(self.vars)
    self.pi = torch.pow(torch.add(torch.exp(-self.p),1),-1)

    self.mu_hat = (1/self.W)*torch.sum(self.means) -- comment out for grad check
    self.mu_sqe = torch.add(self.means, -self.mu_hat):pow(2)

    self.var_hat = (1/(self.W))*torch.sum(torch.add(self.vars, self.mu_sqe))
    self.pi_hat = (1/self.W)*torch.sum(self.pi)

    return self.mu_hat, self.var_hat
end

function VBSSparams:compute_mugrads(gradsum, opt)
    local lcg = torch.add(self.means, -self.mu_hat):mul(1/(opt.B*self.var_hat))
    return torch.add(lcg, torch.mul(gradsum, 1/opt.S)), lcg
end

function VBSSparams:compute_vargrads(gradsum, opt)
    local lcg = torch.add(-torch.pow(self.vars, -1), 1/self.var_hat):cmul(self.vars):mul(1/opt.B)
    return torch.add(lcg, gradsum:mul(1/opt.S)):mul(1/2), lcg:mul(1/2)
--    return torch.add(lcg, gradsum:mul(1/(2*opt.S)):cmul(torch.pow(self.stdv, -1))), lcg:mul(1/2):cmul(self.vars)
end

function VBSSparams:compute_pgrads(gradsum, opt)
    local minpi = torch.add(-self.pi, 1)
    local grad = torch.log(torch.mul(torch.pow(minpi, -1), 1-self.pi_hat))
    grad:add(torch.log(torch.mul(self.pi, 1/self.pi_hat)))
    grad:mul(1/opt.B)
    grad:add(gradsum:mul(1/opt.S))
    grad:cmul(torch.cmul(self.pi, torch.add(self.pi, -1)))

    return grad
end

function VBSSparams:calc_LC(B)
--    print("stdv", torch.mean(self.stdv))
--    print("varhat", torch.sqrt(self.var_hat))
    local LCfirst = torch.add(-torch.log(self.stdv), torch.log(torch.sqrt(self.var_hat)))
--    local diffvar = torch.add(self.vars, -self.var_hat):mul(1/(2*self.var_hat))
--    print(torch.mean(diffvar))
--    print(torch.mean(self.mu_sqe))
    local LCsecond = torch.add(self.mu_sqe, torch.add(self.vars, -self.var_hat)):mul(1/(2*self.var_hat))

    local minpi = torch.add(-self.pi, 1)
    local LCthird = torch.add(torch.cmul(minpi, torch.log(minpi)), torch.cmul(self.pi, torch.log(self.pi)))
    LCthird:add(-torch.add(torch.mul(minpi, torch.log(1-self.pi_hat)), torch.mul(self.pi, torch.log(self.pi_hat))))
    if opt.plotlc then
    self.thirdlclog:add(LCthird:sum())
    self.thirdlclog:plot()
    self.secondlc:add(LCsecond:sum())
    self.secondlc:plot()
    self.firstlc:add(LCfirst:sum())
    self.firstlc:plot()
    end
--    print("LCFirst: ", torch.sum(torch.mul(LCfirst, 1/B)))
--    print("LCFirst: ", torch.min(torch.mul(LCfirst, 1/B)))
--    print("LCFirst: ", torch.max(torch.mul(LCfirst, 1/B)))
--    print("LCsecond: ", torch.sum(torch.mul(LCsecond, 1/B)))
--    print("LCthird: ", torch.sum(torch.mul(LCthird, 1/B)))
    return torch.add(LCfirst, torch.add(LCsecond, LCthird)):mul(1/B)
end

function VBSSparams:check_pgrads(gradsum, opt)
    local numgrad = u.num_grad(self.p, function() return self:calc_LC(opt.B) end)
    local _, lc_grad = self:compute_pgrads(gradsum, opt)
    return lc_grad, numgrad
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

function VBSSparams:load(model_dir)
    local dir = paths.concat(model_dir, 'parameters')
    self.means = torch.load(paths.concat(dir, 'means'))
    self.lvars = torch.load(paths.concat(dir, 'lvars'))
    self.p = torch.load(paths.concat(dir, 'p'))
    self.smp = torch.load(paths.concat(dir, 'smp'))
    dir = paths.concat(model_dir, 'optimstate')
    self.meanState = torch.load(paths.concat(dir, 'mean'))
    self.varState = torch.load(paths.concat(dir, 'var'))
    self.piState = torch.load(paths.concat(dir, 'pi'))
    self.smState = torch.load(paths.concat(dir, 'smp'))
    self.W = torch.load(paths.concat(model_dir, 'opt')).W

    parameters:narrow(1, self.W, parameters:size(1)-self.W):copy(self.smp) --load softmax layer
    return self
end

function VBSSparams:save(opt)
    u.safe_save(opt, opt.network_name, 'opt')
    local dir = paths.concat(opt.network_name, 'parameters')
    os.execute('mkdir -p ' .. dir)
    u.safe_save(self.means, dir, 'means')
    u.safe_save(self.lvars, dir, 'lvars')
    u.safe_save(self.p, dir, 'p')
    u.safe_save(self.smp, dir, 'smp')
    local dir = paths.concat(opt.network_name, 'optimstate')
    os.execute('mkdir ' .. dir)
    u.safe_save(self.meanState, dir, 'mean')
    u.safe_save(self.varState, dir, 'var')
    u.safe_save(self.piState, dir, 'pi')
    u.safe_save(self.smState, dir, 'smp')
end

function VBSSparams:runModel(inputs, targets, model, criterion, parameters, gradParameters, opt)
    local smsize = parameters:size(1)-self.W
    local var_gradsum = torch.Tensor(self.W):zero()
    local mu_gradsum = torch.Tensor(self.W):zero()
    local pi_gradsum = torch.Tensor(self.W):zero()
    local pi_gradsum2 = torch.Tensor(self.W):zero()
    local sm_gradsum = torch.Tensor(smsize):zero()
    if opt.cuda then
        var_gradsum = var_gradsum:cuda()
        mu_gradsum = mu_gradsum:cuda()
        pi_gradsum = pi_gradsum:cuda()
        pi_gradsum2 = pi_gradsum2:cuda()
    end

    local outputs
    local LE = 0
    local accuracy = 0.0
    local LL_sum = 0


    for i = 1, opt.S do
        local e, z = self:sampleTheta()
        local w = torch.cmul(torch.add(self.means, torch.cmul(self.stdv, e)), z)

        local p = parameters:narrow(1,1, self.W)
        local g = gradParameters:narrow(1,1, self.W)
        p:copy(w)

        outputs = model:forward(inputs)
        local LL = criterion:forward(outputs, targets)
        accuracy = accuracy + u.get_accuracy(outputs, targets)

        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do)


        LE = LE + LL
        LL_sum = LL_sum + LL
        local smg = gradParameters:narrow(1, self.W, smsize)
        sm_gradsum:add(smg)
        mu_gradsum:add(torch.cmul(z, g))
        var_gradsum:add(torch.cmul(self.stdv, torch.cmul(e, torch.cmul(z, g))))

        local pz = torch.add(z, -self.pi)
        pi_gradsum:add(pz:mul(LL))
--        pi_gradsum2:add(pz)
        gradParameters:zero()
    end
--    print((LL_sum/opt.S))
--    self.c = self.c*opt.alpha - (1-opt.alpha)*(LL_sum/opt.S)
--    self.c = self.c/opt.batchSize
--    print(self.c)
--    pi_gradsum = pi_gradsum + pi_gradsum2:mul(self.c)
    LE = LE/(opt.S)
    accuracy  = accuracy/opt.S
    local LC = self:calc_LC(opt.B)
    print("LE: ", LE)
    print("LC: ",torch.sum(LC))
--    print("LCmin: ",torch.min(LC))
--    print("LCmax: ",torch.max(LC))
--    print(mu_gradsum:norm(), var_gradsum:norm(), pi_gradsum:norm())
--    exit()
--    accuracy = accuracy/opt.S
    return torch.add(LC, LE), LE, LC, accuracy, mu_gradsum, var_gradsum, -pi_gradsum, sm_gradsum:mul(1/opt.S)
end


function VBSSparams:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
    -- update optimal prior alpha
    self:compute_prior(opt.B)

    local L,LE, LC, accuracy, mu_gradsum, var_gradsum, pi_gradsum, sm_grad = self:runModel(inputs, targets, model, criterion, parameters, gradParameters, opt)
--    print("var_gradsum: ", torch.mean(var_gradsum), torch.var(var_gradsum))

    local vb_mugrads, mlcg = self:compute_mugrads(mu_gradsum, opt)
    local vb_vargrads, vlcg = self:compute_vargrads(var_gradsum, opt)
    local pi_grads = self:compute_pgrads(pi_gradsum, opt)

    local LD = LE + torch.sum(LC)

--    print("vb_vargrads: ",torch.min(vb_vargrads), torch.max(vb_vargrads))
--    print("var_vargrads: ", torch.mean(vb_vargrads), torch.var(vb_vargrads))
--    print("vb_mugrads: ", torch.min(vb_mugrads), torch.max(vb_mugrads))
--    print("vb_pigrads: ", torch.min(pi_grads), torch.max(pi_grads))


    local x, _, update = adam(function(_) return LD, vb_mugrads:mul(1/opt.batchSize) end, self.means, self.meanState)
    local mu_normratio = torch.norm(update)/torch.norm(x)

    local x, _, update = adam(function(_) return LD, vb_vargrads:mul(1/opt.batchSize) end, self.lvars, self.varState)
    local var_normratio = torch.norm(update)/torch.norm(x)
--    local var_normratio = 0

    local x, _, update = adam(function(_) return LD, pi_grads:mul(1/opt.batchSize) end, self.p, self.piState)
    local pi_normratio = torch.norm(update)/torch.norm(x)
--
    local x, _, update = adam(function(_) return _, sm_grad:mul(1/opt.batchSize) end, self.smp, self.smState)

--    local pi_normratio = 0
    if opt.normcheck then
        nrlogger:style({['mu'] = '-',['var'] = '-',['pi'] = '-'})
        nrlogger:add{['mu'] = mu_normratio, ['var'] = var_normratio, ['pi'] = pi_normratio }
        nrlogger:plot()
    end


    return LE, torch.sum(LC), accuracy
end

return VBSSparams
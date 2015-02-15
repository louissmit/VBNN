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


function VBSSparams:init(W, opt)
    self.W = W
    self.lvars = torch.Tensor(W):fill(torch.log(opt.var_init))
    self.means = randomkit.normal(
        torch.Tensor(W):zero(),
        torch.Tensor(W):fill(opt.mu_init)):float()
    self.p = randomkit.normal(
        torch.Tensor(self.W):fill(opt.pi_init.mu),
        torch.Tensor(self.W):fill(opt.pi_init.var)):float()
    self.smp = parameters:narrow(1, self.W, parameters:size(1)-self.W) -- softmax layer params

    -- optimisation state
    self.levarState = opt.levarState
    self.lcvarState = opt.lcvarState
    self.lemeanState = opt.lemeanState
    self.lcmeanState = opt.lcmeanState
    self.lepiState = opt.lepiState
    self.lcpiState = opt.lcpiState
    self.smState = opt.smState
    self.c = 0

    self.zeros = torch.Tensor(W):zero()
    self.ones = torch.Tensor(W):fill(1.0)

    return self
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
        self.thirdlclog = viz.graph_things(opt, 'thirdlc')
    self.secondlc = viz.graph_things(opt, 'secondlc')
    self.firstlc = viz.graph_things(opt, 'firstlc')
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

function VBSSparams:sampleTheta()
    local e = randomkit.normal(self.zeros, self.ones)
    local z = randomkit.binomial(self.ones, self.pi:float())
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
    local lcg = torch.add(self.means, -self.mu_hat):div(opt.B*self.var_hat)
    return gradsum:div(opt.S), lcg
end

function VBSSparams:compute_vargrads(gradsum, opt)
    local lcg = torch.add(-torch.pow(self.vars, -1), 1/self.var_hat):div(2*opt.B)
    return gradsum:div(2*opt.S):cmul(self.vars), lcg:cmul(self.vars)
--    return torch.add(lcg, gradsum:mul(1/(2*opt.S)):cmul(torch.pow(self.stdv, -1))), lcg:mul(1/2):cmul(self.vars)
end

function VBSSparams:compute_pgrads(gradsum, opt)
    local minpi = torch.add(-self.pi, 1)
    local lcg = torch.log(torch.mul(torch.pow(minpi, -1), 1-self.pi_hat))
    lcg:add(torch.log(torch.mul(self.pi, 1/self.pi_hat)))
    lcg:cmul(torch.cmul(self.pi, torch.add(self.pi, -1)))
    lcg:div(opt.B)
    local leg = torch.div(gradsum, opt.S)
    leg:cmul(torch.cmul(self.pi, torch.add(self.pi, -1)))
    return leg, lcg
end

function VBSSparams:calc_LC(B)
    local LCfirst = torch.add(-torch.log(self.stdv), torch.log(torch.sqrt(self.var_hat)))
    local LCsecond = torch.add(self.mu_sqe, torch.add(self.vars, -self.var_hat)):mul(1/(2*self.var_hat))

    local minpi = torch.add(-self.pi, 1)
    local LCthird = torch.add(torch.cmul(minpi, torch.log(minpi)), torch.cmul(self.pi, torch.log(self.pi)))
    LCthird:add(-torch.add(torch.mul(minpi, torch.log(1-self.pi_hat)), torch.mul(self.pi, torch.log(self.pi_hat))))
    print("LCFirst: ", torch.sum(torch.mul(LCfirst, 1/B)))
--    print("LCFirst: ", torch.min(torch.mul(LCfirst, 1/B)))
--    print("LCFirst: ", torch.max(torch.mul(LCfirst, 1/B)))
    print("LCsecond: ", torch.sum(torch.mul(LCsecond, 1/B)))
    print("LCthird: ", torch.sum(torch.mul(LCthird, 1/B)))
    return torch.add(LCfirst, torch.add(LCsecond, LCthird)):mul(1/B)
--    return LCthird
end

function VBSSparams:runModel(inputs, targets, model, criterion, parameters, gradParameters, opt)
    local smsize = parameters:size(1)-self.W
    local var_gradsum = torch.Tensor(self.W):zero()
    local mu_gradsum = torch.Tensor(self.W):zero()
    local pi_gradsum = torch.Tensor(self.W):zero()
    local pi_gradsum2 = torch.Tensor(self.W):zero()
    local sm_gradsum = torch.Tensor(smsize):zero()

    local outputs
    local LE = 0
    local accuracy = 0.0
    local LL_sum = 0


    for i = 1, opt.S do
        local e, z = self:sampleTheta()
        local w, pz
        if opt.cuda then
            w = torch.cmul(torch.add(self.means:cuda(), torch.cmul(self.stdv:cuda(), e)), z)
            pz = torch.add(z, -self.pi:cuda())
            sm_gradsum = sm_gradsum:cuda()
        else
            w = torch.cmul(torch.add(self.means, torch.cmul(self.stdv, e)), z)
            pz = torch.add(z, -self.pi)
        end

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
        mu_gradsum:add(torch.cmul(z, g):float())
        var_gradsum:add(torch.cmul(e, torch.cmul(z, g)):float())

        pi_gradsum:add(pz:mul(LL):float())
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
    return torch.add(LC, LE), LE, LC, accuracy, mu_gradsum, var_gradsum, pi_gradsum, sm_gradsum:mul(1/opt.S)
end


function VBSSparams:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
    -- update optimal prior alpha
    self:compute_prior(opt.B)

    local L,LE, LC, accuracy, mu_gradsum, var_gradsum, pi_gradsum, sm_grad = self:runModel(inputs, targets, model, criterion, parameters, gradParameters, opt)

    local mleg, mlcg = self:compute_mugrads(mu_gradsum, opt)
    local vleg, vlcg = self:compute_vargrads(var_gradsum, opt)
    local pleg, plcg = self:compute_pgrads(pi_gradsum, opt)
    local LD = LE + torch.sum(LC)

--    print("vb_vargrads: ",torch.min(vb_vargrads), torch.max(vb_vargrads))
--    print("var_vargrads: ", torch.mean(vb_vargrads), torch.var(vb_vargrads))
--    print("vb_mugrads: ", torch.min(vb_mugrads), torch.max(vb_mugrads))
--    print("vb_pigrads: ", torch.min(pi_grads), torch.max(pi_grads))
    print("VLEG: ", vleg:norm())
    print("VLCG: ", vlcg:norm())
    mleg = torch.add(mleg, mlcg)
    vleg = torch.add(vleg, vlcg)
    pleg = torch.add(pleg, plcg)

    local x, _, update = optim.adam(function(_) return LD, mleg:mul(1/opt.batchSize) end, self.means, self.lemeanState)
    local mule_normratio = torch.norm(update)/torch.norm(x)
--    local x, _, update = optim.adam(function(_) return LD, mlcg:mul(1/opt.batchSize) end, self.means, self.lcmeanState)
--    local mulc_normratio = torch.norm(update)/torch.norm(x)

    local x, _, update = optim.adam(function(_) return LD, vleg:mul(1/opt.batchSize) end, self.lvars, self.levarState)
    local varle_normratio = torch.norm(update)/torch.norm(x)
--    local x, _, update = optim.adam(function(_) return LD, vlcg:mul(1/opt.batchSize) end, self.lvars, self.lcvarState)
--    local varlc_normratio = torch.norm(update)/torch.norm(x)
--    local mule_normratio = 0


    local x, _, update = optim.adam(function(_) return LD, pleg:mul(1/opt.batchSize) end, self.p, self.lepiState)
    local ple_normratio = torch.norm(update)/torch.norm(x)
--    local x, _, update = optim.adam(function(_) return LD, plcg:mul(1/opt.batchSize) end, self.p, self.lcpiState)
--    local plc_normratio = torch.norm(update)/torch.norm(x)
    print("varlenr: ",varle_normratio)
    print("mulenr : ",mule_normratio)
    print("pilenr : ",ple_normratio)
    local x, _, update = optim.adam(
        function(_) return _, sm_grad:mul(1/opt.batchSize) end,
            self.smp, self.smState)
    if opt.normcheck then
--        print("MU: ", mu_normratio)
--        print("VAR: ", var_normratio)
        nrlogger:style({['mule'] = '-',['mulc'] = '-',['varle'] = '-',['varlc'] = '-', ['ple'] = '-',['plc'] = '-'})
--        nrlogger:style({['mu'] = '-',['var'] = '-'})
--        nrlogger:add{['mu'] = mu_normratio, ['var'] = var_normratio}
        nrlogger:add{['mule'] = mule_normratio, ['mulc'] = mulc_normratio,['varle'] = varle_normratio,['varlc'] = varlc_normratio
            ,['ple'] = ple_normratio,['plc'] = plc_normratio
        }
        nrlogger:plot()
    end
    print("beta.means:min(): ", torch.min(beta.means))
    print("beta.means:max(): ", torch.max(beta.means))
    print("beta.vars:min(): ", torch.min(torch.exp(beta.lvars)))
    print("beta.vars:avg(): ", torch.mean(torch.exp(beta.lvars)))
    print("beta.vars:max(): ", torch.max(torch.exp(beta.lvars)))
    print("beta.pi:min(): ", torch.min(beta.pi))
    print("beta.pi:max(): ", torch.max(beta.pi))
    print("beta.pi:avg(): ", torch.mean(beta.pi))
--

    return LE, torch.sum(LC), accuracy
end

return VBSSparams
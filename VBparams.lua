--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 11/21/14
-- Time: 1:34 PM
-- To change this template use File | Settings | File Templates.
--
require 'randomkit'
local u = require('utils')
local inspect = require 'inspect'

local VBparams = {}

function VBparams:init(W, opt)
    self.W = W
    self.lvars = torch.Tensor(W):fill(torch.log(opt.var_init))
--    self.lvars:narrow(1, self.W-1010, 1010):fill(opt.var_init2)
    if opt.mu_init == 0 then
        self.means = torch.Tensor(W):zero()
    else
        self.means = randomkit.normal(
            torch.Tensor(W):zero(),
            torch.Tensor(W):fill(opt.mu_init)):float()
    end


    -- optimisation state
    self.levarState = opt.levarState
    self.lcvarState = opt.lcvarState
    self.lemeanState = opt.lemeanState
    self.lcmeanState = opt.lcmeanState
    self.w = torch.Tensor(W)
    return self
end

function VBparams:load(model_dir)
    local dir = paths.concat(model_dir, 'parameters')
    self.means = torch.load(paths.concat(dir, 'means'))
    self.lvars = torch.load(paths.concat(dir, 'lvars'))
    self.smp = torch.load(paths.concat(dir, 'smp'))
    dir = paths.concat(model_dir, 'optimstate')
    self.lemeanState = torch.load(paths.concat(model_dir, 'opt')).lemeanState
    self.lcmeanState = torch.load(paths.concat(model_dir, 'opt')).lcmeanState
    self.levarState = torch.load(paths.concat(model_dir, 'opt')).levarState
    self.lcvarState = torch.load(paths.concat(model_dir, 'opt')).lcvarState
    self.smState = torch.load(paths.concat(dir, 'smp'))
    self.W = torch.load(paths.concat(model_dir, 'opt')).W
    self.w = torch.Tensor(self.W)
    return self
end

function VBparams:save(opt)
    u.safe_save(opt, opt.network_name, 'opt')
    local dir = paths.concat(opt.network_name, 'parameters')
    os.execute('mkdir -p ' .. dir)
    u.safe_save(self.means, dir, 'means')
    u.safe_save(self.lvars, dir, 'lvars')
    u.safe_save(self.smp, dir, 'smp')
    local dir = paths.concat(opt.network_name, 'optimstate')
    os.execute('mkdir ' .. dir)
    u.safe_save(self.meanState, dir, 'mean')
    u.safe_save(self.varState, dir, 'var')
    u.safe_save(self.smState, dir, 'smp')
end

function VBparams:sampleW()
--    return randomkit.normal(self.means, torch.sqrt(torch.exp(self.lvars)))
    randomkit.normal(self.w, self.means, torch.sqrt(torch.exp(self.lvars)))
    return self.w
end

function VBparams:compute_prior()
    self.vars = torch.exp(self.lvars)
    self.stdv = torch.sqrt(self.vars)
    self.mu_hat = (1/self.W)*torch.sum(self.means)
--    self.mu_hat = 0
    self.mu_sqe = torch.add(self.means, -self.mu_hat):pow(2)

--    self.var_hat = torch.pow(0.075, 2)
    self.var_hat = (1/self.W)*torch.sum(torch.add(self.vars, self.mu_sqe))
    return self.mu_hat, self.var_hat
end

function VBparams:compute_mugrads(gradsum, opt)
    local lcg = torch.add(self.means, -self.mu_hat):div(opt.B*self.var_hat)
    return gradsum:div(opt.S), lcg
end

function VBparams:compute_vargrads(gradsum, opt)
    local lcg = torch.add(-torch.pow(self.vars, -1), 1/self.var_hat):div(2*opt.B)
    return gradsum:div(2*opt.S):cmul(self.vars), lcg:cmul(self.vars)
end

function VBparams:calc_LC(opt)
    local vars = torch.exp(self.lvars)
--    print(torch.mean(torch.log(torch.sqrt(vars))))
--    print(torch.var(torch.log(torch.sqrt(vars))))
--    print(torch.log(torch.sqrt(self.var_hat)))
    local LCfirst = torch.add(-torch.log(torch.sqrt(vars)), torch.log(torch.sqrt(self.var_hat)))
    local LCsecond = torch.add(self.mu_sqe, torch.add(vars, -self.var_hat)):div(2*self.var_hat)
--    print("LCFirst: ", torch.sum(torch.div(LCfirst, opt.B)))
--    print("LCsecond: ", torch.sum(torch.div(self.mu_sqe, 2*self.var_hat):div(opt.B)))
--    print("LCthird: ", torch.sum(torch.div(vars, 2*self.var_hat):div(opt.B)))
--    print('LCfourth: ', -opt.W/(2*opt.B))
--    exit()
    return torch.add(LCfirst, LCsecond):mul(1/opt.B)
end

function VBparams:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
    local var_gradsum = torch.Tensor(self.W):zero()
    local gradsum = torch.Tensor(self.W):zero()
    local outputs
    local LE = 0
    local accuracy = 0.0
    local mu_hat, var_hat = self:compute_prior()
    for i = 1, opt.S do
        local w = self:sampleW()
        if opt.cuda then
            w = w:cuda()
        end

        parameters:copy(w)
        outputs = model:forward(inputs)
        LE = LE + criterion:forward(outputs, targets)
        accuracy = accuracy + u.get_accuracy(outputs, targets)

        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do)

        var_gradsum:add(torch.pow(gradParameters:float(), 2))
        gradsum:add(gradParameters:float())
        gradParameters:zero()
    end
    LE = LE/opt.S

    accuracy = accuracy/opt.S

    -- update optimal prior alpha
    local mleg, mlcg = self:compute_mugrads(gradsum, opt)

    local vleg, vlcg = self:compute_vargrads(var_gradsum, opt)

    local LC = self:calc_LC(opt)
    print("LC: ", LC:sum())
    print("LE: ", LE)
    print("vleg: ", vleg:norm())
    print("vlcg: ", vlcg:norm())
    print("mleg: ", mleg:norm())
    print("mlcg: ", mlcg:norm())

    local LD = LE + torch.sum(LC)
--    mleg = torch.add(mleg, mlcg)
--    vleg = torch.add(vleg, vlcg)

    local x, _, update = optim.adam(function(_) return LD, mleg:mul(1/opt.batchSize) end, self.means, self.lemeanState)
    local mule_normratio = torch.norm(update)/torch.norm(x)
--    local x, _, update = optim.adam(function(_) return LD, mlcg:mul(1/opt.batchSize) end, self.means, self.lcmeanState)
--    local mulc_normratio = torch.norm(update)/torch.norm(x)


    local x, _, update = optim.adam(function(_) return LD, vleg:mul(1/opt.batchSize) end, self.lvars, self.levarState)
    local varle_normratio = torch.norm(update)/torch.norm(x)
--    local x, _, update = optim.adam(function(_) return LD, vlcg:mul(1/opt.batchSize) end, self.lvars, self.lcvarState)
--    local varlc_normratio = torch.norm(update)/torch.norm(x)
--    local mule_normratio = 0
    print("varlenr: ",varle_normratio)
--    print("varlcnr: ",varlc_normratio)
    print("mulenr: ",mule_normratio)
--    print("mulcnr: ",mulc_normratio)
    if opt.normcheck then
--        print("MU: ", mu_normratio)
--        print("VAR: ", var_normratio)
        nrlogger:style({['mule'] = '-',['mulc'] = '-',['varle'] = '-',['varlc'] = '-'})
--        nrlogger:style({['mu'] = '-',['var'] = '-'})
--        nrlogger:add{['mu'] = mu_normratio, ['var'] = var_normratio}
        nrlogger:add{['mule'] = mule_normratio, ['mulc'] = mulc_normratio,['varle'] = varle_normratio,['varlc'] = varlc_normratio}
        nrlogger:plot()
    end
    print("beta.means:min(): ", torch.min(beta.means))
    print("beta.means:max(): ", torch.max(beta.means))
    print("beta.vars:min(): ", self.vars:min())
    print("beta.vars:dist(): ", self.vars:mean(), self.vars:std())
    print("beta.vars:max(): ", self.vars:max())

    return torch.sum(LC), LE, accuracy
end

return VBparams
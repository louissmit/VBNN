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
    self.means = randomkit.normal(
        torch.Tensor(W):zero(),
        torch.Tensor(W):fill(opt.mu_init)):float()
    self.smp = parameters:narrow(1, self.W, parameters:size(1)-self.W) -- softmax layer params

    -- optimisation state
    self.levarState = opt.levarState
    self.lcvarState = opt.lcvarState
    self.lemeanState = opt.lemeanState
    self.lcmeanState = opt.lcmeanState
    self.smState = opt.smState
    return self
end

function VBparams:load(model_dir)
    local dir = paths.concat(model_dir, 'parameters')
    self.means = torch.load(paths.concat(dir, 'means'))
    self.lvars = torch.load(paths.concat(dir, 'lvars'))
    self.smp = torch.load(paths.concat(dir, 'smp'))
    dir = paths.concat(model_dir, 'optimstate')
    self.meanState = torch.load(paths.concat(dir, 'mean'))
    self.varState = torch.load(paths.concat(dir, 'var'))
    self.smState = torch.load(paths.concat(dir, 'smp'))
    self.W = torch.load(paths.concat(model_dir, 'opt')).W
    print(self.W)
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
    return randomkit.normal(self.means, torch.sqrt(torch.exp(self.lvars)))
end

function VBparams:compute_prior()
    self.mu_hat = (1/self.W)*torch.sum(self.means)
--    self.mu_hat = 0
    local vars = torch.exp(self.lvars)
    self.mu_sqe = torch.add(self.means, -self.mu_hat):pow(2)

    self.var_hat = (1/self.W)*torch.sum(torch.add(vars, self.mu_sqe))
--    self.var_hat = 0.0001
    return self.mu_hat, self.var_hat
end

function VBparams:compute_mugrads(gradsum, opt)
    local lcg = torch.add(self.means, -self.mu_hat):mul(1/(opt.B*self.var_hat))
    return gradsum:div(opt.S), lcg
end

function VBparams:compute_vargrads(LN_squared, opt)
    local vars = torch.exp(self.lvars)
    local lcg = torch.add(-torch.pow(vars, -1), 1/self.var_hat):mul(1/opt.B)
    return LN_squared:mul(1/opt.S):mul(1/2):cmul(vars), lcg:mul(1/2):cmul(vars)
end

function VBparams:calc_LC(opt)
    local vars = torch.exp(self.lvars)
--    print(torch.mean(torch.log(torch.sqrt(vars))))
--    print(torch.var(torch.log(torch.sqrt(vars))))
--    print(torch.log(torch.sqrt(self.var_hat)))
    local LCfirst = torch.add(-torch.log(torch.sqrt(vars)), torch.log(torch.sqrt(self.var_hat)))
    local LCsecond = torch.add(self.mu_sqe, torch.add(vars, -self.var_hat)):mul(1/(2*self.var_hat))
--    print("LCFirst: ", torch.sum(torch.div(LCfirst, opt.B)))
--    print("LCsecond: ", torch.sum(torch.div(self.mu_sqe, 2*self.var_hat):div(opt.B)))
--    print("LCthird: ", torch.sum(torch.div(vars, 2*self.var_hat):div(opt.B)))
--    print('LCfourth: ', -opt.W/(2*opt.B))
    return torch.add(LCfirst, LCsecond):mul(1/opt.B)
end

function VBparams:train(inputs, targets, model, criterion, parameters, gradParameters, opt)
    local smsize = parameters:size(1)-self.W
    local LN_squared = torch.Tensor(self.W):zero()
    local gradsum = torch.Tensor(self.W):zero()
    local sm_gradsum = torch.Tensor(smsize):zero()
    local outputs
    local LE = 0
    local accuracy = 0.0
    for i = 1, opt.S do
        local p = parameters:narrow(1,1, self.W)
        local g = gradParameters:narrow(1,1, self.W)
        local w = self:sampleW()
        if opt.cuda then
            w = w:cuda()
        end

        p:copy(w)
        outputs = model:forward(inputs)
        LE = LE + criterion:forward(outputs, targets)
        accuracy = accuracy + u.get_accuracy(outputs, targets)

        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do)

        LN_squared:add(torch.pow(g:float(), 2))
        gradsum:add(g:float())
        local smg = gradParameters:narrow(1, self.W, smsize)
        sm_gradsum:add(smg)
        gradParameters:zero()
    end
    LE = LE/opt.S
    local sm_grad= sm_gradsum:mul(1/opt.S)
    accuracy = accuracy/opt.S

    -- update optimal prior alpha
    local mu_hat, var_hat = self:compute_prior()
    local mleg, mlcg = self:compute_mugrads(gradsum, opt)

    local vleg, vlcg = self:compute_vargrads(LN_squared, opt)

    local LC = self:calc_LC(opt)
    print("LC: ", LC:sum())
    print("LE: ", LE)

    local LD = LE + torch.sum(LC)

    local x, _, update = optim.adam(function(_) return LD, mleg:mul(1/opt.batchSize) end, self.means, self.lemeanState)
    local mule_normratio = torch.norm(update)/torch.norm(x)
    local x, _, update = optim.adam(function(_) return LD, mlcg:mul(1/opt.batchSize) end, self.means, self.lcmeanState)
    local mulc_normratio = torch.norm(update)/torch.norm(x)


    local x, _, update = optim.adam(function(_) return LD, vleg:mul(1/opt.batchSize) end, self.lvars, self.levarState)
    local varle_normratio = torch.norm(update)/torch.norm(x)
    local x, _, update = optim.adam(function(_) return LD, vlcg:mul(1/opt.batchSize) end, self.lvars, self.lcvarState)
    local varlc_normratio = torch.norm(update)/torch.norm(x)
    local mule_normratio = 0

    local x, _, update = optim.adam(
        function(_) return _, sm_grad:mul(1/opt.batchSize) end,
            self.smp, self.smState)
    if opt.normcheck then
--        print("MU: ", mu_normratio)
--        print("VAR: ", var_normratio)
        nrlogger:style({['mule'] = '-',['mulc'] = '-',['varle'] = '-',['varlc'] = '-'})
--        nrlogger:style({['mu'] = '-',['var'] = '-'})
--        nrlogger:add{['mu'] = mu_normratio, ['var'] = var_normratio}
        nrlogger:add{['mule'] = mule_normratio, ['mulc'] = mulc_normratio,['varle'] = varle_normratio,['varlc'] = varlc_normratio}
        nrlogger:plot()
    end

    return torch.sum(LC), LE, accuracy
end

return VBparams
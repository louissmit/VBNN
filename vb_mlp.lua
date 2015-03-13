local vbmlp, parent = torch.class('VBMLP', 'MLP')

function vbmlp:buildModel(opt)
    parent:buildModel(opt)
    self.lvars = torch.Tensor(self.W):fill(torch.log(opt.var_init))
    --    self.lvars:narrow(1, self.W-1010, 1010):fill(opt.var_init2)
    if opt.mu_init == 0 then
        self.means = torch.Tensor(self.W):zero()
    else
        self.means = randomkit.normal(
            torch.Tensor(self.W):zero(),
            torch.Tensor(self.W):fill(opt.mu_init)):float()
    end
    self.zeros = torch.Tensor(self.W):zero()
    self.ones = torch.Tensor(self.W):fill(1.0)
    self.e = torch.Tensor(self.W)
    self.w = torch.Tensor(self.W)
    self.var_gradsum = torch.Tensor(self.W):zero()
    self.gradsum = torch.Tensor(self.W):zero()
    if opt.cuda then
        self.gradsum = self.gradsum:cuda()
        self.var_gradsum = self.var_gradsum:cuda()
        self.means = self.means:cuda()
        self.lvars = self.lvars:cuda()
    end

    self:compute_prior()

    self.meanState = opt.meanState
    self.varState = opt.varState
    return self
end


function vbmlp:sample()
    randomkit.normal(self.e:float(), self.zeros, self.ones)
    if self.opt.cuda then
        self.e = self.e:cuda()
    end
    return torch.add(self.means, torch.cmul(self.stdv, self.e))
end

function vbmlp:train(inputs, targets)
    local outputs
    local LE = 0
    local accuracy = 0.0
    -- update optimal prior alpha
    local mu_hat, var_hat = self:compute_prior()
    for i = 1, self.opt.S do
        local w = self:sample()
        if self.opt.cuda then
            w = w:cuda()
        end

        self.parameters:copy(w)
        local err, acc = self:run(inputs, targets)
        LE = LE + err
        accuracy = accuracy + acc

        self.var_gradsum:add(torch.cmul(self.gradParameters, torch.cmul(self.e, self.stdv)))
        self.gradsum:add(self.gradParameters)
        self.gradParameters:zero()
    end
    self:update()
    LE = LE/self.opt.S
    print("LE: ", LE)
    accuracy = accuracy/self.opt.S
    return LE, accuracy
end

function vbmlp:update()
    local mleg, mlcg = self:compute_mugrads()
    local vleg, vlcg = self:compute_vargrads()

    local LC = self:calc_LC()


    mleg = torch.add(mleg, mlcg)
    vleg = torch.add(vleg, vlcg)

    local x, _, update = optim.adam(function(_) return _, mleg:mul(1/self.opt.batchSize) end, self.means, self.meanState)
    local mu_normratio = torch.norm(update)/torch.norm(x)
    --    local x, _, update = optim.adam(function(_) return LD, mlcg:mul(1/opt.batchSize) end, self.means, self.lcmeanState)
    --    local mulc_normratio = torch.norm(update)/torch.norm(x)


    local x, _, update = optim.adam(function(_) return _, vleg:mul(1/self.opt.batchSize) end, self.lvars, self.varState)
    local var_normratio = torch.norm(update)/torch.norm(x)
    --    local x, _, update = optim.adam(function(_) return LD, vlcg:mul(1/opt.batchSize) end, self.lvars, self.lcvarState)
    --    local varlc_normratio = torch.norm(update)/torch.norm(x)
    --    local mule_normratio = 0
    if self.opt.log then
        Log:add('vlc grad', vlcg:norm())
        Log:add('vle grad', vleg:norm())
        Log:add('mlc grad', mlcg:norm())
        Log:add('mle grad', mleg:norm())
        Log:add('mean variance', self.vars:mean())
        Log:add('min. variance', self.vars:min())
        Log:add('max. variance', self.vars:max())
        Log:add('mean means', self.means:mean())
        Log:add('min. means', self.means:min())
        Log:add('max. means', self.means:max())
        Log:add('mu normratio', mu_normratio)
        Log:add('var normratio', var_normratio)
    end
    if self.opt.print then
        print("LC: ", LC)
        print("vleg: ", vleg:norm())
        print("vlcg: ", vlcg:norm())
        print("mleg: ", mleg:norm())
        print("mlcg: ", mlcg:norm())
        print("varlenr: ",var_normratio)
        --    print("varlcnr: ",varlc_normratio)
        print("mulenr: ",mu_normratio)
        --    print("mulcnr: ",mulc_normratio)
        print("beta.means:min(): ", torch.min(self.means))
        print("beta.means:max(): ", torch.max(self.means))
        print("beta.vars:min(): ", torch.min(torch.exp(self.lvars)))
        print("beta.vars:avg(): ", torch.mean(torch.exp(self.lvars)))
        print("beta.vars:max(): ", torch.max(torch.exp(self.lvars)))
    end
end


function vbmlp:test(input, target)
    if self.opt.quicktest then
        for _, i in pairs(self.vb_indices) do
            self.model:get(i):clamp_to_map()
        end
        return self:run(input, target)
    else
        local error = 0
        local accuracy = 0
        for _ = 1, self.opt.testSamples do
            self:sample()
            local err, acc = self:run(input, target)
            error = error + err
            accuracy = accuracy + acc
        end
        return error/self.opt.testSamples, accuracy/self.opt.testSamples
    end
end

function vbmlp:resetGradients()
    self.gradsum:zero()
    self.var_gradsum:zero()
    self.model:zeroGradParameters()
end


function vbmlp:compute_prior()
    self.vars = torch.exp(self.lvars)
    self.stdv = torch.sqrt(self.vars)
    self.mu_hat = (1/self.W)*torch.sum(self.means)
    --    self.mu_hat = 0
    self.mu_sqe = torch.add(self.means, -self.mu_hat):pow(2)

    --    self.var_hat = torch.pow(0.075, 2)
    self.var_hat = (1/self.W)*torch.sum(torch.add(self.vars, self.mu_sqe))
    return self.mu_hat, self.var_hat
end

function vbmlp:compute_mugrads()
    local lcg = torch.add(self.means, -self.mu_hat):div(self.opt.B*self.var_hat)
    return self.gradsum:div(self.opt.S), lcg
end

function vbmlp:compute_vargrads()
    local lcg = torch.add(-torch.pow(self.vars, -1), 1/self.var_hat):div(2*self.opt.B)
    return self.var_gradsum:div(2*self.opt.S):cmul(self.vars), lcg:cmul(self.vars)
end

function vbmlp:calc_LC()
    --    print(torch.mean(torch.log(torch.sqrt(vars))))
    --    print(torch.var(torch.log(torch.sqrt(vars))))
    --    print(torch.log(torch.sqrt(self.var_hat)))
    local LCfirst = torch.add(-torch.log(torch.sqrt(self.vars)), torch.log(torch.sqrt(self.var_hat)))
    local LCsecond = torch.add(self.mu_sqe, torch.add(self.vars, -self.var_hat)):div(2*self.var_hat)
    --    print("LCFirst: ", torch.sum(torch.div(LCfirst, opt.B)))
    --    print("LCsecond: ", torch.sum(torch.div(self.mu_sqe, 2*self.var_hat):div(opt.B)))
    --    print("LCthird: ", torch.sum(torch.div(vars, 2*self.var_hat):div(opt.B)))
    --    print('LCfourth: ', -opt.W/(2*opt.B))
    --    exit()
    return torch.add(LCfirst, LCsecond):mul(1/self.opt.B):sum()
end

return vbmlp
local opt = {}

opt = {}

opt.threads = 8
opt.network_to_load = ""
opt.network_name = "exp"
opt.type = "vb"
opt.dataset = 'mnist'
opt.cuda = true
opt.batchSize = 1
opt.testBatchSize = 100

if opt.dataset == 'mnist' then
opt.trainSize = 100
opt.testSize = 1000
    opt.classes = {'0','1','2','3','4','5','6','7','8','9' }
    opt.geometry = {28,28}
    opt.input_size = opt.geometry[1]*opt.geometry[2]
else
    opt.trainSize = 50--100
    opt.testSize = 49--1000
    opt.classes = {'0','1'}
    opt.testBatchSize = 49
    opt.input_size = 2283
end

opt.plot = true

opt.B = (opt.trainSize/opt.batchSize)--*100
opt.hidden = {10}
opt.S = 1
opt.testSamples = 5
--opt.quicktest = true
opt.log = true
--opt.normcheck = true
--opt.plotlc = true
--opt.viz = true

torch.manualSeed(3)

--opt.weight_init = 0.14--0.01
opt.mu_init = 0.0
--opt.var_init = 0.1--torch.pow(0.75, 2)--torch.sqrt(2/opt.hidden[1])--0.01
opt.msr_init = true
opt.pi_init = {
    mu = 5,
    var = 0.00001
}
-- optimisation params
opt.state = {
--    lambda = 1-1e-8,
    learningRate = 0.0001,
}
opt.varState = {
--    lambda = 1-1e-8,
    learningRate = 0.01,
--    learningRateDecay = 0.01
}
opt.meanState = {
--    lambda = 1-1e-8,
    learningRate = 0.0001,
--    learningRateDecay = 0.01
}
opt.piState = {
    lambda = 1-1e-8,
    learningRate = 0.00000001,
}
return opt
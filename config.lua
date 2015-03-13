local opt = {}

opt = {}
opt.classes = {'0','1','2','3','4','5','6','7','8','9'}
--opt.classes = {'0','1'}
opt.threads = 8
opt.network_to_load = ""
opt.network_name = "exp"
opt.type = "vb"
opt.cuda = true
opt.trainSize = 600
opt.testSize = 1000

opt.plot = true
opt.batchSize = 1
opt.testBatchSize = 100
opt.B = (opt.trainSize/opt.batchSize)--*100
opt.hidden = {100, 100, 100, 100, 100}
opt.S = 1
opt.testSamples = 5
opt.quicktest = true
opt.log = true
--opt.normcheck = true
--opt.plotlc = true
--opt.viz = true
opt.geometry = {28,28}
opt.input_size = opt.geometry[1]*opt.geometry[2] -- 2283

torch.manualSeed(3)
opt.mu_init = 0.0
opt.weight_init = 0.075--torch.sqrt(2/100)
opt.var_init = torch.pow(0.075,2)--torch.pow(torch.sqrt(2/784),2)--0.01
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
    learningRate = 0.0001,
--    learningRateDecay = 0.01
}
opt.meanState = {
--    lambda = 1-1e-8,
    learningRate = 0.0001,
--    learningRateDecay = 0.001
}
opt.piState = {
    lambda = 1-1e-8,
    learningRate = 0.00000001,
}
return opt
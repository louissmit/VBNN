local opt = {}

opt = {}
opt.classes = {'0','1','2','3','4','5','6','7','8','9'}
opt.threads = 1
opt.network_to_load = ""
opt.network_name = "vbbase"
opt.type = "vb"
--opt.cuda = true
opt.trainSize = 100
opt.testSize = 1000

opt.plot = true
opt.batchSize = 1
opt.B = (opt.trainSize/opt.batchSize)--*100
opt.hidden = {100}
opt.S = 10
opt.alpha = 0.8 -- NVIL
--opt.normcheck = true
--opt.plotlc = true
--opt.viz = true
-- fix seed
opt.geometry = {28,28}

opt.mu_init = 0.1
opt.var_init = torch.pow(0.075, 2)--torch.sqrt(2/opt.hidden[1])--0.01
opt.pi_init = {
    mu = 5,
    var = 0.00001
}
-- optimisation params
opt.state = {
    learningRate = 0.00000001,
}
opt.varState = {
    learningRate = 0.00001,
--    learningRateDecay = 0.01
}
opt.meanState = {
    learningRate = 0.00000001,
--    learningRateDecay = 0.01
}
opt.piState = {
    learningRate = 0.00000001,
}
return opt
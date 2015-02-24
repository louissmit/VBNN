local opt = {}

opt = {}
--opt.classes = {'0','1','2','3','4','5','6','7','8','9'}
opt.classes = {'0','1'}
opt.threads = 1
opt.network_to_load = ""
opt.network_name = "nretest"
opt.type = ""
opt.cuda = true
opt.trainSize = 90
opt.testSize = 9

opt.plot = true
opt.batchSize = 15
opt.testBatchSize = 1
opt.B = 6000--(opt.trainSize/opt.batchSize)--*100
opt.hidden = {100}
opt.S = 5
opt.testSamples = 5
opt.quicktest = true
--opt.normcheck = true
--opt.plotlc = true
--opt.viz = true
--opt.geometry = {28,28}
opt.input_size = 2283

opt.mu_init = 0.1
opt.var_init = torch.pow(0.075, 2)--torch.sqrt(2/opt.hidden[1])--0.01
opt.pi_init = {
    mu = 5,
    var = 0.00001
}
-- optimisation params
opt.state = {
    lambda = 1-1e-8,
    learningRate = 0.00000001,
}
opt.varState = {
    lambda = 1-1e-8,
    learningRate = 0.0005,
--    learningRateDecay = 0.01
}
opt.meanState = {
    lambda = 1-1e-8,
    learningRate = 0.000001,
--    learningRateDecay = 0.01
}
opt.piState = {
    lambda = 1-1e-8,
    learningRate = 0.00000001,
}
return opt
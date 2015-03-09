local opt = {}

opt = {}
opt.classes = {'0','1','2','3','4','5','6','7','8','9'}
--opt.classes = {'0','1'}
opt.threads = 8
opt.network_to_load = ""
opt.network_name = "vb6kdurk"
opt.type = "vb"
opt.cuda = true
opt.trainSize = 100
opt.testSize = 100

opt.plot = true
opt.batchSize = 1
opt.testBatchSize = 1
opt.B = (opt.trainSize/opt.batchSize)--*100
opt.hidden = {100}
opt.S = 5
opt.testSamples = 5
opt.quicktest = true
opt.log = true
--opt.normcheck = true
--opt.plotlc = true
--opt.viz = true
opt.geometry = {28,28}
opt.input_size = opt.geometry[1]*opt.geometry[2] -- 2283

opt.weight_init = 0.01
--opt.mu_init = 0.1
opt.var_init = 0.01--torch.pow(0.075, 2)--torch.sqrt(2/opt.hidden[1])--0.01
opt.pi_init = {
    mu = 5,
    var = 0.00001
}
-- optimisation params
opt.state = {
--    lambda = 1-1e-8,
    learningRate = 0.001,
}
opt.varState = {
--    lambda = 1-1e-8,
    learningRate = 0.01,
--    learningRateDecay = 0.01
}
opt.meanState = {
--    lambda = 1-1e-8,
    learningRate = 0.001,
--    learningRateDecay = 0.001
}
opt.piState = {
    lambda = 1-1e-8,
    learningRate = 0.00000001,
}
return opt
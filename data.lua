local mnist = require('mnist')
local u = require('utils')

local data = {}

function data.getMnist()
    local trainData = mnist.traindataset()
    local testData = mnist.testdataset()
    trainData = {inputs=trainData.data:type('torch.FloatTensor'), targets=trainData.label}
    u.normalize(trainData.inputs)
    testData = {inputs=testData.data:type('torch.FloatTensor'), targets=testData.label}
    u.normalize(testData.inputs)
    return trainData, testData
end

function data.getBacteria()

end

return data


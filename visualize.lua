--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 11/23/14
-- Time: 12:00 PM
-- To change this template use File | Settings | File Templates.
--

require 'gfx.js'
require 'randomkit'
local viz = {}

--print(torch.DiskFile('logs/logs/train.log', 'r').readString())
function split(string)
    local res = {}
    for i in string.gmatch(string, "%S+") do
        table.insert(res, i)
    end
    return res
end

function viz.show_parameters(weights, vars, hidden)

    local weights = torch.Tensor(W):copy(weights):resize(hidden[1], 28, 28)
    local vars = torch.Tensor(W):copy(vars):resize(hidden[1], 28, 28)

    local meanimgs = {}
    local varimgs = {}
    for i = 1, hidden[1] do
        table.insert(meanimgs, weights[i])
        if vars then
            table.insert(varimgs, vars[i])
        end
    end

    print("weights:min(): ", torch.min(weights))
    print("weights:max(): ", torch.max(weights))


    gfx.image(meanimgs,{zoom=3.5, legend='means', min=-0.36, max=0.30, win='means', refresh=true})
    if vars then
        gfx.image(varimgs,{zoom=3.5, legend='vars', min=0.008, max=0.0255, win='vars', refresh=true})
        print("vars:min(): ", torch.min(vars))
        print("vars:max(): ", torch.max(vars))
    end

    --   gfx.chart(data, {
    --      chart = 'scatter', -- or: bar, stacked, multibar, scatter
    --      width = 600,
    --      height = 450,
    --   })
end

function viz.show_uncertainty(output, sample, means, vars, hidden, label)
    local mresult = torch.Tensor(28,28):zero()
    local vresult = torch.Tensor(28,28):zero()
    for i = 1, hidden do
        mresult:add(torch.mul(means[i], output[i]))
        vresult:add(torch.mul(vars[i], output[i]))
    end
    mresult:mul(1/hidden)
    vresult:mul(1/hidden)
--    print("res:min(): ", torch.min(result))
--    print("res:max(): ", torch.max(result))
    gfx.image({sample, mresult, vresult}, {zoom=3.5, legend=label})
end

function viz.show_uncertainties(model, parameters, testData, means, vars, hidden)
    parameters:copy(means)
    local nr_samples = 10
    local samples = torch.Tensor(nr_samples)
    randomkit.randint(samples, 1, testData.inputs:size(1))
    for i =1 , nr_samples do
        local input = testData.inputs[samples[i]]

        local out = model:forward(input)
        _, index = out:max(1)
        local pred = index[1]-1
        local correct = testData.targets[samples[i]]
        print("prediction: ", pred)
        print("correct: ", correct)
        local label = "p:"..pred.." c:"..correct
        local vars = torch.Tensor(W):copy(vars):resize(hidden[1], 28, 28)
        local means = torch.Tensor(W):copy(means):resize(hidden[1], 28, 28)
        viz.show_uncertainty(model:get(2).output, input, means, vars, hidden[1], label)
    end

end

--file = io.open('logs/logs/test.log', "r")
--data = torch.Tensor(split(file:read("*all")))
--gfx.chart(data)
return viz
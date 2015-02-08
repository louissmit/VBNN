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

function viz.show_tensors(tensors, dim, id)
    local tensor_imgs = {}
    for i = 1, dim do
        table.insert(tensor_imgs, tensors[i])
    end
    gfx.image(tensor_imgs,{zoom=3.5, legend=id, win=id})
end

function viz.show_input_parameters(params, size, id, opt)
    local weights = torch.Tensor(size):copy(params):resize(opt.hidden[1], 28, 28)
    if cuda then
        weights = weights:float()
    end
    viz.show_tensors(weights, opt.hidden[1], id)
    print("weights:min(): ", torch.min(weights))
    print("weights:max(): ", torch.max(weights))
end

function viz.show_images(set, indices, id)
    local img_size =784

    local tensor_result = torch.Tensor(#indices*img_size):zero()
    local prev = torch.Tensor(img_size):zero()
    local result = {}
    for k, i in pairs(indices) do
--        print(set.inputs:select(1,i):mean())
--        print(set.inputs:select(1,i):var())
        local img_tens = set.inputs:select(1,i)

--        img_tens:add(-img_tens:min())
--        img_tens:div(img_tens:max())
--        print(img_tens:min(), img_tens:max())
--        img_tens:resize(img_size)
        local sub = tensor_result:narrow(1, (k-1)*img_size+1, img_size)
        sub:copy(img_tens)
        table.insert(result, img_tens)
        prev = img_tens
    end
--    viz.show_tensors(result, #indices, id)
    return tensor_result
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

function viz.graph_things(opt, name)
    local graphthing = {}
    graphthing.name = name
    graphthing.logger = optim.Logger(paths.concat(opt.network_name, name))
    function graphthing:add(value)
        self.logger:add({[self.name] = value})
    end
    function graphthing:plot()
        self.logger:style{[self.name] = '-' }
        self.logger:plot()
    end
    return graphthing
end

--file = io.open('logs/logs/test.log', "r")
--data = torch.Tensor(split(file:read("*all")))
--gfx.chart(data)
return viz
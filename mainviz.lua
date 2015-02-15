--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 13/02/15
-- Time: 11:12
-- To change this template use File | Settings | File Templates.
--

local viz = require('visualize')

local model_dir = 'vbloooooooong'
local dir = paths.concat(model_dir, 'parameters')
local means = torch.load(paths.concat(dir, 'means'))
local lvars = torch.load(paths.concat(dir, 'lvars'))
local opt = torch.load(paths.concat(model_dir, 'opt'))
local vars = torch.exp(lvars)

--viz.show_input_parameters(means, means:size(), 'means', opt)
--viz.show_input_parameters(vars, vars:size(), 'vars', opt)
local pruned = torch.abs(torch.cdiv(means, torch.sqrt(vars)))
pruned = torch.lt(pruned, 0.005):float()
print(pruned:sum())
print(vars:size(1))
local prunedvars = torch.cmul(pruned, vars)

print(vars:mean())
print(prunedvars:mean())
--viz.show_input_parameters(pruned, pruned:size(), 'pruned', opt)

--    viz.show_uncertainties(model, parameters, testData, beta.means, beta.vars, opt.hidden)
--    local init = viz.show_images(trainData, {1,2,3,13,15,16,17,18,19,20})

--    viz.show_input_parameters(parameters, parameters:size(), opt)
--    model:forward(trainData.inputs[5])
--    print(model:get(2).output)
--    viz.show_tensors(model:get(2).output, opt.hidden[1], 'activations')
--    break

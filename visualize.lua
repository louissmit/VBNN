--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 11/23/14
-- Time: 12:00 PM
-- To change this template use File | Settings | File Templates.
--

require 'gfx.js'
local viz = {}

--print(torch.DiskFile('logs/logs/train.log', 'r').readString())
function split(string)
    local res = {}
    for i in string.gmatch(string, "%S+") do
        table.insert(res, i)
    end
    return res
end

function viz.show_parameters(weights, vars)

    local meanimgs = {}
    local varimgs = {}
    for i = 1, opt.hidden do
        table.insert(meanimgs, weights[i])
        if vars then
            table.insert(varimgs, vars[i])
        end
    end

    print("weights:min(): ", torch.min(weights))
    print("weights:max(): ", torch.max(weights))


    gfx.image(meanimgs,{zoom=3.5, legend='means', min=-0.12, max=0.12, win='means', refresh=true})
    if vars then
        gfx.image(varimgs,{zoom=3.5, legend='vars', min=0.01, max=0.0255, win='vars', refresh=true})
        print("vars:min(): ", torch.min(vars))
        print("vars:max(): ", torch.max(vars))
    end

    --   gfx.chart(data, {
    --      chart = 'scatter', -- or: bar, stacked, multibar, scatter
    --      width = 600,
    --      height = 450,
    --   })
end

--file = io.open('logs/logs/test.log', "r")
--data = torch.Tensor(split(file:read("*all")))
--gfx.chart(data)
return viz
--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 11/23/14
-- Time: 12:00 PM
-- To change this template use File | Settings | File Templates.
--

require 'gfx.js'

--print(torch.DiskFile('logs/logs/train.log', 'r').readString())
function split(string)
    local res = {}
    for i in string.gmatch(string, "%S+") do
        table.insert(res, i)
    end
    return res
end

file = io.open('logs/logs/test.log', "r")
data = torch.Tensor(split(file:read("*all")))
gfx.chart(data)
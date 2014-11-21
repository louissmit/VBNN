--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 11/21/14
-- Time: 1:34 PM
-- To change this template use File | Settings | File Templates.
--
require 'torch'

local VBparams = {}
torch.setdefaulttensortype('torch.FloatTensor')

function VBparams:init(params)
    self.vars = torch.Tensor():typeAs(params):resizeAs(params):fill(0.075)
    self.means = torch.Tensor():typeAs(params):resizeAs(params):zero()
    self.W = params:size()
    return self
end

function VBparams:sampleW()
    local w = torch.Tensor(self.W)
    return w:map2(self.means, self.vars, function(_, mean, var)
        return torch.normal(mean, torch.sqrt(var))
    end)
end


return VBparams
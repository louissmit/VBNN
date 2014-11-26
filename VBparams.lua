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
    self.W = params:size(1)
    self.vars = torch.Tensor(W):fill(0.005625)
    self.means = torch.Tensor(W):apply(function(_)
        return torch.normal(0, 0.1)
    end)
    return self
end

function VBparams:sampleW()
    return torch.Tensor(W):map2(self.means, self.vars, function(_, mean, var)
        return torch.normal(mean, torch.sqrt(var))
    end)
end


return VBparams
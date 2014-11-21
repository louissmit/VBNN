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
    self.stdv = torch.Tensor(params:size()):fill(0.075)
    self.means = torch.Tensor(params:size()):zero()
    return self
end

function VBparams:sampleW()
    local w = torch.Tensor(self.means:size())
    return w:map2(self.means, self.stdv, function(_, mean, st)
        return torch.normal(mean, st)
    end)
end


return VBparams
--
-- Created by IntelliJ IDEA.
-- User: louissmit
-- Date: 11/21/14
-- Time: 12:04 AM
-- To change this template use File | Settings | File Templates.
--
local VBCriterion, parent = torch.class('nn.VBCriterion', 'nn.Criterion')

function VBCriterion:__init(weights)
    parent.__init(self)
    self.sizeAverage = true
    if weights then
        assert(weights:dim() == 1, "weights input should be 1-D Tensor")
        self.weights = weights
    end
end

function VBCriterion:updateOutput(input, target)
    if input:dim() == 1 then
        self.output = -input[target]
        if self.weights then
            self.output = self.output*self.weights[target]
        end
    elseif input:dim() == 2 then
        local output = 0
        for i=1,target:size(1) do
            if self.weights then
                output = output - input[i][target[i]]*self.weights[target[i]]
            else
                output = output - input[i][target[i]]
            end
        end
        if self.sizeAverage then
            output = output / target:size(1)
        end
        self.output = output
    else
        error('matrix or vector expected')
    end
    return self.output
end

function VBCriterion:updateGradInput(input, target)
    self.gradInput:resizeAs(input)
    self.gradInput:zero()

    if input:dim() == 1 then
        self.gradInput[target] = -1
        if self.weights then
            self.gradInput[target] = self.gradInput[target]*self.weights[target]
        end
    else
        local z = -1
        if self.sizeAverage then
            z = z / target:size(1)
        end
        for i=1,target:size(1) do
            self.gradInput[i][target[i]] = z
            if self.weights then
                self.gradInput[i][target[i]] = self.gradInput[i][target[i]]*self.weights[target[i]]
            end
        end
    end
    return self.gradInput
end


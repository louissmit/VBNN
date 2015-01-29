--[[ An implementation of Adam

ARGS:

- 'opfunc' : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- 'x'      : the initial point
- 'config` : a table with configuration parameters for the optimizer
- 'config.learningRate'      : learning rate
- 'config.learningRateDecay' : learning rate decay
- 'config.weightDecay'       : weight decay
- 'config.momentum'          : momentum
- 'config.dampening'         : dampening for momentum
- 'config.nesterov'          : enables Nesterov momentum
- 'state'                    : a table describing the state of the optimizer; after each
                              call the state is modified
- 'state.rms'                 : vector of individual learning rates

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]

function adam(opfunc, x, config, state)
    -- get parameters
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 2e-6
    local gamma = config.updateDecay or 0.9
    local beta = config.momentumDecay or 1

    local beta1 = 0.1
    local beta2 = 0.001
    local epsilon = 10e-8
    local lambda = 10e-8

    local fx, f_prime = opfunc(x)

    state.t= state.t or 1
    state.m = state.m or torch.Tensor():typeAs(f_prime):resizeAs(f_prime):fill(0)
    state.v = state.v or torch.Tensor():typeAs(f_prime):resizeAs(f_prime):fill(0)

    local bt1 = 1 - (1-beta1)*torch.pow(lambda,state.t-1)
    state.m = torch.add(torch.mul(f_prime, bt1), torch.mul(state.m, 1-bt1))
    state.v = torch.add(torch.mul(torch.pow(f_prime, 2), beta2), torch.mul(state.v, 1-beta2))

    local update = torch.cmul(state.m, torch.pow(torch.add(torch.pow(state.v, 2), epsilon),-1))
    update:mul(lr * torch.sqrt(1-torch.pow((1-beta2),2)) * torch.pow(1-torch.pow((1-beta1),2), -1))
--    print("update: ", torch.norm(update))
--    print("x:", torch.norm(x))
--    if opt.normcheck then
--        print("norm_ratio: ", torch.norm(update)/torch.norm(x))
--    end
    x:add(-update)
    state.t = state.t + 1

    -- return x*, f(x) before optimization
    return x,{fx}, update
end

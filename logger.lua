local u = require('utils')
local inspect = require('inspect')
local logger = {}

function logger:init(dir, append)
    os.execute('mkdir ' .. dir)
    self.dir = dir
    self.loggers = {}
    self.append = append
--        assert(not(u.file_exists(paths.concat(dir, id))), 'File already exists!')
    return self
end

function logger:_create(id, type)
    self.loggers[id] = io.open(paths.concat(self.dir, id), type)
end

function logger:add(id, value)
    if not self.loggers[id] then
        if not self.append then
            io.open(paths.concat(self.dir, id), 'w'):write(""):close()
        end
        self:_create(id, 'a')
    end
    self.loggers[id]:write(value..'\n')
end

function logger:append(id, value)
    if not self.loggers[id] or not self.append then
        self:_create(id, 'a+')
        self.append = true
    end
    self:add(id, value)
end

function logger:flush()
    for _, logger in pairs(self.loggers) do
        logger:flush()
    end
end

function logger:close()
    for _, logger in pairs(self.loggers) do
        logger:close()
    end
end

return logger
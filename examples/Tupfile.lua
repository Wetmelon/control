
for k, v in ipairs(INCLUDES) do
    INCLUDES[k] = '-I'..ROOT..'/'..v
end

-- Compile control library source files
libcontrol = tup.foreach_rule(ROOT..'/source/*.cpp', '^j^'..CC_PATH..'g++ $(INCLUDES) '..CXXFLAGS..' '..WARNINGS..' -c %f -o %o', 'build/obj/%B.o')

-- Compile all example source files
examples = tup.foreach_rule('*.cpp', '^j^'..CC_PATH..'g++ $(INCLUDES) '..CXXFLAGS..' '..WARNINGS..' -c %f -o %o', 'build/obj/%B.o')
examples.extra_inputs = {ROOT..'/<libmatplot>'}

-- Add control library object files as extra inputs to example builds and prepare link line
lib = ''
for k, v in ipairs(libcontrol) do
    table.insert(examples.extra_inputs, v)
    lib = lib..' '..v
end

-- Link example executables
tup.foreach_rule(examples, 'g++ '..LDFLAGS..' %f %<libmatplot> '..lib..' -lstdc++exp -lgdi32 -o %o', 'build/%B.exe')

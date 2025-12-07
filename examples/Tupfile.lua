
for k, v in ipairs(INCLUDES) do
    INCLUDES[k] = '-I'..ROOT..'/'..v
end

examples += tup.foreach_rule('*.cpp', '^j^'..CC_PATH..'g++ $(INCLUDES) '..COMMON_FLAGS..' '..CXXFLAGS..' '..WARNINGS..' -c %f -o %o', 'build/obj/%B.o')
libs = {ROOT..'/source/libcontrol.a', ROOT..'/libs/libmatplot.a'}
examples.extra_inputs = libs

-- Generate executable for this example
tup.foreach_rule(examples, 'g++ '..LDFLAGS..'%f $(libs)  -lstdc++exp -lgdi32 -o %o', 'build/%B.exe')

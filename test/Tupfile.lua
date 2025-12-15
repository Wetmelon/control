
for k, v in ipairs(INCLUDES) do
    INCLUDES[k] = '-I'..ROOT..'/'..v
end

tests += tup.foreach_rule('*.cpp', '^j^'..CC_PATH..'g++ $(INCLUDES) '..COMMON_FLAGS..' '..CXXFLAGS..' '..WARNINGS..' -c %f -o %o', 'build/obj/%B.o')
libs = {ROOT..'/source/libcontrol.a', ROOT..'/libs/libmatplot.a'}
tests.extra_inputs = libs

-- Generate test executables
tup.foreach_rule(tests, 'g++ '..LDFLAGS..'%f $(libs) -lstdc++exp -lgdi32 -o %o', 'build/%B.exe')


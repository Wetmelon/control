for k, v in ipairs(INCLUDES) do
    INCLUDES[k] = '-I'..ROOT..'/'..v
end

-- Compile all test source files
tests = tup.foreach_rule('*.cpp', '^j^'..CC_PATH..'g++ $(INCLUDES) '..CXXFLAGS..' '..WARNINGS..' -c %f -o %o', 'build/obj/%B.o')
tests.extra_inputs = { ROOT..'/<libcontrol>', ROOT..'/<libfmt>' }

-- Link test executables
tup.rule(tests, 'g++ '..LDFLAGS..' %f %<libcontrol> %<libfmt> -lstdc++exp -lgdi32 -o %o', 'build/test_runner.exe')
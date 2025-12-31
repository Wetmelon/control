for k, v in ipairs(INCLUDES) do
    INCLUDES[k] = '-I'..ROOT..'/'..v
end

-- Compile all example source files
examples = tup.foreach_rule('*.cpp', '^j^'..CC_PATH..'g++ $(INCLUDES) '..CXXFLAGS..' '..WARNINGS..' -c %f -o %o', 'build/obj/%B.o')
examples.extra_inputs = {ROOT..'/<libmatplot>', ROOT..'/<libcontrol>', ROOT..'/<libfmt>'}

-- Link example executables
tup.foreach_rule(examples, '^o^ g++ '..LDFLAGS..' %f %<libmatplot> %<libcontrol> %<libfmt> -lstdc++exp -lgdi32 -o %o', 'build/%B.exe')
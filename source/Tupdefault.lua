
for k, v in ipairs(INCLUDES) do
    INCLUDES[k] = '-I'..ROOT..'/'..v
end

-- Compile control library source files
objs = tup.foreach_rule(ROOT..'/source/*.cpp', '^j^'..CC_PATH..'g++ $(INCLUDES) '..CXXFLAGS..' '..WARNINGS..' -c %f -o %o', {'build/obj/%B.o', ROOT..'/<libcontrol>'})


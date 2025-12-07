
for k, v in ipairs(INCLUDES) do
    INCLUDES[k] = '-I'..ROOT..'/'..v
end

-- Compile all objects
objs = tup.foreach_rule('*.c', '^j^'..CC_PATH..'gcc $(INCLUDES) '..COMMON_FLAGS..' '..CCFLAGS..' '..WARNINGS..' -c %f -o %o', 'build/obj/%B.o')
objs += tup.foreach_rule('*.cpp', '^j^'..CC_PATH..'g++ $(INCLUDES) '..COMMON_FLAGS..' '..CXXFLAGS..' '..WARNINGS..' -c %f -o %o', 'build/obj/%B.o')

-- Generate assembly for each object
tup.foreach_rule(objs, CC_PATH..'objdump -dC %f > %o', 'build/asm/%B.asm')

--- Create static library
tup.rule(objs, CC_PATH..'ar rcs %o %f', 'libcontrol.a')
INCLUDES = "-I. -Isource -Ieigen"
WARNINGS = '-Wall -Wextra'

COMMON_FLAGS = '-O2'
CCFLAGS = '-std=c17'
CXXFLAGS = '-std=c++23'
LDFLAGS = '-Wl,--gc-sections -static'

CC_PATH = tup.getconfig("COMPILER_PATH")
if CC_PATH ~= "" then
    print("Compiler Path: "..CC_PATH)
end

if tup.getconfig('LTO') == 'true' then
    COMMON_FLAGS += ' -flto'
    LDFLAGS += ' -flto'
end

-- Compile all objects
objs = tup.foreach_rule('source/*.c', '^j^'..CC_PATH..'gcc '..INCLUDES..' '..COMMON_FLAGS..' '..CCFLAGS..' '..WARNINGS..' -c %f -o %o', 'build/obj/%B.o')
objs = tup.foreach_rule('source/*.cpp', '^j^'..CC_PATH..'g++ '..INCLUDES..' '..COMMON_FLAGS..' '..CXXFLAGS..' '..WARNINGS..' -c %f -o %o', 'build/obj/%B.o')

-- Generate assembly for each object
tup.foreach_rule(objs, CC_PATH..'objdump -dC %f > %o', 'build/asm/%B.asm')

-- Generate test executable
test_runner = tup.rule(objs, 'g++ '..LDFLAGS..' %f -lstdc++exp -o %o', 'build/test_runner.exe')


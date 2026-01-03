
INCLUDES = '-I.'
INCLUDES += '-I../inc'
INCLUDES += '-I../fmt'

-- Compile all fmt source files
fmt_objs = tup.foreach_rule('../fmt/*.cc', '^j^'..CXX..' '..CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/fmt/%B.o')

-- Compile all .cpp files in the example directory
ex_objs = tup.foreach_rule('*.cpp', '^j^'..CXX..' '..CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/%B.o')
ex_objs.extra_inputs = fmt_objs

-- tup.foreach_rule(ex_objs, 'objdump -dsC %f > build/%B.asm', 'build/%B.asm')

-- Link with g++
examples = tup.foreach_rule(ex_objs, CXX..' '..CXXFLAGS..' '..LDFLAGS..' %f $(fmt_objs) -o %o', 'build/%B.exe')

-- Run test executables
-- tup.frule{inputs = examples, command = './%f', outputs = {}}

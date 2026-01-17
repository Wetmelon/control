
INCLUDES = '-I.'
INCLUDES += '-I../inc'
INCLUDES += '-I../fmt'

-- Compile all .cpp files in the tests directory
objs = tup.foreach_rule('*.cpp', '^j^'..CXX..' '..CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/%B.o')

-- Compile all fmt source files
objs += tup.foreach_rule('../fmt/*.cc', '^j^'..CXX..' '..CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/fmt/%B.o')

-- Link with g++
tup.rule(objs, CXX..' '..CXXFLAGS..' '..LDFLAGS..' %f -o %o', 'build/test_runner.exe')

-- Run test executable
-- tup.frule{inputs = 'build/test_runner.exe', command = './%f', outputs = {}}


INCLUDES = '-I.'
INCLUDES += '-I../inc'
INCLUDES += '-I../inc/matrix'
INCLUDES += '-I../libs'
INCLUDES += '-I../libs/fmt/include'
INCLUDES += '-I../libs/plotlypp/include'
INCLUDES += '-I../libs/json/single_include'

-- Compile all .cpp files in the tests directory
objs = tup.foreach_rule('*.cpp', '^j^'..CXX..' '..CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/%B.o')

-- Compile all fmt source files
fmt_sources = {'../libs/fmt/src/format.cc', '../libs/fmt/src/os.cc'}
objs += tup.foreach_rule(fmt_sources, '^j^'..CXX..' '..CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/fmt/%B.o')

-- Link with g++
tup.rule(objs, CXX..' '..CXXFLAGS..' '..LDFLAGS..' %f -o %o', 'build/test_runner.exe')

-- Run test executable
-- tup.frule{inputs = 'build/test_runner.exe', command = './%f', outputs = {}}

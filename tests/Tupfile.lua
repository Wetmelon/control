
CXX = 'C:/Tools/gcc-14.2.0-2/bin/g++'

INCLUDES = '-I.'
INCLUDES += '-I../inc'
INCLUDES += '-I../fmt'

WARNINGS = '-Wall -Wextra -Wdouble-promotion'
CXXFLAGS = '-O3 -std=c++20 -march=native -ffunction-sections -fdata-sections'
LDFLAGS = '-O3 -march=native -ffunction-sections -fdata-sections -Wl,--gc-sections'

-- Compile all .cpp files in the tests directory
objs = tup.foreach_rule('*.cpp', '^j^'..CXX..' '..CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/%B.o')

-- Compile all fmt source files
objs += tup.foreach_rule('../fmt/*.cc', '^j^'..CXX..' '..CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/fmt/%B.o')

-- Link with g++
tup.rule(objs, CXX..' '..CXXFLAGS..' '..LDFLAGS..' %f -o %o', 'build/test_runner.exe')

-- Run test executable
-- tup.frule{inputs = 'build/test_runner.exe', command = './%f', outputs = {}}


INCLUDES = '-I.'
INCLUDES += '-I../inc'
INCLUDES += '-I../libs'
INCLUDES += '-I../libs/fmt/include'
INCLUDES += '-I../libs/plotlypp/include'
INCLUDES += '-I../libs/json/single_include'

-- The test runner builds with -ffast-math by default. Constexpr static_asserts
-- across the library act as a tripwire for the property that compile-time
-- evaluation stays IEEE-strict even when the optimizer is set to fast-math. If
-- a future toolchain leaks fast-math into constant evaluation, the suite will
-- fail on the next build. Drop the flag here to spot-check strict-IEEE runtime
-- behavior, or lower -O for precision investigations.
TEST_CXXFLAGS = CXXFLAGS..' -ffast-math'

-- Compile all .cpp files in the tests directory
objs = tup.foreach_rule('*.cpp', '^j^'..CXX..' '..TEST_CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/objs/%B.o')

-- Compile all fmt source files
fmt_sources = {'../libs/fmt/src/format.cc', '../libs/fmt/src/os.cc'}
objs += tup.foreach_rule(fmt_sources, '^j^'..CXX..' '..TEST_CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/objs/fmt/%B.o')

-- Link with g++
tup.rule(objs, CXX..' '..TEST_CXXFLAGS..' '..LDFLAGS..' -static %f -o %o', 'build/test_runner.exe')

-- Run test executable
-- tup.frule{inputs = 'build/test_runner.exe', command = './%f', outputs = {}}

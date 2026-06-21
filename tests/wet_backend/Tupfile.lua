-- Standalone executable for the WET_MATH_BACKEND_WET profile. Kept out of the
-- main tests/ glob (and its single test_runner.exe) because the wet
-- MathBackend<float> specialization is ODR-incompatible with the default std
-- backend every other test object links against. tup.foreach_rule globs are
-- per-directory, so the parent tests/Tupfile.lua never picks this file up.
INCLUDES = '-I. -I.. -I../../inc'

TEST_CXXFLAGS = CXXFLAGS..' -ffast-math'

obj = tup.foreach_rule('*.cpp', '^j^'..CXX..' '..TEST_CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/objs/%B.o')
tup.rule(obj, CXX..' '..TEST_CXXFLAGS..' '..LDFLAGS..' -static %f -o %o', 'build/test_wet_backend.exe')

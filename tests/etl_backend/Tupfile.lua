-- Standalone executable for the WET_BACKEND_ETL profile (ETL container backend).
-- Kept out of the main tests/ glob and its single test_runner.exe because the
-- wet:: aliases resolve to etl:: types here, ODR-incompatible with the std types
-- every other test object links against. tup.foreach_rule globs are per-directory,
-- so the parent tests/Tupfile.lua never picks this file up.
--
-- -I. puts this directory's wet_profile.hpp ahead of tests/wet_profile.hpp, so
-- wet/config.hpp selects ETL. The ETL headers live under libs/etl/include.
INCLUDES = '-I. -I.. -I../../inc -I../../libs/etl/include'

TEST_CXXFLAGS = CXXFLAGS..' -ffast-math'

obj = tup.foreach_rule('*.cpp', '^j^'..CXX..' '..TEST_CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -c %f -o %o', 'build/objs/%B.o')
tup.rule(obj, CXX..' '..TEST_CXXFLAGS..' '..LDFLAGS..' -static %f -o %o', 'build/test_etl_backend.exe')

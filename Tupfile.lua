INCLUDES = "-I. -Isource -Ieigen"
WARNINGS = '-Wall -Wextra'

-- Compile all objects
test_objs = tup.foreach_rule('source/*.cpp', '^j^g++ '..INCLUDES..' -O3 '..WARNINGS..' -std=c++23 -c %f -o %o', 'build/obj/%B.o')

-- Generate test executable
test_runner = tup.rule(test_objs, 'g++ %f -lstdc++exp -o %o', 'build/test_runner.exe')

-- Generate assembly for each object
tup.foreach_rule(test_objs, 'objdump -dSC %f > %o', 'build/asm/%B.asm')

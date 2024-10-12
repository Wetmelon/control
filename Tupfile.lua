
INCLUDES = "-I. -Isource -IEigen"
test_objs = tup.foreach_rule('source/*.cpp', '^j^g++ '..INCLUDES..' -O3 -std=c++23 -c %f -o %o', 'build/obj/%B.o')
test_runner = tup.rule(test_objs, 'g++ %f -o %o', 'build/test_runner.exe')
tup.frule{inputs = test_objs, command = 'objdump -dSC %f > %o', outputs = 'build/asm/%B.asm'}

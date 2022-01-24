
test_objs = tup.foreach_rule('test/*.cpp', 'g++ -I. -O3 -std=c++20 -c %f -o %o', 'test/build/obj/%B.o')
test_runner = tup.rule(test_objs, 'g++ %f -o %o', 'test/build/test_runner.exe')
tup.rule(test_runner, './%f')

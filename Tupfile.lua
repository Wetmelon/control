
tup.foreach_rule('test/*.cpp', 'g++ -I. -O3 -std=c++17 -c %f -o %o', 'test/build/obj/%B.o')

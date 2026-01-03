.PHONY: all clean test

all:
	@clang-format -i inc/*.hpp tests/*.cpp examples/*.cpp
	@tup --quiet compiledb
	@tup
	@./tests/build/test_runner.exe
	@./examples/build/example_cart_pole.exe
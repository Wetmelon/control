.PHONY: all clean test

all:
	@clang-format -i inc/*.hpp tests/*.cpp
	@tup --quiet compiledb
	@tup
	@./tests/build/test_runner.exe
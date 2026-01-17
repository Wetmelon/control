.PHONY: all clean test docs

all:
	@clang-format -i inc/*.hpp tests/*.cpp examples/*.cpp
	@tup --quiet compiledb
	@tup
	@./tests/build/test_runner.exe

docs:
	@mkdir -p docs/html
	@doxygen Doxyfile

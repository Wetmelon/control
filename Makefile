.PHONY: all compile format test

all: format compile test

compile:
	@tup --quiet compiledb
	@tup --quiet

format:
	@clang-format -i source/*.cpp source/*.hpp examples/*.cpp

test:
	@./test/build/test_lti_operations.exe

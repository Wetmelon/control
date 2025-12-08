.PHONY: all format

all:
	@tup --quiet compiledb
	@tup

	@./examples/build/example_second_order.exe
	@./examples/build/example_solve_lti.exe
	@./examples/build/example_solve_nonlinear.exe

format:
	@clang-format -i source/*.cpp source/*.hpp examples/*.cpp
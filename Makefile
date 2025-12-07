.PHONY: all

all:
	@tup --quiet compiledb
	@tup

	@./examples/build/example_critical_damping.exe
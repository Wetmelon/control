.PHONY: all format compile test examples gui coverage

all: format
	@tup --quiet compiledb
	@tup
	@./test/build/test_runner.exe

format:
	@clang-format -i source/*.hpp source/*.cpp examples/*.cpp test/*.cpp

compile: format
	@tup --quiet compiledb
	@tup

test: format
	@tup --quiet test
	@./test/build/test_runner.exe

examples: format
	@tup --quiet examples

gui: format
	@tup --quiet python
	@py -3 python/odrive_tuning_gui.py
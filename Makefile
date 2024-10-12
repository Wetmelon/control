all:
	@tup --quiet compiledb
	@tup --quiet
	@./build/test_runner.exe
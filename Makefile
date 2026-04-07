.PHONY: all clean tests docs gui

all:
	@clang-format -i inc/*.hpp tests/*.cpp examples/*.cpp
	@tup --quiet compiledb
	@tup
	@./tests/build/test_runner.exe

docs:
	@mkdir -p docs/html
	@doxygen Doxyfile

tests:
	@tup --quiet tests
	@./tests/build/test_runner.exe

gui:
	@cd examples/servo_drive && g++ -std=c++20 -O3 -shared -DBUILD_DLL -static-libgcc -static-libstdc++ -I../../inc -I../../inc/matrix servo_sim.cpp -o servo_sim.dll
	@cd examples/servo_drive && pip install dearpygui plotly
	@cd examples/servo_drive && python servo_gui.py
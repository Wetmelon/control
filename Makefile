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
ifeq ($(OS),Windows_NT)
	@cd examples/servo_drive && g++ -std=c++20 -O3 -shared -DBUILD_DLL -static-libgcc -static-libstdc++ -I../../inc -I../../inc/matrix servo_sim.cpp -o servo_sim.dll
else ifeq ($(shell uname -s),Darwin)
	@cd examples/servo_drive && clang++ -std=c++20 -O3 -shared -fPIC -DBUILD_DLL -I../../inc -I../../inc/matrix servo_sim.cpp -o libservo_sim.dylib
else
	@cd examples/servo_drive && g++ -std=c++20 -O3 -shared -fPIC -DBUILD_DLL -I../../inc -I../../inc/matrix servo_sim.cpp -o libservo_sim.so
endif
	@cd examples/servo_drive && (test -d .venv || python3 -m venv .venv) && . .venv/bin/activate && pip install dearpygui plotly && python servo_gui.py
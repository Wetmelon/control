.PHONY: all clean tests tidy docs gui

# Build compiler, derived from tup.config (mirrors Tuprules.lua: path + prefix + g++).
# tup.config lines are KEY=value, i.e. valid make assignments -- include them
# directly rather than shelling out to sed (which isn't on PATH under PowerShell).
include tup.config
GXX := $(CONFIG_COMPILER_PATH)/$(CONFIG_COMPILER_PREFIX)g++

all:
	@clang-format -i inc/*.hpp inc/matrix/*.hpp tests/*.cpp examples/*.cpp
	@tup --quiet compiledb
	@tup
	@./tests/build/test_runner.exe

docs:
	@mkdir -p docs/html
	@doxygen Doxyfile

tests:
	@tup --quiet tests
	@./tests/build/test_runner.exe

# Run clang-tidy with --fix over all .cpp files (and inc/ headers they pull in).
# Lives here, not in tup: clang-tidy --fix rewrites sources in place, which tup's
# input/output tracking forbids (same reason clang-format -i is a make step).
# clang on Windows targets MSVC by default, so we hand it the build compiler's
# target triple and system include paths or it won't find <cmath> et al.
tidy:
	@tup --quiet compiledb
	@TGT=$$($(GXX) -dumpmachine); \
	ISYS=$$(echo | $(GXX) -std=c++20 -E -x c++ -v - 2>&1 \
		| sed -n '/search starts here:/,/End of search list/p' \
		| grep '^ ' | sed 's,^ ,-extra-arg=-isystem,'); \
	run-clang-tidy -p . -fix -header-filter='inc[/\\].*' \
		-extra-arg=--target=$$TGT $$ISYS '\.cpp$$'

gui:
ifeq ($(OS),Windows_NT)
	@cd examples/servo_drive && g++ -std=c++20 -O3 -shared -DBUILD_DLL -static-libgcc -static-libstdc++ -I../../inc -I../../inc/matrix servo_sim.cpp -o servo_sim.dll
else ifeq ($(shell uname -s),Darwin)
	@cd examples/servo_drive && clang++ -std=c++20 -O3 -shared -fPIC -DBUILD_DLL -I../../inc -I../../inc/matrix servo_sim.cpp -o libservo_sim.dylib
else
	@cd examples/servo_drive && g++ -std=c++20 -O3 -shared -fPIC -DBUILD_DLL -I../../inc -I../../inc/matrix servo_sim.cpp -o libservo_sim.so
endif
	@cd examples/servo_drive && (test -d .venv || python3 -m venv .venv) && . .venv/bin/activate && pip install dearpygui plotly && python servo_gui.py
.PHONY: all clean tests tidy docs gui embedded-check

# Build compiler, derived from tup.config (mirrors Tuprules.lua: path + prefix + g++).
# tup.config lines are KEY=value, i.e. valid make assignments -- include them
# directly rather than shelling out to sed (which isn't on PATH under PowerShell).
include tup.config
GXX := $(CONFIG_COMPILER_PATH)/$(CONFIG_COMPILER_PREFIX)g++

all:
	@clang-format -i $$(find inc -name '*.hpp') $$(find tests examples -name '*.cpp' -o -name '*.hpp')
	@tup --quiet compiledb
	@tup
	@./tests/build/test_runner.exe
	@$(MAKE) --no-print-directory embedded-check

# Guard the embedded contract: nothing reachable from the wet/control.hpp
# umbrella may pull <vector> (or any heap allocation). Fails if it leaks in.
embedded-check:
	@printf '#include "wet/control.hpp"\nint main(){}\n' > .embedded_check.cpp
	@if $(GXX) -std=c++20 -Iinc -M .embedded_check.cpp | grep -q stl_vector.h; then \
		rm -f .embedded_check.cpp; \
		echo "embedded-check FAILED: <vector> is reachable from wet/control.hpp"; exit 1; \
	else \
		rm -f .embedded_check.cpp; \
		echo "embedded-check OK: wet/control.hpp is allocation-free"; \
	fi

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
	@cd examples/servo_drive && g++ -std=c++20 -O3 -shared -DBUILD_DLL -static-libgcc -static-libstdc++ -I../../inc servo_sim.cpp -o servo_sim.dll
else ifeq ($(shell uname -s),Darwin)
	@cd examples/servo_drive && clang++ -std=c++20 -O3 -shared -fPIC -DBUILD_DLL -I../../inc servo_sim.cpp -o libservo_sim.dylib
else
	@cd examples/servo_drive && g++ -std=c++20 -O3 -shared -fPIC -DBUILD_DLL -I../../inc servo_sim.cpp -o libservo_sim.so
endif
	@cd examples/servo_drive && (test -d .venv || python3 -m venv .venv) && . .venv/bin/activate && pip install dearpygui plotly && python servo_gui.py
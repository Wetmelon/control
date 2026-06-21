.PHONY: all clean tests tidy docs gui embedded-check freestanding-check profile-compile

# Build compiler, derived from tup.config (mirrors Tuprules.lua: path + prefix + g++).
# tup.config lines are KEY=value, i.e. valid make assignments -- include them
# directly rather than shelling out to sed (which isn't on PATH under PowerShell).
include tup.config
GXX := $(CONFIG_COMPILER_PATH)/$(CONFIG_COMPILER_PREFIX)g++

all: format build
	@./tests/build/test_runner.exe
	@./tests/wet_backend/build/test_wet_backend.exe
# 	@$(MAKE) --no-print-directory embedded-check
# 	@$(MAKE) --no-print-directory freestanding-check

format:
	@clang-format -i $$(find inc -name '*.hpp') $$(find tests examples -name '*.cpp' -o -name '*.hpp')

compiledb:
	@tup --quiet compiledb

build: format compiledb
	@tup

examples: format compiledb
	@tup examples

tests: format compiledb
	@tup tests
	@./tests/build/test_runner.exe
	@./tests/wet_backend/build/test_wet_backend.exe

docs:
	@mkdir -p docs/html
	@doxygen Doxyfile

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

# Guard the freestanding contract: under the ETL container backend + the
# constexpr-series math backend, the wet/control.hpp umbrella must (1) compile
# and (2) not have any of *our* headers unconditionally pull a hosted-only std
# header. (ETL's own transitive includes are ETL's concern, configured per
# target; an `__has_include`-guarded include is self-disabling and allowed.)
freestanding-check:
	@printf '#define WET_BACKEND_ETL\n#define WET_MATH_BACKEND_FREESTANDING\n#include "wet/control.hpp"\nint main(){}\n' > .fs_check.cpp
	@if ! $(GXX) -std=c++20 -Iinc -Ilibs/etl/include -fsyntax-only .fs_check.cpp 2>.fs_check.log; then \
		cat .fs_check.log; rm -f .fs_check.cpp .fs_check.log; \
		echo "freestanding-check FAILED: umbrella does not compile under the ETL + series backend"; exit 1; \
	fi
	@bad=0; \
	for f in $$($(GXX) -std=c++20 -Iinc -Ilibs/etl/include -M .fs_check.cpp 2>/dev/null | tr ' ' '\n' | grep 'inc/wet/' | sort -u); do \
		case $$f in *['/\\']backend.hpp) continue;; esac; \
		for h in cmath algorithm vector string numbers optional tuple memory functional complex valarray array iostream sstream map set deque list; do \
			if grep -q "#include <$$h>" $$f && ! grep -q "__has_include(<$$h>)" $$f; then \
				echo "  LEAK: $$f unconditionally includes <$$h>"; bad=1; \
			fi; \
		done; \
	done; \
	rm -f .fs_check.cpp .fs_check.log; \
	if [ $$bad -ne 0 ]; then echo "freestanding-check FAILED: hosted headers reachable from wet/control.hpp"; exit 1; fi; \
	echo "freestanding-check OK: wet/control.hpp is freestanding-clean (ETL backend, series math)"




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

# Compile-time profiling. Builds the examples with clang's -ftime-trace (GCC 14.2
# predates -ftime-trace, so we borrow the clang already on PATH for tidy/format),
# emits a per-TU Chrome-trace flame graph under analysis/compile_profile/traces/,
# and prints a frontend-vs-backend / per-header breakdown. Pass FILES=... to scope.
profile-compile:
	@python analysis/compile_profile/profile.py $(FILES)

gui:
ifeq ($(OS),Windows_NT)
	@cd examples/servo_drive && g++ -std=c++20 -O3 -shared -DBUILD_DLL -static-libgcc -static-libstdc++ -I../../inc servo_sim.cpp -o servo_sim.dll
else ifeq ($(shell uname -s),Darwin)
	@cd examples/servo_drive && clang++ -std=c++20 -O3 -shared -fPIC -DBUILD_DLL -I../../inc servo_sim.cpp -o libservo_sim.dylib
else
	@cd examples/servo_drive && g++ -std=c++20 -O3 -shared -fPIC -DBUILD_DLL -I../../inc servo_sim.cpp -o libservo_sim.so
endif
	@cd examples/servo_drive && (test -d .venv || python3 -m venv .venv) && . .venv/bin/activate && pip install dearpygui plotly && python servo_gui.py
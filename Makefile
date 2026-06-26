.PHONY: all clean tests tidy iwyu fix_includes docs gui embedded-check freestanding-check profile-compile

# Build compiler, derived from tup.config (mirrors Tuprules.lua: path + prefix + g++).
# tup.config lines are KEY=value, i.e. valid make assignments -- include them
# directly rather than shelling out to sed (which isn't on PATH under PowerShell).
include tup.config
GXX := $(CONFIG_COMPILER_PATH)/$(CONFIG_COMPILER_PREFIX)g++

all: format build
	@./tests/build/test_runner.exe
	@./tests/wet_backend/build/test_wet_backend.exe
	@./tests/etl_backend/build/test_etl_backend.exe
# 	@$(MAKE) --no-print-directory embedded-check
# 	@$(MAKE) --no-print-directory freestanding-check

format:
	@clang-format -i $$(find inc -name '*.hpp') $$(find tests examples -name '*.cpp' -o -name '*.hpp')

compiledb:
	@tup --quiet compiledb

build: format compiledb
	@tup --quiet

examples: format compiledb
	@tup --quiet examples

tests: format compiledb
	@tup --quiet tests
	@./tests/build/test_runner.exe
	@./tests/wet_backend/build/test_wet_backend.exe
	@./tests/etl_backend/build/test_etl_backend.exe

docs:
	@mkdir -p docs/html
	@doxygen Doxyfile

refs:
	@py -3 tools/gen_reference.py

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

# Run include-what-you-use over every TU in compile_commands.json. The headers
# are already annotated with IWYU pragmas (keep/export); this drives the tool.
#
# IWYU is clang-based but the project builds with mingw-GCC, so we point clang's
# driver at the GCC toolchain (--gcc-toolchain) to analyze against the SAME
# libstdc++ the real build uses -- otherwise on Windows clang defaults to the
# MSVC STL and the include suggestions are wrong. The conda-forge IWYU package
# ships no clang resource headers (mm_malloc.h / intrinsics) and bakes a dead
# -resource-dir, so we borrow the system LLVM's resource dir instead.
#
# Needs: IWYU on PATH (conda-forge include-what-you-use) and a system clang
# (the one already used for tidy/format) for -print-resource-dir.
# Scope to tests/*.cpp: those TUs transitively pull all of inc/, so they cover
# the library, while keeping examples/'s third-party TUs (fmt/json/plotlypp) out
# of the sweep. IWYU reports each TU's filename relative to its compile dir
# (tests/), which is why fix_includes runs with `-p tests` below.
GCC_ROOT := $(dir $(CONFIG_COMPILER_PATH))
IWYU_TUS := $(wildcard tests/*.cpp)
iwyu:
	@tup --quiet compiledb
	@export PYTHONUTF8=1; TGT=$$($(GXX) -dumpmachine); \
	RESDIR=$$(clang -print-resource-dir); \
	iwyu_tool.py -j 0 -p . $(IWYU_TUS) -- --target=$$TGT --gcc-toolchain="$(GCC_ROOT)" \
		-resource-dir="$$RESDIR" -Xiwyu --no_fwd_decls -Xiwyu --mapping_file=$(CURDIR)/iwyu.imp

# Same IWYU sweep, but apply the suggestions: pipe IWYU's output straight into
# fix_includes.py (in place, like clang-tidy --fix -- a make step, not tup).
# --noreorder leaves include ordering to clang-format; `format` runs after to
# sort + align the edits.
fix_includes:
	@tup --quiet compiledb
	@export PYTHONUTF8=1; TGT=$$($(GXX) -dumpmachine); \
	RESDIR=$$(clang -print-resource-dir); \
	iwyu_tool.py -j 0 -p . $(IWYU_TUS) -- --target=$$TGT --gcc-toolchain="$(GCC_ROOT)" \
		-resource-dir="$$RESDIR" -Xiwyu --no_fwd_decls -Xiwyu --mapping_file=$(CURDIR)/iwyu.imp \
		| fix_includes.py --noreorder -p tests
	@$(MAKE) --no-print-directory format

# Compile-time profiling. Builds the examples with clang's -ftime-trace (GCC 14.2
# predates -ftime-trace, so we borrow the clang already on PATH for tidy/format),
# emits a per-TU Chrome-trace flame graph under analysis/compile_profile/traces/,
# and prints a frontend-vs-backend / per-header breakdown. Pass FILES=... to scope.
profile-compile:
	@python analysis/compile_profile/profile.py $(FILES)

# Windows venvs live under Scripts/, POSIX under bin/; call the venv python by
# path rather than sourcing activate (no cross-platform activate script). PYTHON
# is the bootstrap interpreter (override with `make gui PYTHON=...` if needed).
ifeq ($(OS),Windows_NT)
PYTHON ?= python
VENV_PY := .venv/Scripts/python.exe
else
PYTHON ?= python3
VENV_PY := .venv/bin/python
endif

gui:
ifeq ($(OS),Windows_NT)
	@cd examples/servo_drive && g++ -std=c++20 -O3 -shared -DBUILD_DLL -static-libgcc -static-libstdc++ -I../../inc servo_sim.cpp -o servo_sim.dll
else ifeq ($(shell uname -s),Darwin)
	@cd examples/servo_drive && clang++ -std=c++20 -O3 -shared -fPIC -DBUILD_DLL -I../../inc servo_sim.cpp -o libservo_sim.dylib
else
	@cd examples/servo_drive && g++ -std=c++20 -O3 -shared -fPIC -DBUILD_DLL -I../../inc servo_sim.cpp -o libservo_sim.so
endif
	@cd examples/servo_drive && (test -d .venv || $(PYTHON) -m venv .venv) && $(VENV_PY) -m pip install dearpygui && $(VENV_PY) servo_gui.py
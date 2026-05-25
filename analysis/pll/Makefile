CXX = g++
# -static so the report binary is self-contained: without it the exe picks up
# whichever libstdc++/libgcc DLL is first on PATH at runtime, which can fail with
# STATUS_ENTRYPOINT_NOT_FOUND (0xC0000139). The main tup build links -static too.
CXXFLAGS = -O3 -std=c++20 -static -Wall -Wextra -Wdouble-promotion
INCLUDES = -I../../inc -I../../libs/fmt/include -I../../libs/plotlypp/include -I../../libs/json/single_include

OUT_DIR = build
BIN = $(OUT_DIR)/pll_report.exe
SRC = generate_pll_report.cpp pll_report_text.cpp pll_report_plots.cpp
REPORT = pll_metrics_snippet.md

.PHONY: all report clean

all: report
	@clang-format -i *.hpp *.cpp

$(OUT_DIR):
	@mkdir -p $(OUT_DIR)

$(BIN): $(SRC) | $(OUT_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $@

report: $(BIN)
	./$(BIN)

pyreport:
	py -3 pll_report_compare.py

clean:
	rm -rf $(OUT_DIR) $(REPORT) pll_report.html
	rm -rf $(OUT_DIR) plots
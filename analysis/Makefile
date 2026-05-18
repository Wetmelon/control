
CXXFLAGS = -Ofast -std=c++20 -mfloat-abi=hard -mfpu=fpv5-d16 -mcpu=cortex-m7
CXX      = arm-none-eabi-g++

SRCS = ti_arm_trig wet_trig math_utils

.PHONY: all clean

all: clean $(addprefix build/,$(addsuffix .asm,$(SRCS)))

clean:
	@rm -rf build

build/%.asm: %.cpp %.hpp | build
# 	@$(CXX) $(CXXFLAGS) -S $< -o build/$*.s
	@$(CXX) $(CXXFLAGS) -c $< -o build/$*.o
	@arm-none-eabi-objdump build/$*.o -dC > $@

build:
	@mkdir -p build

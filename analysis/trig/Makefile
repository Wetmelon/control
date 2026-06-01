CXXFLAGS = -O3 -ffast-math -std=c++20 -mfloat-abi=hard -mfpu=fpv5-d16 -mcpu=cortex-m7 -Wno-psabi -I../../inc

# GCC toolchain — also used by clang for headers/runtime.
# Windows-style path (C:/...) so it works identically whether passed through
# Git Bash, MSYS, or native cmd.  MSYS-style /c/... paths get mangled by
# path conversion when handed to native Windows binaries like clang.exe.
# TOOLCHAIN_VER  = 10.3.1
TOOLCHAIN_VER  = 14.2.1
TOOLCHAIN_ROOT = C:/Tools/arm-none-eabi-gcc-$(TOOLCHAIN_VER)
TOOLCHAIN_BIN  = $(TOOLCHAIN_ROOT)/bin
TOOLCHAIN_INC  = $(TOOLCHAIN_ROOT)/arm-none-eabi/include/c++/$(TOOLCHAIN_VER)

GCC      = $(TOOLCHAIN_BIN)/arm-none-eabi-g++
CLANG    = clang++
OBJDUMP  = $(TOOLCHAIN_BIN)/arm-none-eabi-objdump

CLANGFLAGS = --target=arm-none-eabi \
             --sysroot=$(TOOLCHAIN_ROOT)/arm-none-eabi \
             -isystem $(TOOLCHAIN_INC) \
             -isystem $(TOOLCHAIN_INC)/arm-none-eabi \
             -Wno-invalid-constexpr

SRCS = math_utils ti_arm_trig

.PHONY: all clean gcc clang

all: clean gcc clang

gcc:   $(addprefix build/gcc/,$(addsuffix .asm,$(SRCS)))
clang: $(addprefix build/clang/,$(addsuffix .asm,$(SRCS)))

build/gcc/%.asm: %.cpp | build/gcc
	@$(GCC) $(CXXFLAGS) -c $< -o build/gcc/$*.o
	@$(OBJDUMP) build/gcc/$*.o -dC > $@

build/clang/%.asm: %.cpp | build/clang
	@$(CLANG) $(CLANGFLAGS) $(CXXFLAGS) -c $< -o build/clang/$*.o
	@llvm-objdump --mcpu=cortex-m7 build/clang/$*.o -dC > $@

build/gcc build/clang:
	@mkdir -p $@

clean:
	@rm -rf build

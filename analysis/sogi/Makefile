
.PHONY: all

all:
	@mkdir -p build
	@arm-none-eabi-g++ -Wno-abi -Ofast -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16 -std=c++20 -S sogi_gen.cpp -I../../inc -I. -o build/sogi_gen.s
	@arm-none-eabi-g++ -Wno-abi -Ofast -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16 -std=c++20 -c sogi_gen.cpp -I../../inc -I. -o build/sogi_gen.o
	@arm-none-eabi-objdump -dC build/sogi_gen.o > build/sogi_gen.asm
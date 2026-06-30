
INCLUDES = '-I.'
INCLUDES += '-I../inc'
INCLUDES += '-I../libs'
INCLUDES += '-I../libs/etl/include'
INCLUDES += '-I../libs/fmt/include'
INCLUDES += '-I../libs/plotlypp/include'
INCLUDES += '-I../libs/json/single_include'

CXXFLAGS = '-std=c++20 -O3 -ffast-math -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16 -fno-exceptions -fno-rtti -fno-threadsafe-statics'

objs = tup.foreach_rule('*.cpp', 'arm-none-eabi-g++ -c %f '..CXXFLAGS..' '..WARNINGS..' '..INCLUDES..' -o %o -ffunction-sections -fdata-sections', 'build/obj/%B.o')

tup.foreach_rule(objs, 'arm-none-eabi-objdump -drC %f > %o', 'build/%B.asm')


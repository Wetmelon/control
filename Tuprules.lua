INCLUDES = {
    '.',
     'source',
     'libs',
     'libs/eigen',
     'libs/matplotplusplus/source',
     'build/libs/matplotplusplus',
}

WARNINGS = '-Wall -Wextra'

COMMON_FLAGS = '-O3 -ffunction-sections -fdata-sections -march=native -mtune=native -fno-omit-frame-pointer'
CCFLAGS = '-std=c17'
CXXFLAGS = '-std=c++23'
LDFLAGS = ''

CC_PATH = tup.getconfig("COMPILER_PATH")
if CC_PATH ~= "" then
    print("Compiler Path: "..CC_PATH)
end

if tup.getconfig('LTO') == 'true' then
    COMMON_FLAGS += ' -flto'
    LDFLAGS += ' -flto'
end

ROOT = tup.getcwd()
BUILD_DIR = ROOT..'/build/'
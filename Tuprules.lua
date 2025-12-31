INCLUDES = {
    '.',
     'source',
     'libs',
     'libs/eigen',
     'libs/fmt/include',
     'libs/matplotplusplus/source',
}

WARNINGS = '-Wall -Wextra'

COMMON_FLAGS = '-O3 -ffunction-sections -fdata-sections -march=native -mtune=native -fno-omit-frame-pointer'
CCFLAGS = COMMON_FLAGS..' -std=c17'
CXXFLAGS = COMMON_FLAGS..' -std=c++23'
LDFLAGS = '-Wl,--gc-sections'

CC_PATH = tup.getconfig("COMPILER_PATH")
if CC_PATH ~= "" then
    print("Compiler Path: "..CC_PATH)
end

if tup.getconfig('LTO') == 'true' then
    CCFLAGS += '-flto'
    CXXFLAGS += '-flto'
    LDFLAGS += '-flto'
end

ROOT = tup.getcwd()
BUILD_DIR = ROOT..'/build/'
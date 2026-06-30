
-- Append path separator if it doesn't end with one
local compiler_path = tup.getconfig('CONFIG_COMPILER_PATH', '')
if compiler_path ~= '' then
    local last_char = compiler_path:sub(-1)
    if last_char ~= '/' and last_char ~= '\\' then
        compiler_path = compiler_path .. '/'
    end
end

CXX =  compiler_path .. tup.getconfig('CONFIG_COMPILER_PREFIX', '') .. 'g++'

WARNINGS = '-Wall -Wextra -Wdouble-promotion -Wno-nan-infinity-disabled -Wno-psabi'
CXXFLAGS = '-O3 -std=c++20 -march=native -ffunction-sections -fdata-sections '
LDFLAGS = '-O3 -march=native -ffunction-sections -fdata-sections -Wl,--gc-sections'

-- Backend selection (mirrors wet/config.hpp). The default profile is the C++
-- stdlib; a tup variant with CONFIG_BACKEND=ETL builds the library against the
-- ETL container backend + the freestanding (constexpr-series) math backend.
BACKEND = tup.getconfig('CONFIG_BACKEND', '')
if BACKEND == 'ETL' then
    CXXFLAGS = CXXFLAGS .. '-DWET_BACKEND_ETL -DWET_MATH_BACKEND_FREESTANDING -I../libs/etl/include '
end


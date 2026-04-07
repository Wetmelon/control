
-- Append path separator if it doesn't end with one
local compiler_path = tup.getconfig('CONFIG_COMPILER_PATH', '')
if compiler_path ~= '' then
    local last_char = compiler_path:sub(-1)
    if last_char ~= '/' and last_char ~= '\\' then
        compiler_path = compiler_path .. '/'
    end
end

CXX =  compiler_path .. tup.getconfig('CONFIG_COMPILER_PREFIX', '') .. 'g++'

WARNINGS = '-Wall -Wextra -Wdouble-promotion'
CXXFLAGS = '-O3 -std=c++20 -march=native -ffunction-sections -fdata-sections -fconstexpr-ops-limit=100000000'
LDFLAGS = '-O3 -march=native -ffunction-sections -fdata-sections -Wl,--gc-sections'


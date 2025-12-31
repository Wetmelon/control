-- Python bindings build file

-- Get Python paths dynamically using Python itself
-- This makes the build portable across different machines/users
local function get_python_config(query)
    local handle = io.popen('py -3 -c "'..query..'"')
    local result = handle:read("*a")
    handle:close()
    return result:gsub("%s+$", "")  -- trim trailing whitespace
end

PYTHON_INCLUDE = get_python_config([[import sysconfig; print(sysconfig.get_path('include'))]])
PYBIND11_INCLUDE = get_python_config([[import pybind11; print(pybind11.get_include())]])
PYTHON_LIBS = get_python_config([[import sysconfig; import os; print(os.path.join(os.path.dirname(sysconfig.get_path('include')), 'libs'))]])

-- Get extension suffix dynamically
EXT_SUFFIX = get_python_config([[import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.pyd')]])

-- Python module includes
PY_INCLUDES = {
    ROOT..'/source',
    ROOT..'/libs/eigen',
    ROOT..'/libs/fmt/include',
    PYTHON_INCLUDE,
    PYBIND11_INCLUDE,
}

-- Convert to compiler flags
PY_INCLUDE_FLAGS = ''
for k, v in ipairs(PY_INCLUDES) do
    PY_INCLUDE_FLAGS = PY_INCLUDE_FLAGS..' -I'..v
end

-- Compiler flags for Python module
PY_CXXFLAGS = CXXFLAGS..' -shared'
PY_LDFLAGS = LDFLAGS..' -L'..PYTHON_LIBS..' -lpython311 -static-libgcc -static-libstdc++'

-- Use same flags as the binding for consistency
PY_CONTROL_FLAGS = COMMON_FLAGS

-- Compile the Python binding
tup.rule('pycontrol.cpp', 
    '^j Python Bindings^ '..CC_PATH..'g++ '..PY_INCLUDE_FLAGS..' '..PY_CXXFLAGS..' -c %f -o %o',
    'build/pycontrol.o')

-- Link to create Python module - reference .o files directly
tup.rule({'build/pycontrol.o'}, 
    '^o Python Module^ '..CC_PATH..'g++ '..PY_CXXFLAGS..' %f '..PY_LDFLAGS..' -o %o',
    'pycontrol'..EXT_SUFFIX)

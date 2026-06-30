
-- The example programs are hosted demos (fmt / plotlypp). They are not part of
-- the freestanding contract, so skip them under the ETL backend variant.
if BACKEND == 'ETL' then
    return
end

INCLUDES = '-I.'
INCLUDES += '-I../inc'
INCLUDES += '-I../libs'
INCLUDES += '-I../libs/fmt/include'
INCLUDES += '-I../libs/plotlypp/include'
INCLUDES += '-I../libs/json/single_include'

-- Compile all fmt source files into the <fmt> bin so the per-example link can
-- reference them with %<fmt> (bin paths carry the variant prefix correctly;
-- a bare $(var) group does not).
fmt_sources = {'../libs/fmt/src/format.cc', '../libs/fmt/src/os.cc'}
tup.foreach_rule(fmt_sources, '^j^'..CXX..' -c %f '..CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -o %o', {'build/fmt/%B.o', '<fmt>'})

-- Compile every example .cpp: the flat top-level demos plus the grouped ones in
-- subfolders (e.g. PMAC/). Basenames are unique, so objects share build/obj/.
sources = tup.glob('*.cpp')
for _, f in ipairs(tup.glob('PMAC/*.cpp')) do
    sources[#sources + 1] = f
end
ex_objs = tup.foreach_rule(sources, '^j^'..CXX..' -c %f '..CXXFLAGS..' '..INCLUDES..' '..WARNINGS..' -o %o', 'build/obj/%B.o')
ex_objs.extra_inputs = {'<fmt>'}

-- Link with g++ (each example + the fmt objects from the bin)
examples = tup.foreach_rule(ex_objs, CXX..' '..CXXFLAGS..' '..LDFLAGS..' %f %<fmt> -o %o', 'build/%B.exe')

-- Run test executables
-- tup.frule{inputs = examples, command = './%f', outputs = {}}

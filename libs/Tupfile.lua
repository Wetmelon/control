-- ============================================================================
-- Matplot++ Build Configuration
-- ============================================================================

-- Matplot++ compiler flags (C++17, Windows-specific defines)
-- Include nodesoup include directory for nodesoup.hpp
MATPLOT_INCLUDES = '-I. -I./eigen -I./matplotplusplus/source -I./matplotplusplus/source/3rd_party/nodesoup/include -I./matplotplusplus/source/3rd_party/cimg -I./matplotplusplus'
MATPLOT_CXXFLAGS = '-std=c++17 -Wall -Wextra -pedantic -Werror -Wno-error=class-memaccess -Wno-class-memaccess  -Wno-char-subscripts -Wno-misleading-indentation'
MATPLOT_DEFINES = '-Dcimg_display=2 -DMATPLOT_EXPORTS='
MATPLOT_FLAGS = MATPLOT_INCLUDES..' '..COMMON_FLAGS..' '..MATPLOT_CXXFLAGS..' '..MATPLOT_DEFINES..' '..WARNINGS

nodesoup_cpp_files = {
    '3rd_party/nodesoup/src/algebra.cpp',
    '3rd_party/nodesoup/src/fruchterman_reingold.cpp',
    '3rd_party/nodesoup/src/kamada_kawai.cpp',
    '3rd_party/nodesoup/src/layout.cpp',
    '3rd_party/nodesoup/src/nodesoup.cpp',
}

for i, f in ipairs(nodesoup_cpp_files) do
    nodesoup_cpp_files[i] = 'matplotplusplus/source/' .. f
end

-- List of all Matplot++ source files
matplot_cpp_files = {
    'backend/backend_interface.cpp',
    'backend/gnuplot.cpp',
    'backend/backend_registry.cpp',

    'core/axes_type.cpp',
    'core/axes_object.cpp',
    'core/axis_type.cpp',
    'core/figure_type.cpp',
    'core/figure_registry.cpp',
    'core/legend.cpp',
    'core/line_spec.cpp',

    'util/colors.cpp',
    'util/common.cpp',
    'util/contourc.cpp',
    'util/popen.cpp',
    'util/world_cities.cpp',
    'util/world_map_10m.cpp',
    'util/world_map_50m.cpp',
    'util/world_map_110m.cpp',

    'axes_objects/bars.cpp',
    'axes_objects/box_chart.cpp',
    'axes_objects/circles.cpp',
    'axes_objects/contours.cpp',
    'axes_objects/error_bar.cpp',
    'axes_objects/filled_area.cpp',
    'axes_objects/function_line.cpp',
    'axes_objects/histogram.cpp',
    'axes_objects/labels.cpp',
    'axes_objects/line.cpp',
    'axes_objects/matrix.cpp',
    'axes_objects/network.cpp',
    'axes_objects/parallel_lines.cpp',
    'axes_objects/stair.cpp',
    'axes_objects/string_function.cpp',
    'axes_objects/surface.cpp',
    'axes_objects/vectors.cpp',

    'freestanding/axes_functions.cpp',
    'freestanding/histcounts.cpp',
}

for i, f in ipairs(matplot_cpp_files) do
    matplot_cpp_files[i] = 'matplotplusplus/source/matplot/' .. f
end

matplot_objs = tup.foreach_rule(matplot_cpp_files, '^j^'..CC_PATH..'g++ '..MATPLOT_FLAGS..' -c %f -o %o', BUILD_DIR..'matplotplusplus/obj/%B.o')
matplot_objs += tup.foreach_rule(nodesoup_cpp_files, '^j^'..CC_PATH..'g++ '..MATPLOT_FLAGS..' -D_USE_MATH_DEFINES -c %f -o %o', BUILD_DIR..'matplotplusplus/obj/%B.o')

-- Link matplot library (combine matplot objects and nodesoup objects into one library)
matplot_lib = tup.rule(
    matplot_objs,
    'ar rcs %o %f',
    'libmatplot.a'
)

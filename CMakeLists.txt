cmake_minimum_required(VERSION 3.20)
project(control
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Header-only C++20 control systems library for embedded systems"
)

# --- Header-only library target ---
add_library(control INTERFACE)
add_library(wet ALIAS control)

target_include_directories(control INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/matrix>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/math>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/matrix>
    $<INSTALL_INTERFACE:include/math>
)
target_compile_features(control INTERFACE cxx_std_20)

# --- Install rules ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS control
    EXPORT controlTargets
)

install(DIRECTORY inc/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT controlTargets
    FILE controlTargets.cmake
    NAMESPACE wetmelon::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/control
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/controlConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/controlConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/control
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/controlConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/controlConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/controlConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/control
)

# --- Optional: tests & examples ---
option(CONTROL_BUILD_TESTS    "Build unit tests"  OFF)
option(CONTROL_BUILD_EXAMPLES "Build examples"    OFF)

if(CONTROL_BUILD_TESTS OR CONTROL_BUILD_EXAMPLES)
    add_library(fmt STATIC
        libs/fmt/src/format.cc
        libs/fmt/src/os.cc
    )
    target_include_directories(fmt PUBLIC libs/fmt/include)
    target_compile_features(fmt PUBLIC cxx_std_20)
endif()

# Shared warning/optimization flags (mirrors Tuprules.lua WARNINGS + CXXFLAGS).
# FAST_MATH adds -ffast-math, which the tup build applies to test TUs only --
# constexpr static_asserts act as a tripwire that compile-time eval stays
# IEEE-strict even under -ffast-math.
function(control_apply_flags target fast_math)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:
            -O3 -march=native
            -ffunction-sections -fdata-sections
            -Wall -Wextra -Wdouble-promotion -Wno-nan-infinity-disabled
        >
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
    )
    target_link_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,--gc-sections>)
    # -ffast-math + -static apply to the test runners; examples build IEEE-strict
    # and link shared, matching Tuprules.lua (TEST_CXXFLAGS / LDFLAGS).
    if(fast_math)
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-ffast-math>)
        target_link_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-static>)
    endif()
endfunction()

# --- Tests ---
if(CONTROL_BUILD_TESTS)
    enable_testing()

    # Default profile: every tests/*.cpp linked into one runner (matches tup).
    file(GLOB TEST_SOURCES tests/*.cpp)
    add_executable(test_runner ${TEST_SOURCES})
    target_link_libraries(test_runner PRIVATE control fmt)
    target_include_directories(test_runner PRIVATE
        tests
        libs
        libs/plotlypp/include
        libs/json/single_include
    )
    control_apply_flags(test_runner TRUE)
    add_test(NAME control_tests COMMAND test_runner)

    # The wet-math and ETL container backends are ODR-incompatible with the
    # default std backend the runner links against, so each gets its own
    # executable (mirrors tests/{wet,etl}_backend/Tupfile.lua). No fmt; the ETL
    # variant's own include dir must precede tests/ so its wet_profile.hpp wins.
    file(GLOB WET_BACKEND_SOURCES tests/wet_backend/*.cpp)
    add_executable(test_wet_backend ${WET_BACKEND_SOURCES})
    target_link_libraries(test_wet_backend PRIVATE control)
    target_include_directories(test_wet_backend PRIVATE tests/wet_backend tests)
    control_apply_flags(test_wet_backend TRUE)
    add_test(NAME control_tests_wet_backend COMMAND test_wet_backend)

    file(GLOB ETL_BACKEND_SOURCES tests/etl_backend/*.cpp)
    add_executable(test_etl_backend ${ETL_BACKEND_SOURCES})
    target_link_libraries(test_etl_backend PRIVATE control)
    target_include_directories(test_etl_backend PRIVATE
        tests/etl_backend tests libs/etl/include)
    control_apply_flags(test_etl_backend TRUE)
    add_test(NAME control_tests_etl_backend COMMAND test_etl_backend)
endif()

# --- Examples ---
if(CONTROL_BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SOURCES examples/*.cpp)
    foreach(src ${EXAMPLE_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE control fmt)
        target_include_directories(${name} PRIVATE
            examples
            libs
            libs/plotlypp/include
            libs/json/single_include
        )
        control_apply_flags(${name} FALSE)  # examples build IEEE-strict (no -ffast-math)
    endforeach()
endif()

# --- Docs (mirrors `make docs`) ---
find_package(Doxygen)
if(DOXYGEN_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
    add_custom_target(docs
        COMMAND ${CMAKE_COMMAND} -E make_directory docs/html
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()

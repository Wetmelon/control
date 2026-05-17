cmake_minimum_required(VERSION 3.20)
project(control
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "Header-only C++20 control systems library for embedded systems"
)

# --- Header-only library target ---
add_library(control INTERFACE)
add_library(wetmelon::control ALIAS control)

target_include_directories(control INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/matrix>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/matrix>
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

# --- Tests ---
if(CONTROL_BUILD_TESTS)
    enable_testing()

    file(GLOB TEST_SOURCES tests/*.cpp)
    add_executable(test_runner ${TEST_SOURCES})
    target_link_libraries(test_runner PRIVATE control fmt)
    target_include_directories(test_runner PRIVATE
        tests
        libs
        libs/plotlypp/include
        libs/json/single_include
    )
    target_compile_options(test_runner PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:
            -O3 -march=native
            -ffunction-sections -fdata-sections
            -Wall -Wextra -Wdouble-promotion
        >
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
    )
    target_link_options(test_runner PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,--gc-sections -static>
    )

    add_test(NAME control_tests COMMAND test_runner)
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
        target_compile_options(${name} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:
                -O3 -march=native
                -ffunction-sections -fdata-sections
                -Wall -Wextra -Wdouble-promotion
            >
            $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
        )
        target_link_options(${name} PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,--gc-sections>
        )
    endforeach()
endif()

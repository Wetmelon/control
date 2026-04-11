cmake_minimum_required(VERSION 3.20)
project(control LANGUAGES CXX)

# --- Header-only library target ---
add_library(control INTERFACE)
target_include_directories(control INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/matrix>
)
target_compile_features(control INTERFACE cxx_std_20)

# --- Optional: tests & examples ---
option(CONTROL_BUILD_TESTS    "Build unit tests"  OFF)
option(CONTROL_BUILD_EXAMPLES "Build examples"    OFF)

if(CONTROL_BUILD_TESTS OR CONTROL_BUILD_EXAMPLES)
    # fmt (used by tests and examples)
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
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wdouble-promotion -fconstexpr-ops-limit=100000000>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /constexpr:steps100000000>
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
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wdouble-promotion -fconstexpr-ops-limit=100000000>
            $<$<CXX_COMPILER_ID:MSVC>:/W4 /constexpr:steps100000000>
        )
    endforeach()
endif()

# aurora_bridge: Aurora (and its SDL3) inside one DLL for libultraship's
# Aurora backend. Built on its own, not as part of libultraship's build:
# Aurora brings its own dependency set and CRT, and must not share a module
# with libultraship's static SDL2 (see include/aurora_bridge.h).
#
#   cmake -S aurora-bridge -B <out> -G Ninja -DAURORA_DIR=<aurora source>
#
# The result libultraship consumes (LUS_AURORA_BRIDGE_DIR):
#   <out>/aurora_bridge.dll, <out>/aurora_bridge.lib, include/aurora_bridge.h,
#   plus Aurora's public headers and Dawn's headers/import library.
cmake_minimum_required(VERSION 3.25)
project(aurora_bridge LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

set(AURORA_DIR "" CACHE PATH "Aurora source")
if (NOT EXISTS "${AURORA_DIR}/include/aurora/gfx.hpp")
  message(FATAL_ERROR "AURORA_DIR does not point at Aurora: ${AURORA_DIR}")
endif ()

# Only what a Fast3D backend needs: core + the WebGPU renderer.
set(AURORA_ENABLE_DVD OFF CACHE BOOL "" FORCE)
set(AURORA_ENABLE_CARD OFF CACHE BOOL "" FORCE)
set(AURORA_ENABLE_THP OFF CACHE BOOL "" FORCE)
set(AURORA_ENABLE_RMLUI OFF CACHE BOOL "" FORCE)
set(AURORA_ENABLE_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory("${AURORA_DIR}" aurora EXCLUDE_FROM_ALL)

add_library(aurora_bridge SHARED src/aurora_bridge.cpp include/aurora_bridge.h)
target_include_directories(aurora_bridge PUBLIC include)
target_compile_definitions(aurora_bridge PRIVATE AURORA_BRIDGE_BUILD)
target_link_libraries(aurora_bridge PRIVATE aurora::core aurora::gx aurora::vi)
target_link_libraries(aurora_bridge PUBLIC dawn::webgpu_dawn)

#=================== Top-Level ===================
file(GLOB Source_Files__TopLevel RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

source_group("" FILES ${Source_Files__TopLevel})
target_sources(libultraship PRIVATE ${Source_Files__TopLevel})

#=================== Debug ===================

file(GLOB Source_Files__Debug RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "debug/*.cpp")

source_group("debug" FILES ${Source_Files__Debug})
target_sources(libultraship PRIVATE ${Source_Files__Debug})

#=================== Resource ===================

file(GLOB Source_Files__Resource__Types RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "resource/type/*.h" "resource/type/*.cpp")
source_group("resource/type" FILES ${Source_Files__Resource__Types})

file(GLOB Source_Files__Resource__Factories RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "resource/factory/*.h" "resource/factory/*.cpp")
source_group("resource/factory" FILES ${Source_Files__Resource__Factories})

target_sources(libultraship PRIVATE ${Source_Files__Resource} ${Source_Files__Resource__Types} ${Source_Files__Resource__Factories} ${Source_Files__Resource__Archive})

#=================== Graphic ===================

file(GLOB Source_Files__Graphic RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "backends/*.cpp")

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    list(FILTER Source_Files__Graphic EXCLUDE REGEX "graphic/Fast3D/backends/gfx_dxgi*")
    list(FILTER Source_Files__Graphic EXCLUDE REGEX "graphic/Fast3D/backends/gfx_direct3d*")
    list(FILTER Source_Files__Graphic EXCLUDE REGEX "graphic/Fast3D/backends/dxsdk/*")
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    list(FILTER Source_Files__Graphic EXCLUDE REGEX "graphic/Fast3D/backends/gfx_metal*")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    list(FILTER Source_Files__Graphic EXCLUDE REGEX "graphic/Fast3D/backends/gfx_opengl*")
endif()

source_group("graphic" FILES ${Source_Files__Graphic})
target_sources(libultraship PRIVATE ${Source_Files__Graphic})

#=================== Aurora (optional) ===================
# Fast3D on Aurora's WebGPU layer through aurora_bridge.dll (aurora-bridge/).
# Off unless LUS_AURORA_BRIDGE_DIR points at a built bridge SDK
# (include/, lib/aurora_bridge.lib, lib/webgpu_dawn.lib, bin/*.dll).

set(LUS_AURORA_BRIDGE_DIR "" CACHE PATH "Built aurora_bridge SDK directory; empty disables the Aurora backend")
if (LUS_AURORA_BRIDGE_DIR)
    if (NOT EXISTS "${LUS_AURORA_BRIDGE_DIR}/include/aurora_bridge.h")
        message(FATAL_ERROR "LUS_AURORA_BRIDGE_DIR has no include/aurora_bridge.h: ${LUS_AURORA_BRIDGE_DIR}")
    endif()
    file(GLOB Source_Files__Graphic__Aurora RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "backends/aurora/*.cpp")
    source_group("graphic/aurora" FILES ${Source_Files__Graphic__Aurora})
    target_sources(libultraship PRIVATE ${Source_Files__Graphic__Aurora})
    target_include_directories(libultraship PRIVATE "${LUS_AURORA_BRIDGE_DIR}/include")
    target_compile_definitions(libultraship PRIVATE ENABLE_AURORA IMGUI_IMPL_WEBGPU_BACKEND_DAWN)
    # ImGui's stock WebGPU renderer, against libultraship's own ImGui.
    target_sources(libultraship PRIVATE ${imgui_SOURCE_DIR}/backends/imgui_impl_wgpu.cpp)
    target_link_libraries(libultraship PUBLIC "${LUS_AURORA_BRIDGE_DIR}/lib/aurora_bridge.lib"
                                              "${LUS_AURORA_BRIDGE_DIR}/lib/webgpu_dawn.lib")
    if (MSVC)
        # Load the bridge only when the Aurora backend is chosen, so the other
        # backends still start when its DLLs are absent.
        target_link_options(libultraship INTERFACE "/DELAYLOAD:aurora_bridge.dll" "/DELAYLOAD:webgpu_dawn.dll")
        target_link_libraries(libultraship PUBLIC delayimp.lib)
    endif()
    file(GLOB LUS_AURORA_RUNTIME_DLLS "${LUS_AURORA_BRIDGE_DIR}/bin/*.dll")
    set(LUS_AURORA_RUNTIME_DLLS "${LUS_AURORA_RUNTIME_DLLS}" CACHE INTERNAL "DLLs a program using the Aurora backend needs beside it")
    message(STATUS "libultraship: Aurora backend enabled (${LUS_AURORA_BRIDGE_DIR})")
endif()

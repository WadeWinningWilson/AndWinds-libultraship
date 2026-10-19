include("${CMAKE_CURRENT_LIST_DIR}/Default.cmake")

# SHIP_OUTPUT_ROOT lets a second build tree (e.g. the hosted provider) keep its outputs apart.
if(NOT SHIP_OUTPUT_ROOT)
    set(SHIP_OUTPUT_ROOT "${CMAKE_SOURCE_DIR}")
endif()
set_config_specific_property("OUTPUT_DIRECTORY" "${SHIP_OUTPUT_ROOT}$<$<NOT:$<STREQUAL:${CMAKE_VS_PLATFORM_NAME},Win32>>:/${CMAKE_VS_PLATFORM_NAME}>/${PROPS_CONFIG}")

if(MSVC)
    create_property_reader("DEFAULT_CXX_EXCEPTION_HANDLING")
    create_property_reader("DEFAULT_CXX_DEBUG_INFORMATION_FORMAT")

    # set_target_properties("${PROPS_TARGET}" PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    set_config_specific_property("DEFAULT_CXX_EXCEPTION_HANDLING" "/EHsc")
    if (CMAKE_C_COMPILER_LAUNCHER STREQUAL "sccache")
        set_config_specific_property("DEFAULT_CXX_DEBUG_INFORMATION_FORMAT" "/Z7")
    else()
        set_config_specific_property("DEFAULT_CXX_DEBUG_INFORMATION_FORMAT" "/Zi")
    endif()
endif()
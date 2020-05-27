set(target_arch_c "
#if defined(__i386) || defined(__i386__) || defined(_M_IX86)
    #error TARGET_ARCH x86
#elif defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || defined(_M_X64)
    #error TARGET_ARCH x64
#else
  #error TARGET_ARCH unknown
#endif
")

function(target_arch output_var)
    if(APPLE AND CMAKE_OSX_ARCHITECTURES)
        foreach(osx_arch ${CMAKE_OSX_ARCHITECTURES})
            if("${osx_arch}" STREQUAL "i386")
                set(osx_arch_i386 TRUE)
            elseif("${osx_arch}" STREQUAL "x86_64")
                set(osx_arch_x86_64 TRUE)
            else()
                message(FATAL_ERROR "Unknown OS X architecture: ${osx_arch}")
            endif()
        endforeach()

        if(osx_arch_i386)
            list(APPEND ARCH x86)
        endif()

        if(osx_arch_x86_64)
            list(APPEND ARCH x64)
        endif()

    else()
        file(WRITE "${CMAKE_BINARY_DIR}/target_arch.c" "${target_arch_c}")
        enable_language(C)
        try_run(
            run_result_unused
            compile_result_unused
            "${CMAKE_BINARY_DIR}"
            "${CMAKE_BINARY_DIR}/target_arch.c"
            COMPILE_OUTPUT_VARIABLE ARCH
            CMAKE_FLAGS CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
        )

        string(REGEX MATCH "TARGET_ARCH ([a-zA-Z0-9_]+)" ARCH "${ARCH}")
        string(REPLACE "TARGET_ARCH " "" ARCH "${ARCH}")

        if (NOT ARCH)
            set(ARCH unknown)
        endif()
    endif()

    set(${output_var} "${ARCH}" PARENT_SCOPE)
endfunction()

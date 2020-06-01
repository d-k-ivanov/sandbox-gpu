if(MAKE_TESTS)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)

    file(GLOB_RECURSE SRC
    )
    source_group("Main" FILES ${SRC})

    file(GLOB_RECURSE SRC_LIBS
    )
    source_group("Libs" FILES ${SRC_LIBS})

    # add_executable(test ${SRC} ${SRC_LIBS})
endif()


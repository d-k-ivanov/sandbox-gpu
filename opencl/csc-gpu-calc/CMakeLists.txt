cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(csc-gpu-calc LANGUAGES CXX VERSION 1.0.0.0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include(Common)
include(Policies)
include(Architecture)
include(CompilerFlags)
include(Macros)

# Set basic variables: output paths, install paths, lib paths, etc.
set_base_vars()

# Set PROJECT_VERSION_PATCH and PROJECT_VERSION_TWEAK to 0 if not present, needed by add_project_meta
fix_project_version()

# Set additional project information
set(COMPANY "Dmitriy Ivanov")
set(COPYRIGHT "Copyright (c) 2020 Dmitriy Ivanov. All rights reserved.")
set(IDENTIFIER "com.pupupon.dev.csc-gpu-calc")

include(Debug)
cmake_get_cxx_compiler()
cmake_list_all_init_variables()
#cmake_list_all_variables()
#cmake_get_includes()


file(WRITE  ${OUTPUT_DIR_MAIN}/about.txt "${COPYRIGHT}\n")
file(APPEND ${OUTPUT_DIR_MAIN}/about.txt "--------------------------------------------------------------------------------\n")
file(APPEND ${OUTPUT_DIR_MAIN}/about.txt "Application Name: ${CMAKE_PROJECT_NAME}\n")
file(APPEND ${OUTPUT_DIR_MAIN}/about.txt "Application Version: ${CMAKE_PROJECT_VERSION}\n")
file(APPEND ${OUTPUT_DIR_MAIN}/about.txt "Company: ${COMPANY}\n")
file(APPEND ${OUTPUT_DIR_MAIN}/about.txt "Identifier: ${IDENTIFIER}\n")

# Find dependencies
find_package(OpenCL 1.1 REQUIRED)

include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/lib)

add_subdirectory("lib")     # ${LIBS}
add_subdirectory("src")     # ${SRC}

add_executable(csc-gpu-calc ${SRC} ${LIBS})
target_link_libraries(csc-gpu-calc ${OpenCL_LIBRARIES})

if(NOT TARGET csc-gpu-calc)
    add_custom_target(csc-gpu-calc)
    add_dependencies(csc-gpu-calc csc-gpu-calc)
endif()

# Testing
option(MAKE_TESTS "Set to ON to build tests" OFF)
add_subdirectory(${CMAKE_SOURCE_DIR}/test)

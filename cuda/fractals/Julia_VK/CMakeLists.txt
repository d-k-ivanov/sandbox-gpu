#--------------------------------------------------------------------------------
# Base
#--------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.5)
project(fractal-julia)

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(${CMAKE_SOURCE_DIR}/cmake/arch.cmake)
target_arch(arch)
if (MSVC)
  set(COMPILER msvc)
elseif(MINGW)
  set(COMPILER mingw)
else()
  message( FATAL_ERROR "Unsupported compiler. Exited..." )
endif()

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#include_directories("${PROJECT_SOURCE_DIR}")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14 -Weverything")

#--------------------------------------------------------------------------------
# Dependencies
#--------------------------------------------------------------------------------

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(LIBRARY_DIR ${PROJECT_SOURCE_DIR}/libs/${arch}/${COMPILER})
include_directories(${INCLUDE_DIR})
link_directories(${LIBRARY_DIR})

#--------------------------------------------------------------------------------
# Add subdirectories
#--------------------------------------------------------------------------------
# add_subdirectory(src)

#--------------------------------------------------------------------------------
# Binary
#--------------------------------------------------------------------------------
add_executable(fractal-julia ${PROJECT_SOURCE_DIR}/src/fractal-julia.c)
target_link_libraries(fractal-julia glfw3 glu32 opengl32)

message("INCLUDE_DIR: " ${INCLUDE_DIR})
message("LIBRARY_DIR: " ${LIBRARY_DIR})

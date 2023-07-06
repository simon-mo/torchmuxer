cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(fijit-sys LANGUAGES CXX CUDA)
enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CXX_COMPILER)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch;${CMAKE_PREFIX_PATH})

find_package(CUDAToolkit REQUIRED)
include(cmake/FindLibunwind.cmake)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(third_party/redox EXCLUDE_FROM_ALL)
add_subdirectory(third_party/glog EXCLUDE_FROM_ALL)
add_subdirectory(third_party/fmt)
add_subdirectory(third_party/pybind11)

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(third_party/json)



find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

macro(CUDA_TARGET)
    target_compile_features(${ARGV} PUBLIC cxx_std_14)
    set_target_properties(${ARGV} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endmacro(CUDA_TARGET)

# Our Main Executable is fijit-sys binary
add_executable(${PROJECT_NAME}
               src/main.cc
               src/fijit.h
               src/fijit.cc
               third_party/cxxopts/cxxopts.hpp)
CUDA_TARGET(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME}
                      PRIVATE
                      ${LIBUNWIND_LIBRARIES}
                      glog::glog
                      CUDA::cudart
                      CUDA::cuda_driver
                      CUDA::cupti
                      fmt::fmt
                      PRIVATE nlohmann_json::nlohmann_json
                    #   ${TORCH_LIBRARIES}
)

add_library(${PROJECT_NAME}-lib
               src/fijit.h
               src/fijit.cc
)
CUDA_TARGET(${PROJECT_NAME}-lib)
target_link_libraries(${PROJECT_NAME}-lib
                    PRIVATE
                      glog::glog
                      CUDA::cudart
                      CUDA::cuda_driver
                      CUDA::cupti
                      fmt::fmt
                       nlohmann_json::nlohmann_json
                    #   ${TORCH_LIBRARIES}
)


pybind11_add_module(fijit_py src/mod.cpp)
target_link_libraries(fijit_py PRIVATE ${PROJECT_NAME}-lib)
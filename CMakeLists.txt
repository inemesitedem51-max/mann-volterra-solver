cmake_minimum_required(VERSION 3.10)
project(MannVolterra)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3 (header-only)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# If MPI is available, enable it
find_package(MPI QUIET)
if(MPI_FOUND)
    add_definitions(-DUSE_MPI)
    include_directories(${MPI_INCLUDE_PATH})
endif()

add_executable(mannvolterra src/main.cpp)

if(MPI_FOUND)
    target_link_libraries(mannvolterra ${MPI_LIBRARIES})
endif()

# For OpenMP (optional)
find_package(OpenMP QUIET)
if(OpenMP_FOUND)
    target_link_libraries(mannvolterra OpenMP::OpenMP_CXX)
endif()
Add CMake build config

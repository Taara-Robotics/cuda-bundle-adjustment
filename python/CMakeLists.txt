cmake_minimum_required(VERSION 3.18)

project(cuba LANGUAGES CXX)
set(CUBA_VERSION_INFO "0.0.1")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

# Download pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG       v2.11.1)
FetchContent_MakeAvailable(pybind11)

# required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
# set(SRCS_COMMON object_creator.h)


pybind11_add_module(cuba python_bindings.cpp object_creator.h)
target_compile_definitions(cuba PRIVATE VERSION_INFO=${CUBA_VERSION_INFO})
target_include_directories(cuba PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
target_link_libraries(cuba PRIVATE cuda_bundle_adjustment ${OpenCV_LIBS})

# Fetch python-wheel
FetchContent_Declare(
  python-wheel
  GIT_REPOSITORY https://github.com/Klebert-Engineering/python-cmake-wheel.git
  GIT_TAG v0.9.0)
FetchContent_MakeAvailable(python-wheel)

# Add python-wheel directory to cmake module path
list(APPEND CMAKE_MODULE_PATH ${python-wheel_SOURCE_DIR})
set(WHEEL_DEPLOY_DIRECTORY ${CMAKE_BINARY_DIR})
include(python-wheel)

# Add the wheel target
add_wheel(cuba 
  NAME cuba
  VERSION ${CUBA_VERSION_INFO}
  AUTHOR "Fixstars Corporation"
  PYTHON_REQUIRES ">=3.8"
  DESCRIPTION "C++/Python library for bundle adjustment using CUDA"
  SCRIPTS
    example.py
  PYTHON_PACKAGE_DIRS
    .
)

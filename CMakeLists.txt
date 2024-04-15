cmake_minimum_required(VERSION 3.18)

project(CudaBundleAdjustment)

option(ENABLE_SAMPLES   "Build samples" OFF)
option(ENABLE_PYTHON    "Build python bindings" ON)
option(WITH_G2O         "Build sample with g2o (g2o needs to be installed)" OFF)
option(USE_FLOAT32      "Use 32bit float in internal floating-point operations (default is 64bit float)" OFF)
option(BUILD_SHARED_LIB "Build shared library" OFF)
option(CUDA_ARCHS "List of architectures to generate device code for" "61;72;75;86")

add_subdirectory(src)

if(ENABLE_SAMPLES)
	add_subdirectory(samples)
endif()

if(ENABLE_PYTHON)
	add_subdirectory(python)
endif()
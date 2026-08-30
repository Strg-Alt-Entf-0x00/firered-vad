cmake_minimum_required(VERSION 3.18)
project(firered-vad VERSION 0.6.0 LANGUAGES CXX)

# ============================================================================
# Project Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(firered_vad_BUILD_TESTS "Build FireRed VAD tests" OFF)
option(firered_vad_BUILD_EXAMPLES "Build FireRed VAD examples" ON)
option(firered_vad_USE_CUDA "Enable CUDA support" ON)
option(firered_vad_BUILD_SHARED "Build shared library" OFF)

# ============================================================================
# Find Dependencies
# ============================================================================

# GGML (required)
if(NOT TARGET ggml)
    # Try to find GGML in parent directories (third-party-cpp)
    set(GGML_SEARCH_PATHS
        "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-0.15.1"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party-cpp/ggml-0.15.1"
    )
    
    foreach(SEARCH_PATH ${GGML_SEARCH_PATHS})
        if(EXISTS "${SEARCH_PATH}/CMakeLists.txt")
            message(STATUS "Found GGML at: ${SEARCH_PATH}")
            add_subdirectory("${SEARCH_PATH}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
            set(GGML_FOUND TRUE)
            break()
        endif()
    endforeach()
    
    if(NOT GGML_FOUND)
        message(STATUS "GGML not found locally. Fetching from GitHub (Best Practice Fallback)...")
        include(FetchContent)
        FetchContent_Declare(ggml
            GIT_REPOSITORY https://github.com/ggml-org/ggml.git
            GIT_TAG        master
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(ggml)
    endif()
endif()

# ============================================================================
# Library Sources
# ============================================================================

set(firered_vad_SOURCES
    src/firered_vad.cpp
    src/firered_fbank.cpp
    src/firered_postprocess.cpp
)

set(firered_vad_HEADERS
    include/firered-vad/firered_vad.h
    src/firered_fbank.h
)

# ============================================================================
# Library Target
# ============================================================================

if(firered_vad_BUILD_SHARED)
    add_library(firered-vad SHARED ${firered_vad_SOURCES} ${firered_vad_HEADERS})
else()
    add_library(firered-vad STATIC ${firered_vad_SOURCES} ${firered_vad_HEADERS})
endif()

# Alias for consistency
add_library(FireRed::vad ALIAS firered-vad)

# ============================================================================
# Include Directories
# ============================================================================

target_include_directories(firered-vad
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ============================================================================
# Link Libraries
# ============================================================================

target_link_libraries(firered-vad
    PUBLIC
        ggml
    PRIVATE
        $<$<BOOL:${firered_vad_USE_CUDA}>:ggml-cuda>
)

# ============================================================================
# Compiler Settings
# ============================================================================

target_compile_features(firered-vad PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(firered-vad PRIVATE
        /W4
        /WX-        # Don't treat warnings as errors (for now)
        /permissive-
        /Zc:__cplusplus
        /utf-8
    )
    
    # Suppress specific warnings
    target_compile_definitions(firered-vad PRIVATE
        _CRT_SECURE_NO_WARNINGS
        _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
    )
else()
    target_compile_options(firered-vad PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-function
    )
endif()

# ============================================================================
# Platform-specific Settings
# ============================================================================

if(WIN32)
    target_compile_definitions(firered-vad PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()

# ============================================================================
# Model Files
# ============================================================================

# Install model file with the library (optional)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/firered-vad.gguf")
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/models/firered-vad.gguf"
            DESTINATION share/firered-vad/models)
    
    # Copy to build directory for testing
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/models/firered-vad.gguf"
         DESTINATION "${CMAKE_BINARY_DIR}/models")
    
    message(STATUS "FireRed-VAD model found: models/firered-vad.gguf (97.57% F1 Score)")
elseif(PROJECT_IS_TOP_LEVEL)
    message(WARNING "FireRed-VAD model not found at models/firered-vad.gguf")
    message(WARNING "  Download from: https://huggingface.co/cstr/firered-vad-GGUF")
    message(WARNING "  Place at: models/firered-vad.gguf")
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS firered-vad
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# ============================================================================
# Examples (Optional)
# ============================================================================

if(firered_vad_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Tests (Optional)
# ============================================================================

if(firered_vad_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Configuration Summary
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "FireRed VAD Configuration")
message(STATUS "========================================")
message(STATUS "Version:        ${PROJECT_VERSION}")
message(STATUS "Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:   C++${CMAKE_CXX_STANDARD}")
if(MSVC)
    message(STATUS "Compiler:       MSVC ${MSVC_VERSION}")
else()
    message(STATUS "Compiler:       ${CMAKE_CXX_COMPILER_ID}")
endif()
message(STATUS "Shared Library: ${firered_vad_BUILD_SHARED}")
message(STATUS "CUDA Support:   ${firered_vad_USE_CUDA}")
message(STATUS "Build Tests:    ${firered_vad_BUILD_TESTS}")
message(STATUS "Build Examples: ${firered_vad_BUILD_EXAMPLES}")
message(STATUS "========================================")
message(STATUS "")


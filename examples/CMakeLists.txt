# Examples CMakeLists.txt

# Test C++ Inference
add_executable(test_cpp_inference test_cpp_inference.cpp)

target_link_libraries(test_cpp_inference 
    PRIVATE 
        FireRed::vad
)

# Set C++20
target_compile_features(test_cpp_inference PRIVATE cxx_std_20)

# Platform-specific settings
if(MSVC)
    target_compile_options(test_cpp_inference PRIVATE /W4)
else()
    target_compile_options(test_cpp_inference PRIVATE -Wall -Wextra)
endif()

# Install target (optional)
install(TARGETS test_cpp_inference
    RUNTIME DESTINATION bin
)

# Test with real audio files
add_executable(test_audio_file test_audio_file.cpp)

target_link_libraries(test_audio_file 
    PRIVATE 
        FireRed::vad
)

target_compile_features(test_audio_file PRIVATE cxx_std_20)

if(MSVC)
    target_compile_options(test_audio_file PRIVATE /W4)
else()
    target_compile_options(test_audio_file PRIVATE -Wall -Wextra)
endif()

install(TARGETS test_audio_file
    RUNTIME DESTINATION bin
)

message(STATUS "Example programs configured: test_cpp_inference, test_audio_file")

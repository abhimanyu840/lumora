cmake_minimum_required(VERSION 3.22.1)
project(lumora_mobile_native)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure native JNI bridging library
add_library(
    lumora_jni
    SHARED
    src/main/cpp/lumora_jni.cpp
)

# Find Android NDK system libraries
find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# Link native libraries
target_link_libraries(
    lumora_jni
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}
)

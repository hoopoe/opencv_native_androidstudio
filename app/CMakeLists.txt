
# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

cmake_minimum_required(VERSION 3.4.1)

set (CMAKE_CXX_STANDARD 11)

# OpenCV stuff
include_directories($ENV{ANDROID_OPENCV_SDK}/jni/include)
add_library( lib_opencv SHARED IMPORTED )
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION $ENV{ANDROID_OPENCV_SDK}/libs/${ANDROID_ABI}/libopencv_java3.so)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds it for you.
# Gradle automatically packages shared libraries with your APK.

set(DLIB_DIR ../dlib)
include_directories(${DLIB_DIR})

add_library( # Sets the name of the library.
             lib_dlib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             # Associated headers in the same location as their source
             # file are automatically included.
             ../dlib/dlib/threads/threads_kernel_shared.cpp
             ../dlib/dlib/entropy_decoder/entropy_decoder_kernel_2.cpp
             ../dlib/dlib/base64/base64_kernel_1.cpp
             ../dlib/dlib/threads/threads_kernel_1.cpp
             ../dlib/dlib/threads/threads_kernel_2.cpp )

add_library( # Sets the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             # Associated headers in the same location as their source
             # file are automatically included.
             src/main/cpp/native-lib.cpp )


# Searches for a specified prebuilt library and stores the path as a
# variable. Because system libraries are included in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in the
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       native-lib

                       lib_dlib

                       # OpenCV lib
                       lib_opencv

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )




add_definitions(-DANDROID)
add_executable(face_landmark_detection_ex ../dlib/examples/face_landmark_detection_ex.cpp)
target_link_libraries( face_landmark_detection_ex lib_dlib )

add_executable(face_detection_ex ../dlib/examples/face_detection_ex.cpp)
target_link_libraries( face_detection_ex lib_dlib )
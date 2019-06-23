
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
elseif(COMPILER_SUPPORTS_CXX0X)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

aux_source_directory(. DIR_SRCS)
add_executable(main ${DIR_SRCS})


set(CMAKE_PREFIX_PATH /home/usamahasan/tracking/C++/libtorch-GPU/libtorch)
find_package(realsense2 REQUIRED)
find_package( OpenCV REQUIRED )
find_package(Torch REQUIRED)
include_directories(include ${realsense_INCLUDE_DIR})
include_directories(include ${OpenCV_INCLUDE_DIR})

target_include_directories(main PRIVATE ${realsense_INCLUDE_DIR})
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main ${realsense2_LIBRARY})
target_link_libraries(main "${TORCH_LIBRARIES}")

message ( STATUS "Pytorch status:" )
message ( STATUS " libraries: ${TORCH_LIBRARIES}" )
message (STATUS "OPENCV:" ${DIR_SRCS})

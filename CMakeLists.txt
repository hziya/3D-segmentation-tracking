cmake_minimum_required(VERSION 3.0)
project(Term_Project)

set(CMAKE_CXX_STANDARD 11)

# Find required packages
find_package(OpenCV 4 REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK 7.1 REQUIRED)

# Set include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    /usr/include/eigen3
    /usr/local/include/pcl-1.13
    /usr/include/vtk-7.1
)

# Set link directories
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add executable target
add_executable(main
    main.cpp
    Term_Project.cpp
)

# Link libraries
target_link_libraries(main
    ${OpenCV_LIBS}
    ${PCL_SEARCH_LIBRARIES}
    ${PCL_COMMON_LIBRARIES}
    ${PCL_IO_LIBRARIES}
    ${PCL_VISUALIZATION_LIBRARIES}
    ${PCL_FILTERS_LIBRARIES}
    ${PCL_SEGMENTATION_LIBRARIES}
    ${VTK_LIBRARIES}
)


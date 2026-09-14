cmake_minimum_required(VERSION 3.16)
project(fliftbot_standalone_cartographer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

find_package(cartographer CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(CAIRO REQUIRED cairo)
find_library(ZIP_LIBRARY zip REQUIRED)

add_executable(fliftbot_cartographer_offline runner.cc)
target_include_directories(fliftbot_cartographer_offline PRIVATE
  ${EIGEN3_INCLUDE_DIRS} ${CAIRO_INCLUDE_DIRS})
target_link_libraries(fliftbot_cartographer_offline PRIVATE
  cartographer ${ZIP_LIBRARY} ${CAIRO_LIBRARIES})
target_compile_options(fliftbot_cartographer_offline PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS fliftbot_cartographer_offline RUNTIME DESTINATION bin)

enable_testing()
add_test(NAME standalone_help COMMAND fliftbot_cartographer_offline --help)
set_tests_properties(standalone_help PROPERTIES WILL_FAIL TRUE)

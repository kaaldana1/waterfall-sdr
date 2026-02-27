cmake_minimum_required(VERSION 3.16)
project(waterfall C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_subdirectory(vendored/SDL EXCLUDE_FROM_ALL)

add_executable(waterfall src/main.c)

# Dependencies  
find_package(Threads REQUIRED)
find_library(RTLSDR_LIB rtlsdr REQUIRED)
find_library(USB_LIB usb-1.0 REQUIRED)

target_link_libraries(waterfall PRIVATE
  SDL3::SDL3
  ${RTLSDR_LIB}
  ${USB_LIB}
  Threads::Threads
  m
)


cmake_minimum_required(VERSION 3.20)

project(Extremz LANGUAGES CXX CUDA C)

include(GNUInstallDirs)
find_package(PkgConfig REQUIRED)
pkg_search_module(ZSTD IMPORTED_TARGET libzstd)

# Enable CUDA support
option(ENABLE_CUDA "Enable CUDA acceleration" OFF)
if (ENABLE_CUDA)
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        set(USE_CUDA TRUE)
        message(STATUS "CUDA support enabled in ExtreMz.")
    else()
        message(WARNING "CUDA not found, disabling CUDA acceleration.")
        set(USE_CUDA FALSE)
    endif()
endif()

# Enable OpenMP support
option(ENABLE_OPENMP "Enable OpenMP acceleration" ON)
if (ENABLE_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        set(USE_OPENMP TRUE)
        message(STATUS "OpenMP support enabled in ExtreMz.")
    else()
        message(WARNING "OpenMP not found, disabling OpenMP acceleration.")
        set(USE_OPENMP FALSE)
    endif()
endif()

set(SRC_FILES
    ./src/ExtreM.cpp
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(ExtreMz STATIC ${SRC_FILES})

target_compile_features(ExtreMz PRIVATE cxx_std_17)

target_include_directories(ExtreMz PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ExtreMz PRIVATE PkgConfig::ZSTD)

# Link OpenMP if enabled
if (USE_OPENMP)
    target_link_libraries(ExtreMz PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(ExtreMz PRIVATE ENABLE_OPENMP)
endif()

# Link CUDA if enabled
if (USE_CUDA)
    target_link_libraries(ExtreMz PRIVATE CUDA::cudart)
    target_compile_definitions(ExtreMz PRIVATE ENABLE_CUDA)
endif()

install(TARGETS ExtreMz
        EXPORT ExtreMzTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}  
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT ExtreMzTargets
        FILE ExtreMzTargets.cmake
        NAMESPACE ExtreMz::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ExtreMz)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ExtreMzConfigVersion.cmake"
    VERSION 1.0
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/ExtreMzConfig.cmake.in" # 位于根目录
    "${CMAKE_CURRENT_BINARY_DIR}/ExtreMzConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ExtreMz
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ExtreMzConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ExtreMzConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ExtreMz)

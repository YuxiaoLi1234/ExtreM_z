#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ExtreMz::ExtreMz" for configuration ""
set_property(TARGET ExtreMz::ExtreMz APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(ExtreMz::ExtreMz PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/libExtreMz.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ExtreMz::ExtreMz )
list(APPEND _IMPORT_CHECK_FILES_FOR_ExtreMz::ExtreMz "${_IMPORT_PREFIX}/lib64/libExtreMz.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

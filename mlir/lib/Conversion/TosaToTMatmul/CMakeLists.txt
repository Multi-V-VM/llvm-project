add_mlir_conversion_library(MLIRTosaToTMatmul
  TosaToTMatmul.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/TosaToTMatmul

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTMatmul
  MLIRTosaDialect
  MLIRTransforms
  )

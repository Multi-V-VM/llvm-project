add_mlir_dialect_library(MLIRTMatmul
  TMatmulDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TMatmul

  DEPENDS
  MLIRTMatmulOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  )

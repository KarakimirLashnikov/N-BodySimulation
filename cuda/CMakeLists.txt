cmake_minimum_required(VERSION 3.18)
project(NBodyCUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect GPU architecture or set manually
# "native" compiles for the GPU on the build machine
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(nbody_cuda
    nbody.cu
)

target_compile_options(nbody_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Print device info at build time
add_custom_target(run_nbody_cuda
    COMMAND nbody_cuda
    DEPENDS nbody_cuda
    COMMENT "Running CUDA N-Body benchmark..."
)

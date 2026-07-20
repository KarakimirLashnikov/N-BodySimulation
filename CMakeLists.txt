cmake_minimum_required(VERSION 4.0)
project(N-BodySimulation LANGUAGES CXX VERSION 1.0.0)

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

# --- Shader compilation -----------------------------------------------------
find_program(GLSLC glslc REQUIRED)

set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

set(SHADERS
    ${SHADER_OUT_DIR}/compute.comp.spv
    ${SHADER_OUT_DIR}/particle.vert.spv
    ${SHADER_OUT_DIR}/particle.frag.spv)

add_custom_command(
    OUTPUT ${SHADER_OUT_DIR}/compute.comp.spv
    COMMAND ${GLSLC} -o ${SHADER_OUT_DIR}/compute.comp.spv ${SHADER_SRC_DIR}/compute.comp
    DEPENDS ${SHADER_SRC_DIR}/compute.comp
    COMMENT "Compiling compute.comp → SPIR-V"
)

add_custom_command(
    OUTPUT ${SHADER_OUT_DIR}/particle.vert.spv
    COMMAND ${GLSLC} -o ${SHADER_OUT_DIR}/particle.vert.spv ${SHADER_SRC_DIR}/particle.vert
    DEPENDS ${SHADER_SRC_DIR}/particle.vert
    COMMENT "Compiling particle.vert → SPIR-V"
)

add_custom_command(
    OUTPUT ${SHADER_OUT_DIR}/particle.frag.spv
    COMMAND ${GLSLC} -o ${SHADER_OUT_DIR}/particle.frag.spv ${SHADER_SRC_DIR}/particle.frag
    DEPENDS ${SHADER_SRC_DIR}/particle.frag
    COMMENT "Compiling particle.frag → SPIR-V"
)

add_custom_target(Shaders ALL DEPENDS ${SHADERS})

# --- Executable -------------------------------------------------------------
add_executable(N-BodySimulation
    main.cpp
    render_device.cpp
    window.cpp
)
add_subdirectory(cuda)

target_compile_features(N-BodySimulation PRIVATE cxx_std_17)

target_link_libraries(
    N-BodySimulation PRIVATE Vulkan::Vulkan glfw
)

add_dependencies(N-BodySimulation Shaders)

# Let the app find SPIR-V files relative to the binary dir
target_compile_definitions(N-BodySimulation PRIVATE
    SHADER_OUT_DIR="${SHADER_OUT_DIR}"
    $<$<CONFIG:Debug>:DEBUG>
)
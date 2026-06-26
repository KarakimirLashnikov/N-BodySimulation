# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
# Configure (from repo root, using vcpkg toolchain)
cmake -B out/build -S . -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake

# Build
cmake --build out/build
```

**Requirements:**
- CMake 4.0+, C++17 compiler
- Vulkan SDK (provides `glslc` for shader compilation and Vulkan headers/libs)
- vcpkg with `glfw3` installed (see `vcpkg.json`)

Shaders are compiled to SPIR-V at build time via `glslc` and output to `${CMAKE_CURRENT_BINARY_DIR}/shaders/`. The `SHADER_OUT_DIR` compile definition tells the app where to find them at runtime.

## Architecture

**Namespace:** `vkrd` (Vulkan Render Device). All Vulkan objects use `vk::raii` (RAII wrappers from Vulkan-Hpp) — destruction order is automatic, no manual `vkDestroy*` calls.

### Threading model — dual-threaded compute + render

The app runs two threads coordinated via a mutex, condition variable, and double-buffering:

1. **Compute thread** (`computeLoop`): Runs the N-body simulation on a dedicated compute queue. Each iteration:
   - Copies particle data from `writeBuf` → `nextBuf` via the transfer queue (so it always starts from the most recent state)
   - Dispatches the compute shader on `nextBuf`
   - Signals the render thread that `nextBuf` is ready

2. **Render thread** (`drawFrame`): Picks up the latest ready buffer (non-blocking — skips the frame if no new data), records graphics commands, submits to the graphics queue, and presents.

### Double-buffering (`kBufCount = 2`)

Two copies of every buffer type (particle SSBO, uniform UBO) enable ping-pong: compute writes to one buffer while render reads from the other. The `readyBuf_`/`renderingBuf_` shared state tracks which buffer is in which role.

### Cross-queue synchronization

Queue families (graphics, compute, transfer) may be distinct on some GPUs. When `needCrossQueueSync_` is true:
- Particle buffers are created with `CONCURRENT` sharing mode so QFO ownership transfers are not needed
- A `renderDoneFences_[buf]` fence is signalled on the graphics queue after rendering completes, so the compute thread can safely overwrite that buffer
- Buffer memory barriers use `QUEUE_FAMILY_IGNORED` (valid with CONCURRENT sharing)
- Pipeline stage flags in barriers fall back to `TOP_OF_PIPE`/`BOTTOM_OF_PIPE` when a dedicated transfer queue lacks compute/graphics stage support

### Compute shader optimization

`shaders/compute.comp` uses **shared-memory tiling**: each workgroup of 256 threads cooperatively loads tiles of particles into shared memory (`shared Particle tile[256]`), reducing global memory reads by 256×. The inner loop is **4-way unrolled** with independent force accumulators (`force0`–`force3`) to break the FMA dependency chain and saturate the GPU's parallel FMA pipelines. At 204,800 particles, this is compute-bound rather than memory-bandwidth-bound.

### Rendering

Uses **dynamic rendering** (`VK_KHR_dynamic_rendering`) — no render pass objects. Particles are drawn as `POINT_LIST` primitives; `gl_VertexIndex` indexes the SSBO directly (no vertex buffers). The vertex shader reads particle positions from the SSBO, applies the MVP matrix via push constants, and sets `gl_PointSize = 2.5`. The fragment shader outputs a fixed warm orange-yellow color.

### File map

| File | Role |
|---|---|
| `main.cpp` | Entry point: GLFW init, creates `Window` + `RenderDevice`, runs main loop |
| `window.hpp/cpp` | GLFW window wrapper — GLFW_NO_API (no OpenGL context), non-resizable |
| `render_device.hpp` | All Vulkan types, structs (`Particle`, `SimParams`, `MVPMatrix`), and method declarations |
| `render_device.cpp` | ~1200 lines: instance/device/swapchain creation, buffer management, compute thread, command recording, sync |
| `shaders/compute.comp` | N-body gravity with tiling + 4-way unrolling (GLSL 450) |
| `shaders/particle.vert` | Point rendering from SSBO via `gl_VertexIndex` (GLSL 450) |
| `shaders/particle.frag` | Fixed-color fragment output (GLSL 450) |

// =============================================================================
// CUDA N-Body Gravity Simulation — benchmark against Vulkan compute version
// =============================================================================
// Matches Vulkan compute.comp optimizations:
//   1. Shared-memory tiling (256 particles per tile, cooperative load)
//   2. 4-way FMA loop unrolling with independent force accumulators
//   3. Identical initial conditions (seed=42, same disk distribution)
//   4. Identical physics (softening², semi-implicit Euler)
//
// Build (standalone):
//   nvcc -O3 -arch=sm_86 -o nbody_cuda cuda/nbody.cu
//
// Or with CMake (see cuda/CMakeLists.txt).
// =============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Constants — must match Vulkan version exactly
// ---------------------------------------------------------------------------
constexpr int    kParticleCount   = 100'000;
constexpr int    kBlockSize       = 256;          // matches local_size_x = 256
constexpr int    kNumBlocks       = (kParticleCount + kBlockSize - 1) / kBlockSize;
constexpr float  kDeltaTime       = 0.001f;       // matches SimParams.deltaTime
constexpr float  kGravityConstant = 1.0f;         // matches SimParams.gravityConstant
constexpr float  kSoftening       = 0.5f;         // matches SimParams.softening
// (benchmark iter counts passed directly to benchmark())

// ---------------------------------------------------------------------------
// Particle struct — must be 32 bytes, matching Vulkan layout
// ---------------------------------------------------------------------------
struct Particle {
    float4 pos;   // x, y, z, mass
    float4 vel;   // x, y, z, padding
};

// ---------------------------------------------------------------------------
// CUDA kernel — mirrors Vulkan compute.comp exactly
// ---------------------------------------------------------------------------
__global__ void nbodyKernel(Particle* particles, int particleCount) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= particleCount) return;

    // Fetch "my" particle once
    Particle p = particles[idx];

    const float softeningSq = kSoftening * kSoftening;
    const float myMass      = p.pos.w;
    const float G           = kGravityConstant;

    // ---- 4-way independent accumulators (matches GLSL force0~force3) ----
    float3 force0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 force1 = make_float3(0.0f, 0.0f, 0.0f);
    float3 force2 = make_float3(0.0f, 0.0f, 0.0f);
    float3 force3 = make_float3(0.0f, 0.0f, 0.0f);

    const float GMi = G * myMass;

    // ---- Shared-memory tile cache (8 KB, matches GLSL) ----
    __shared__ Particle tile[kBlockSize];

    const int numTiles = (particleCount + kBlockSize - 1) / kBlockSize;
    const int lid      = threadIdx.x;

    for (int t = 0; t < numTiles; ++t) {
        // Cooperative tile load
        const int loadIdx = t * kBlockSize + lid;
        if (loadIdx < particleCount) {
            tile[lid] = particles[loadIdx];
        }
        __syncthreads();

        // Interactions within this tile
        const int tileBase  = t * kBlockSize;
        const int tileCount = min(kBlockSize, particleCount - tileBase);

        // 4-way unrolled loop (matches GLSL inner loop exactly)
        int j = 0;
        for (; j + 3 < tileCount; j += 4) {
            const int o0 = tileBase + j;
            const int o1 = o0 + 1;
            const int o2 = o0 + 2;
            const int o3 = o0 + 3;

            // Lane 0
            if (o0 != idx) {
                const Particle t0 = tile[j];
                const float3 d0 = make_float3(
                    t0.pos.x - p.pos.x,
                    t0.pos.y - p.pos.y,
                    t0.pos.z - p.pos.z);
                const float rSq0  = d0.x*d0.x + d0.y*d0.y + d0.z*d0.z + softeningSq;
                const float invR0 = rsqrtf(rSq0);
                const float f0    = GMi * t0.pos.w * invR0 * invR0 * invR0;
                force0.x += f0 * d0.x;
                force0.y += f0 * d0.y;
                force0.z += f0 * d0.z;
            }

            // Lane 1
            if (o1 != idx) {
                const Particle t1 = tile[j + 1];
                const float3 d1 = make_float3(
                    t1.pos.x - p.pos.x,
                    t1.pos.y - p.pos.y,
                    t1.pos.z - p.pos.z);
                const float rSq1  = d1.x*d1.x + d1.y*d1.y + d1.z*d1.z + softeningSq;
                const float invR1 = rsqrtf(rSq1);
                const float f1    = GMi * t1.pos.w * invR1 * invR1 * invR1;
                force1.x += f1 * d1.x;
                force1.y += f1 * d1.y;
                force1.z += f1 * d1.z;
            }

            // Lane 2
            if (o2 != idx) {
                const Particle t2 = tile[j + 2];
                const float3 d2 = make_float3(
                    t2.pos.x - p.pos.x,
                    t2.pos.y - p.pos.y,
                    t2.pos.z - p.pos.z);
                const float rSq2  = d2.x*d2.x + d2.y*d2.y + d2.z*d2.z + softeningSq;
                const float invR2 = rsqrtf(rSq2);
                const float f2    = GMi * t2.pos.w * invR2 * invR2 * invR2;
                force2.x += f2 * d2.x;
                force2.y += f2 * d2.y;
                force2.z += f2 * d2.z;
            }

            // Lane 3
            if (o3 != idx) {
                const Particle t3 = tile[j + 3];
                const float3 d3 = make_float3(
                    t3.pos.x - p.pos.x,
                    t3.pos.y - p.pos.y,
                    t3.pos.z - p.pos.z);
                const float rSq3  = d3.x*d3.x + d3.y*d3.y + d3.z*d3.z + softeningSq;
                const float invR3 = rsqrtf(rSq3);
                const float f3    = GMi * t3.pos.w * invR3 * invR3 * invR3;
                force3.x += f3 * d3.x;
                force3.y += f3 * d3.y;
                force3.z += f3 * d3.z;
            }
        }

        // Remainder (0–3 particles)
        for (; j < tileCount; ++j) {
            const int otherIdx = tileBase + j;
            if (otherIdx == idx) continue;

            const Particle other = tile[j];
            const float3 dir = make_float3(
                other.pos.x - p.pos.x,
                other.pos.y - p.pos.y,
                other.pos.z - p.pos.z);
            const float distSq  = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z + softeningSq;
            const float invDist = rsqrtf(distSq);
            const float invDist3 = invDist * invDist * invDist;

            force0.x += GMi * other.pos.w * invDist3 * dir.x;
            force0.y += GMi * other.pos.w * invDist3 * dir.y;
            force0.z += GMi * other.pos.w * invDist3 * dir.z;
        }

        __syncthreads();
    }

    // Merge 4 accumulators
    float3 force;
    force.x = (force0.x + force1.x) + (force2.x + force3.x);
    force.y = (force0.y + force1.y) + (force2.y + force3.y);
    force.z = (force0.z + force1.z) + (force2.z + force3.z);

    // Semi-implicit Euler
    const float invMass = 1.0f / myMass;
    p.vel.x += force.x * invMass * kDeltaTime;
    p.vel.y += force.y * invMass * kDeltaTime;
    p.vel.z += force.z * invMass * kDeltaTime;
    p.pos.x += p.vel.x * kDeltaTime;
    p.pos.y += p.vel.y * kDeltaTime;
    p.pos.z += p.vel.z * kDeltaTime;

    particles[idx] = p;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
struct Timings {
    float kernelAvgMs;
    float kernelMinMs;
    float kernelMaxMs;
};

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static void initParticles(std::vector<Particle>& host) {
    host.resize(kParticleCount);

    std::mt19937 rng(42); // fixed seed — same as Vulkan version
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.1415926535f);
    std::uniform_real_distribution<float> radiusDist(1.0f, 15.0f);
    std::uniform_real_distribution<float> smallMass(0.1f, 1.0f);

    // Particle 0 — heavy central mass
    host[0] = {
        make_float4(0.0f, 0.0f, 0.0f, 100.0f),
        make_float4(0.0f, 0.0f, 0.0f, 0.0f)
    };

    for (int i = 1; i < kParticleCount; ++i) {
        const float angle  = angleDist(rng);
        const float radius = radiusDist(rng);
        const float mass   = smallMass(rng);

        const float px = std::cos(angle) * radius;
        const float pz = std::sin(angle) * radius;
        const float orbitalSpeed = std::sqrt(100.0f / radius);
        const float vx = -std::sin(angle) * orbitalSpeed;
        const float vz =  std::cos(angle) * orbitalSpeed;

        host[i] = {
            make_float4(px, 0.0f, pz, mass),
            make_float4(vx, 0.0f, vz, 0.0f)
        };
    }
}

static Timings benchmark(Particle* d_particles, int warmupIters, int measureIters) {
    cudaEvent_t startEv, stopEv;
    checkCuda(cudaEventCreate(&startEv), "create start event");
    checkCuda(cudaEventCreate(&stopEv),  "create stop event");

    float kernelMin = 1e9f, kernelMax = 0.0f;
    float kernelSum = 0.0f;

    // Warm-up iterations
    for (int i = 0; i < warmupIters; ++i) {
        nbodyKernel<<<kNumBlocks, kBlockSize>>>(d_particles, kParticleCount);
    }
    checkCuda(cudaDeviceSynchronize(), "sync after warm-up");

    // Measurement iterations
    for (int i = 0; i < measureIters; ++i) {
        checkCuda(cudaEventRecord(startEv), "record start");
        nbodyKernel<<<kNumBlocks, kBlockSize>>>(d_particles, kParticleCount);
        checkCuda(cudaEventRecord(stopEv),  "record stop");
        checkCuda(cudaEventSynchronize(stopEv), "sync event");

        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, startEv, stopEv), "elapsed time");

        kernelSum += ms;
        if (ms < kernelMin) kernelMin = ms;
        if (ms > kernelMax) kernelMax = ms;
    }

    checkCuda(cudaEventDestroy(startEv), "destroy start");
    checkCuda(cudaEventDestroy(stopEv),  "destroy stop");

    return { kernelSum / measureIters, kernelMin, kernelMax };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // --- Device info ---
    int devId = 0;
    cudaDeviceProp props{};
    checkCuda(cudaGetDeviceProperties(&props, devId), "get device props");
    printf("GPU: %s\n", props.name);
    printf("  Compute Capability: %d.%d\n", props.major, props.minor);
    printf("  SMs: %d, Max Threads/SM: %d\n",
           props.multiProcessorCount, props.maxThreadsPerMultiProcessor);
    printf("  Shared Memory/SM: %zu KB, Registers/SM: %d\n",
           props.sharedMemPerMultiprocessor / 1024, props.regsPerMultiprocessor);
    printf("  Max Threads/Block: %d, Max Shared Mem/Block: %zu KB\n",
           props.maxThreadsPerBlock, props.sharedMemPerBlock / 1024);
    printf("\n");

    // --- Allocate device memory ---
    const size_t bufSize = sizeof(Particle) * kParticleCount;
    Particle* d_particles = nullptr;
    checkCuda(cudaMalloc(&d_particles, bufSize), "cudaMalloc particles");

    // --- Init particles on host, copy to device ---
    std::vector<Particle> host;
    initParticles(host);
    checkCuda(cudaMemcpy(d_particles, host.data(), bufSize,
                         cudaMemcpyHostToDevice), "memcpy H→D");

    // --- Benchmark ---
    printf("Configuration:\n");
    printf("  Particles: %d  |  Block Size: %d  |  Blocks: %d\n",
           kParticleCount, kBlockSize, kNumBlocks);
    printf("  Warm-up iterations: 20  |  Measurement iterations: 50\n\n");

    auto t = benchmark(d_particles, 20, 50);

    printf("=== N-Body Kernel Timing (100K particles, O(N²)) ===\n");
    printf("  Average: %8.3f ms\n", t.kernelAvgMs);
    printf("  Min:     %8.3f ms\n", t.kernelMinMs);
    printf("  Max:     %8.3f ms\n", t.kernelMaxMs);
    printf("\n");

    // --- Theoretical comparison ---
    float vulkanMs = 125.13f; // from Nsight Systems
    float speedup  = vulkanMs / t.kernelAvgMs;
    printf("=== Vulkan vs CUDA Comparison ===\n");
    printf("  Vulkan (RTX 3050 Mobile): 125.13 ms\n");
    printf("  CUDA   (%s): %.2f ms\n", props.name, t.kernelAvgMs);
    printf("  Speedup: %.2fx %s\n",
           (speedup >= 1.0f ? speedup : 1.0f / speedup),
           (speedup >= 1.0f ? "(CUDA faster)" : "(Vulkan faster)"));
    printf("\n");

    // --- FLOPS estimate ---
    {
        // 100K × 100K = 1e10 particle pairs
        // ~38 FLOP per pair (3 subtract + 3 mul+add dot + 1 FMA soften + rsqrt + 3 mul invR³ + GMi*mass*invR³ + 3 mul force*dir + 3 add force)
        constexpr double pairs   = 1e10;
        constexpr double flopPer = 38.0;
        double totalFlops = pairs * flopPer;
        double tflops     = totalFlops / (t.kernelAvgMs * 1e-3) / 1e12;
        printf("Estimated Compute Throughput:\n");
        printf("  Total FLOP: %.1f GFLOP  |  Throughput: %.2f TFLOPs\n",
               totalFlops / 1e9, tflops);
    }

    // --- Cleanup ---
    checkCuda(cudaFree(d_particles), "cudaFree");
    return EXIT_SUCCESS;
}

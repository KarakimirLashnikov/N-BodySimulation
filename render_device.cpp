#include "render_device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

// ---------------------------------------------------------------------------
// Anonymous helpers
// ---------------------------------------------------------------------------
namespace {

constexpr bool kEnableValidation = true;
constexpr const char *kAppName = "N-Body Simulation";
constexpr const char *kEngineName = "vkrd";

// Validation layers we want
const std::vector<const char *> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"};

// Device extensions we need
const std::vector<const char *> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME};

#ifndef NDEBUG
constexpr bool kIsDebug = true;
#else
constexpr bool kIsDebug = false;
#endif

// ---------------------------------------------------------------------------
// Simple matrix helpers (column-major, Vulkan-friendly)
// ---------------------------------------------------------------------------

// Orthographic projection, right-handed
std::array<float, 16> ortho(float left, float right, float bottom, float top,
                            float near, float far) {
  std::array<float, 16> m{};
  // Vulkan NDC: X/Y in [-1,1], Z in [0,1]
  m[0] = 2.0f / (right - left);
  m[5] = 2.0f / (top - bottom);
  m[10] = -1.0f / (far - near);
  m[12] = -(right + left) / (right - left);
  m[13] = -(top + bottom) / (top - bottom);
  m[14] = -near / (far - near);
  m[15] = 1.0f;
  return m;
}

// 4x4 matrix multiply : a * b
std::array<float, 16> mul(const std::array<float, 16> &a,
                          const std::array<float, 16> &b) {
  std::array<float, 16> r{};
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      r[col * 4 + row] =
          a[0 * 4 + row] * b[col * 4 + 0] +
          a[1 * 4 + row] * b[col * 4 + 1] +
          a[2 * 4 + row] * b[col * 4 + 2] +
          a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return r;
}

// Look-at (right-handed), produces column-major view matrix
std::array<float, 16> lookAt(float eyeX, float eyeY, float eyeZ,
                             float centerX, float centerY, float centerZ,
                             float upX, float upY, float upZ) {
  float fx = centerX - eyeX;
  float fy = centerY - eyeY;
  float fz = centerZ - eyeZ;
  float fLen = std::sqrt(fx * fx + fy * fy + fz * fz);
  fx /= fLen; fy /= fLen; fz /= fLen;

  float uLen = std::sqrt(upX * upX + upY * upY + upZ * upZ);
  float ux = upX / uLen, uy = upY / uLen, uz = upZ / uLen;

  // s = f x up
  float sx = fy * uz - fz * uy;
  float sy = fz * ux - fx * uz;
  float sz = fx * uy - fy * ux;

  // u' = s x f
  float ux2 = sy * fz - sz * fy;
  float uy2 = sz * fx - sx * fz;
  float uz2 = sx * fy - sy * fx;

  std::array<float, 16> m{};
  m[0] = sx;   m[1] = ux2;  m[2] = -fx;  m[3] = 0;
  m[4] = sy;   m[5] = uy2;  m[6] = -fy;  m[7] = 0;
  m[8] = sz;   m[9] = uz2;  m[10] = -fz; m[11] = 0;
  m[12] = -(sx * eyeX + sy * eyeY + sz * eyeZ);
  m[13] = -(ux2 * eyeX + uy2 * eyeY + uz2 * eyeZ);
  m[14] = (fx * eyeX + fy * eyeY + fz * eyeZ);
  m[15] = 1;
  return m;
}

} // anonymous namespace

// ===================================================================
// RenderDevice
// ===================================================================

namespace vkrd {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
RenderDevice::RenderDevice(Window &window) : window_(window) {}

RenderDevice::~RenderDevice() {
  stopComputeThread();
  if (*logicalDevice_) {
    logicalDevice_.waitIdle();
  }
}

// ---------------------------------------------------------------------------
// initialize – orchestrates all Vulkan setup
// ---------------------------------------------------------------------------
void RenderDevice::initialize() {
  createInstance();
  createSurface();
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapchain();
  createImageViews();
  createDescriptorSetLayout();
  createComputePipeline();
  createGraphicsPipeline();
  createCommandPool();
  createComputeCommandPool();
  createParticleBuffers();
  createUniformBuffers();
  createDescriptorPoolAndSets();
  createCommandBuffers();
  createComputeCommandBuffers();
  createSyncObjects();
  createComputeSyncObjects();
  initParticles();
  startComputeThread();
}

// ===================================================================
// Initialisation steps
// ===================================================================

// --- Instance --------------------------------------------------------------
void RenderDevice::createInstance() {
  if (kEnableValidation && !checkValidationLayerSupport()) {
    throw std::runtime_error("Validation layers requested but not available");
  }

  vk::ApplicationInfo appInfo(kAppName, 1, kEngineName, 1,
                              VK_API_VERSION_1_3);

  auto extensions = getRequiredInstanceExtensions();

  vk::DebugUtilsMessengerCreateInfoEXT debugInfo{};
  if (kEnableValidation) {
    debugInfo.messageSeverity =
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    debugInfo.messageType =
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    debugInfo.pfnUserCallback =
        reinterpret_cast<vk::PFN_DebugUtilsMessengerCallbackEXT>(
            debugCallback);
  }

  vk::InstanceCreateInfo createInfo(
      {}, &appInfo,
      kEnableValidation ? static_cast<std::uint32_t>(kValidationLayers.size())
                        : 0u,
      kEnableValidation ? kValidationLayers.data() : nullptr,
      static_cast<std::uint32_t>(extensions.size()), extensions.data());
  if (kEnableValidation) {
    createInfo.pNext = &debugInfo;
  }

  instance_ = vk::raii::Instance(vk::raii::Context{}, createInfo);

  if (kEnableValidation) {
    debugMessenger_ = instance_.createDebugUtilsMessengerEXT(debugInfo);
  }
}

// --- Surface ---------------------------------------------------------------
void RenderDevice::createSurface() {
  VkSurfaceKHR rawSurface;
  if (glfwCreateWindowSurface(*instance_,
                              window_.getGLFWwindow(), nullptr,
                              &rawSurface) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create window surface");
  }
  surface_ = vk::raii::SurfaceKHR(instance_, rawSurface);
}

// --- Physical device -------------------------------------------------------
void RenderDevice::pickPhysicalDevice() {
  auto devices = instance_.enumeratePhysicalDevices();
  if (devices.empty()) {
    throw std::runtime_error("No Vulkan-capable GPU found");
  }

  vk::raii::PhysicalDevice *bestDevice = nullptr;

  for (auto &dev : devices) {
    auto props = dev.getProperties();

    // Resolve queue families for this candidate device
    auto qFamilies = dev.getQueueFamilyProperties();
    QueueFamilyIndices indices;
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(qFamilies.size());
         ++i) {
      if (qFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) {
        indices.graphics = i;
      }
      if (qFamilies[i].queueFlags & vk::QueueFlagBits::eCompute) {
        indices.compute = i;
      }
      if (dev.getSurfaceSupportKHR(i, *surface_)) {
        indices.present = i;
      }
      if (indices.isComplete()) break;
    }
    if (!indices.isComplete()) continue;

    // Check device extension support
    auto availableExts = dev.enumerateDeviceExtensionProperties();
    std::set<std::string> required(kDeviceExtensions.begin(),
                                   kDeviceExtensions.end());
    for (auto &ext : availableExts) {
      required.erase(static_cast<const char *>(ext.extensionName));
    }
    if (!required.empty()) continue;

    // Check swapchain is usable (use dev directly, not physicalDevice_)
    auto caps = dev.getSurfaceCapabilitiesKHR(*surface_);
    auto formats = dev.getSurfaceFormatsKHR(*surface_);
    auto presentModes = dev.getSurfacePresentModesKHR(*surface_);
    if (formats.empty() || presentModes.empty()) continue;

    // Prefer discrete GPU, accept any
    bestDevice = &dev;
    if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
      break; // take the first discrete GPU
    }
  }

  if (!bestDevice) {
    throw std::runtime_error("No suitable physical device found");
  }
  physicalDevice_ = std::move(*bestDevice);
}

// --- Logical device --------------------------------------------------------
void RenderDevice::createLogicalDevice() {
  // Re-resolve queue families for the chosen physical device
  auto qFamilies = physicalDevice_.getQueueFamilyProperties();
  QueueFamilyIndices indices;
  for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(qFamilies.size());
       ++i) {
    if (qFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) {
      indices.graphics = i;
    }
    if (qFamilies[i].queueFlags & vk::QueueFlagBits::eCompute) {
      indices.compute = i;
    }
    if (physicalDevice_.getSurfaceSupportKHR(i, *surface_)) {
      indices.present = i;
    }
  }
  if (!indices.isComplete()) {
    throw std::runtime_error("Queue family indices incomplete");
  }

  graphicsQueueFamily_ = *indices.graphics;
  computeQueueFamily_ = *indices.compute;

  float queuePriority = 1.0f;
  std::set<std::uint32_t> uniqueFamilies = {graphicsQueueFamily_,
                                            computeQueueFamily_};
  std::vector<vk::DeviceQueueCreateInfo> queueInfos;
  for (auto fam : uniqueFamilies) {
    queueInfos.push_back(
        {{}, fam, 1, &queuePriority});
  }

  vk::PhysicalDeviceFeatures deviceFeatures{};

  // Enable dynamic rendering
  vk::PhysicalDeviceDynamicRenderingFeatures dynamicRendering{true};

  vk::DeviceCreateInfo createInfo({}, queueInfos, {}, kDeviceExtensions,
                                  &deviceFeatures);
  createInfo.pNext = &dynamicRendering;
  logicalDevice_ = physicalDevice_.createDevice(createInfo);

  graphicsQueue_ = logicalDevice_.getQueue(graphicsQueueFamily_, 0);
  computeQueue_ = logicalDevice_.getQueue(computeQueueFamily_, 0);
}

// --- Swapchain -------------------------------------------------------------
void RenderDevice::createSwapchain() {
  auto details = querySwapchainDetails();
  surfaceFormat_ = chooseSurfaceFormat();
  auto presentMode = choosePresentMode();
  surfaceExtent_ = chooseSurfaceExtent();

  std::uint32_t imageCount = details.capabilities.minImageCount + 1;
  if (details.capabilities.maxImageCount > 0 &&
      imageCount > details.capabilities.maxImageCount) {
    imageCount = details.capabilities.maxImageCount;
  }

  vk::SwapchainCreateInfoKHR createInfo(
      {}, *surface_, imageCount, surfaceFormat_.format,
      surfaceFormat_.colorSpace, surfaceExtent_, 1,
      vk::ImageUsageFlagBits::eColorAttachment);

  std::uint32_t families[] = {graphicsQueueFamily_, computeQueueFamily_};
  if (graphicsQueueFamily_ == computeQueueFamily_) {
    createInfo.imageSharingMode = vk::SharingMode::eExclusive;
  } else {
    createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = families;
  }
  createInfo.preTransform = details.capabilities.currentTransform;
  createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;

  swapchain_ = logicalDevice_.createSwapchainKHR(createInfo);
  swapchainImages_ = swapchain_.getImages();
}

void RenderDevice::createImageViews() {
  imageViews_.clear();
  for (auto img : swapchainImages_) {
    vk::ImageViewCreateInfo createInfo(
        {}, img, vk::ImageViewType::e2D, surfaceFormat_.format,
        {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
         vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    imageViews_.push_back(logicalDevice_.createImageView(createInfo));
  }
}

// --- Descriptor set layout -------------------------------------------------
void RenderDevice::createDescriptorSetLayout() {
  // Binding 0 – uniform buffer (SimParams) – used by compute
  vk::DescriptorSetLayoutBinding uboBinding(
      0, vk::DescriptorType::eUniformBuffer, 1,
      vk::ShaderStageFlagBits::eCompute);

  // Binding 1 – storage buffer (particles) – used by compute + vertex
  vk::DescriptorSetLayoutBinding ssboBinding(
      1, vk::DescriptorType::eStorageBuffer, 1,
      vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eVertex);

  vk::DescriptorSetLayoutBinding bindings[] = {uboBinding, ssboBinding};
  descriptorSetLayout_ = logicalDevice_.createDescriptorSetLayout(
      {{}, bindings});
}

// --- Pipelines -------------------------------------------------------------
void RenderDevice::createComputePipeline() {
  auto compShader = createShaderModule(
      std::string(SHADER_OUT_DIR) + "/compute.comp.spv");

  vk::PipelineShaderStageCreateInfo stage(
      {}, vk::ShaderStageFlagBits::eCompute, *compShader, "main");

  computePipelineLayout_ = logicalDevice_.createPipelineLayout(
      {{}, *descriptorSetLayout_});

  computePipeline_ = logicalDevice_.createComputePipeline(
      nullptr, {{}, stage, *computePipelineLayout_});
}

void RenderDevice::createGraphicsPipeline() {
  auto vertShader = createShaderModule(
      std::string(SHADER_OUT_DIR) + "/particle.vert.spv");
  auto fragShader = createShaderModule(
      std::string(SHADER_OUT_DIR) + "/particle.frag.spv");

  vk::PipelineShaderStageCreateInfo vertStage(
      {}, vk::ShaderStageFlagBits::eVertex, *vertShader, "main");
  vk::PipelineShaderStageCreateInfo fragStage(
      {}, vk::ShaderStageFlagBits::eFragment, *fragShader, "main");
  vk::PipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

  // Push constant range for MVP matrix
  vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eVertex, 0,
                                  sizeof(MVPMatrix));

  graphicsPipelineLayout_ = logicalDevice_.createPipelineLayout(
      {{}, *descriptorSetLayout_, pushRange});

  // Vertex input – we use gl_VertexIndex, no vertex attributes
  vk::PipelineVertexInputStateCreateInfo vertexInput({}, {}, {});

  vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
      {}, vk::PrimitiveTopology::ePointList);

  // Dynamic states – viewport & scissor are set per-frame
  std::vector<vk::DynamicState> dynamicStates = {
      vk::DynamicState::eViewport, vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);

  // Use empty viewport/scissor counts since they're dynamic
  vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1,
                                                     nullptr);

  vk::PipelineRasterizationStateCreateInfo rasterizer(
      {}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill,
      vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise,
      VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f);

  vk::PipelineMultisampleStateCreateInfo multisample;

  vk::PipelineColorBlendAttachmentState blendAttachment;
  blendAttachment.colorWriteMask =
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

  vk::PipelineColorBlendStateCreateInfo colorBlend({}, VK_FALSE,
                                                   vk::LogicOp::eCopy,
                                                   blendAttachment);

  // Dynamic rendering – specify colour attachment format
  vk::PipelineRenderingCreateInfo renderingInfo({}, 1, &surfaceFormat_.format);

  vk::GraphicsPipelineCreateInfo pipelineInfo(
      {}, stages, &vertexInput, &inputAssembly, nullptr, &viewportState,
      &rasterizer, &multisample, nullptr, &colorBlend, &dynamicState,
      *graphicsPipelineLayout_, nullptr);
  pipelineInfo.pNext = &renderingInfo;

  graphicsPipeline_ =
      logicalDevice_.createGraphicsPipeline(nullptr, pipelineInfo);
}

// --- Command pool ----------------------------------------------------------
void RenderDevice::createCommandPool() {
  commandPool_ = logicalDevice_.createCommandPool(
      {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
       graphicsQueueFamily_});
}

void RenderDevice::createComputeCommandPool() {
  // Use graphics queue family so that pipeline stages (e.g. VERTEX_INPUT)
  // are compatible with compute command buffers when needed for barriers.
  computeCommandPool_ = logicalDevice_.createCommandPool(
      {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
       graphicsQueueFamily_});
}

// --- Buffers ---------------------------------------------------------------
void RenderDevice::createParticleBuffers() {
  vk::DeviceSize size = sizeof(Particle) * kParticleCount;
  for (uint32_t i = 0; i < kBufCount; ++i) {
    createBuffer(size,
                 vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eVertexBuffer |
                     vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eDeviceLocal,
                 particleBuffers_[i], particleBufferMemories_[i]);
  }
}

void RenderDevice::createUniformBuffers() {
  vk::DeviceSize size = sizeof(SimParams);
  for (uint32_t i = 0; i < kBufCount; ++i) {
    createBuffer(size, vk::BufferUsageFlagBits::eUniformBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible |
                     vk::MemoryPropertyFlagBits::eHostCoherent,
                 uniformBuffers_[i], uniformBufferMemories_[i]);
    mappedUniforms_[i] = static_cast<SimParams *>(
        uniformBufferMemories_[i].mapMemory(0, size));
  }
}

void RenderDevice::initParticles() {
  std::vector<Particle> particles(kParticleCount);
  std::mt19937 rng(42); // fixed seed for reproducibility
  std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * 3.1415926535f);
  std::uniform_real_distribution<float> radiusDist(1.0f, 15.0f);
  std::uniform_real_distribution<float> smallMass(0.1f, 1.0f);

  // Particle 0 – heavy central mass
  particles[0] = {0.0f, 0.0f, 0.0f, 100.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // Remaining particles in a disk with near-orbital velocities
  for (std::uint32_t i = 1; i < kParticleCount; ++i) {
    float angle = angleDist(rng);
    float radius = radiusDist(rng);
    float mass = smallMass(rng);

    float px = std::cos(angle) * radius;
    float pz = std::sin(angle) * radius;

    // Circular orbit velocity: v = sqrt(G * M / r)
    // Central mass = 100, G = 1.0 → v ≈ sqrt(100/r)
    float orbitalSpeed = std::sqrt(100.0f / radius);
    // Tangential direction (perpendicular to radius in XZ)
    float vx = -std::sin(angle) * orbitalSpeed;
    float vz = std::cos(angle) * orbitalSpeed;

    particles[i] = {px, 0.0f, pz, mass, vx, 0.0f, vz, 0.0f};
  }

  // Upload via staging buffer
  vk::DeviceSize size = sizeof(Particle) * kParticleCount;
  vk::raii::Buffer stagingBuf{nullptr};
  vk::raii::DeviceMemory stagingMem{nullptr};
  createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible |
                   vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuf, stagingMem);

  void *data = stagingMem.mapMemory(0, size);
  std::memcpy(data, particles.data(), static_cast<std::size_t>(size));
  stagingMem.unmapMemory();

  // Copy staging → device-local (to both buffers)
  auto cmdBuf = std::move(
      logicalDevice_
          .allocateCommandBuffers(
              {*commandPool_, vk::CommandBufferLevel::ePrimary, 1})
          .front());
  cmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  for (uint32_t i = 0; i < kBufCount; ++i) {
    vk::BufferCopy region(0, 0, size);
    cmdBuf.copyBuffer(*stagingBuf, *particleBuffers_[i], region);
  }
  cmdBuf.end();

  vk::SubmitInfo submit({}, {}, *cmdBuf);
  graphicsQueue_.submit(submit);
  graphicsQueue_.waitIdle();
}

// --- Descriptor pool & sets ------------------------------------------------
void RenderDevice::createDescriptorPoolAndSets() {
  std::vector<vk::DescriptorPoolSize> poolSizes = {
      {vk::DescriptorType::eUniformBuffer, kBufCount},
      {vk::DescriptorType::eStorageBuffer, kBufCount}};

  descriptorPool_ = logicalDevice_.createDescriptorPool(
      {vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, kBufCount,
       poolSizes});

  std::array<vk::DescriptorSetLayout, kBufCount> layouts;
  layouts.fill(*descriptorSetLayout_);
  auto sets = logicalDevice_.allocateDescriptorSets(
      {*descriptorPool_, layouts});
  for (uint32_t i = 0; i < kBufCount; ++i)
    descriptorSets_[i] = std::move(sets[i]);

  // Write descriptors: each set binds its own flight buffers
  for (uint32_t i = 0; i < kBufCount; ++i) {
    vk::DescriptorBufferInfo uboInfo(*uniformBuffers_[i], 0,
                                     sizeof(SimParams));
    vk::DescriptorBufferInfo ssboInfo(*particleBuffers_[i], 0,
                                      sizeof(Particle) * kParticleCount);
    vk::WriteDescriptorSet writes[] = {
        {*descriptorSets_[i], 0, 0, vk::DescriptorType::eUniformBuffer, {},
         uboInfo},
        {*descriptorSets_[i], 1, 0, vk::DescriptorType::eStorageBuffer, {},
         ssboInfo},
    };
    logicalDevice_.updateDescriptorSets(writes, nullptr);
  }
}

// --- Command buffers (render thread) ---------------------------------------
void RenderDevice::createCommandBuffers() {
  commandBuffers_ = logicalDevice_.allocateCommandBuffers(
      {*commandPool_, vk::CommandBufferLevel::ePrimary,
       static_cast<std::uint32_t>(swapchainImages_.size())});
}

void RenderDevice::createComputeCommandBuffers() {
  auto bufs = logicalDevice_.allocateCommandBuffers(
      {*computeCommandPool_, vk::CommandBufferLevel::ePrimary, kBufCount});
  for (uint32_t i = 0; i < kBufCount; ++i)
    computeCommandBuffers_[i] = std::move(bufs[i]);
}

// --- Sync objects ----------------------------------------------------------
void RenderDevice::createSyncObjects() {
  std::uint32_t count = static_cast<std::uint32_t>(swapchainImages_.size());
  for (std::uint32_t i = 0; i < count; ++i) {
    inFlightFences_.push_back(
        logicalDevice_.createFence({vk::FenceCreateFlagBits::eSignaled}));
    imageAvailableSemaphores_.push_back(logicalDevice_.createSemaphore({}));
    renderFinishedSemaphores_.push_back(logicalDevice_.createSemaphore({}));
  }
}

void RenderDevice::createComputeSyncObjects() {
  for (uint32_t i = 0; i < kBufCount; ++i) {
    computeFences_[i] =
        logicalDevice_.createFence({vk::FenceCreateFlagBits::eSignaled});
  }
}

// ===================================================================
// Compute thread
// ===================================================================

void RenderDevice::startComputeThread() {
  // --- Bootstrapping: initialise buffer 0 with first compute pass ---
  logicalDevice_.resetFences(*computeFences_[0]);
  {
    auto &cb = computeCommandBuffers_[0];
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    recordComputeCommands(*cb, 0);
    cb.end();

    vk::SubmitInfo submit({}, {}, *cb);
    graphicsQueue_.submit(submit, *computeFences_[0]);
  }
  // Wait for first result, then notify render thread
  (void)logicalDevice_.waitForFences(*computeFences_[0], VK_TRUE,
                                     UINT64_MAX);
  logicalDevice_.resetFences(*computeFences_[0]);
  {
    std::lock_guard<std::mutex> lk(sharedMtx_);
    readyBuf_ = 0;
  }
  sharedCv_.notify_one();

  // Launch background loop
  computeThread_ = std::thread(&RenderDevice::computeLoop, this);
}

void RenderDevice::stopComputeThread() {
  {
    std::lock_guard<std::mutex> lk(sharedMtx_);
    stopCompute_ = true;
  }
  sharedCv_.notify_all();
  if (computeThread_.joinable())
    computeThread_.join();
}

void RenderDevice::computeLoop() {
  constexpr float dt = 0.001f;
  int writeBuf = 0; // buffer that just received new compute results

  while (true) {
    // ---- Switch to the *other* buffer ----
    int nextBuf = 1 - writeBuf;

    // Wait until render is done with nextBuf (so we can safely overwrite it)
    {
      std::unique_lock<std::mutex> lk(sharedMtx_);
      sharedCv_.wait(lk, [&] {
        return stopCompute_ || renderingBuf_ != nextBuf;
      });
      if (stopCompute_)
        break;
    }

    // ---- Copy latest particle data from writeBuf → nextBuf ----
    {
      auto &cb = computeCommandBuffers_[nextBuf];
      cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
      recordCopyCommands(*cb, writeBuf, nextBuf);
      cb.end();

      vk::SubmitInfo submit({}, {}, *cb);
      {
        std::lock_guard<std::mutex> lk(queueMtx_);
        graphicsQueue_.submit(submit);
        graphicsQueue_.waitIdle(); // copy is trivial, quick wait
      }
    }

    // ---- Update uniform & dispatch compute on nextBuf ----
    {
      auto *u = mappedUniforms_[nextBuf];
      u->deltaTime = dt;
      u->particleCount = kParticleCount;
      u->gravityConstant = 1.0f;
      u->softening = 0.5f;

      logicalDevice_.resetFences(*computeFences_[nextBuf]);

      auto &cb = computeCommandBuffers_[nextBuf];
      cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
      recordComputeCommands(*cb, nextBuf);
      cb.end();

      vk::SubmitInfo submit({}, {}, *cb);
      {
        std::lock_guard<std::mutex> lk(queueMtx_);
        graphicsQueue_.submit(submit, *computeFences_[nextBuf]);
      }
    }

    // Wait for compute to finish
    (void)logicalDevice_.waitForFences(*computeFences_[nextBuf], VK_TRUE,
                                       UINT64_MAX);
    logicalDevice_.resetFences(*computeFences_[nextBuf]);

    // ---- Notify render: nextBuf is ready ----
    {
      std::lock_guard<std::mutex> lk(sharedMtx_);
      readyBuf_ = nextBuf;
    }
    sharedCv_.notify_one();

    writeBuf = nextBuf;
  }
}

void RenderDevice::recordCopyCommands(vk::CommandBuffer cb, int srcIdx,
                                      int dstIdx) {
  vk::DeviceSize size = sizeof(Particle) * kParticleCount;

  // Barrier: make compute writes available for transfer read
  vk::BufferMemoryBarrier preCopy(
      vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
      *particleBuffers_[srcIdx], 0, size);
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eTransfer, {}, {}, preCopy,
                     {});

  vk::BufferCopy region(0, 0, size);
  cb.copyBuffer(*particleBuffers_[srcIdx], *particleBuffers_[dstIdx], region);

  // Barrier: make transfer writes available for compute shader
  vk::BufferMemoryBarrier postCopy(
      vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
      *particleBuffers_[dstIdx], 0, size);
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                     vk::PipelineStageFlagBits::eComputeShader, {}, {},
                     postCopy, {});
}

void RenderDevice::recordComputeCommands(vk::CommandBuffer cb, int bufIdx) {
  vk::DeviceSize size = sizeof(Particle) * kParticleCount;

  // Barrier: ensure transfer (or previous graphics) is done with this buffer
  vk::BufferMemoryBarrier preBarrier(
      vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eVertexAttributeRead,
      vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
      *particleBuffers_[bufIdx], 0, size);
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer |
                         vk::PipelineStageFlagBits::eVertexInput,
                     vk::PipelineStageFlagBits::eComputeShader, {}, {},
                     preBarrier, {});

  cb.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline_);
  cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                        *computePipelineLayout_, 0,
                        *descriptorSets_[bufIdx], {});

  std::uint32_t groupCount = (kParticleCount + 255) / 256;
  cb.dispatch(groupCount, 1, 1);

  // Barrier: make compute writes available for transfer / vertex
  vk::BufferMemoryBarrier postBarrier(
      vk::AccessFlagBits::eShaderWrite,
      vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eVertexAttributeRead,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
      *particleBuffers_[bufIdx], 0, size);
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                     vk::PipelineStageFlagBits::eTransfer |
                         vk::PipelineStageFlagBits::eVertexInput,
                     {}, {}, postBarrier, {});
}

// ===================================================================
// Render thread (drawFrame)
// ===================================================================
void RenderDevice::drawFrame() {
  // ---- Grab the latest ready buffer (non-blocking) ----
  int renderBuf;
  {
    std::lock_guard<std::mutex> lk(sharedMtx_);
    if (readyBuf_ >= 0) {
      renderingBuf_ = readyBuf_;
      readyBuf_ = -1;
    }
    renderBuf = renderingBuf_;
  }
  if (renderBuf < 0) {
    // First data not ready yet; skip this frame
    ++currentFrame_;
    return;
  }

  std::uint32_t frameIdx = currentFrame_ % swapchainImages_.size();

  // Wait for fence of the current frame slot
  (void)logicalDevice_.waitForFences(*inFlightFences_[frameIdx], VK_TRUE,
                                     UINT64_MAX);
  logicalDevice_.resetFences(*inFlightFences_[frameIdx]);

  // Acquire swapchain image
  auto [result, imageIndex] = swapchain_.acquireNextImage(
      UINT64_MAX, *imageAvailableSemaphores_[frameIdx]);
  if (result == vk::Result::eErrorOutOfDateKHR) {
    ++currentFrame_;
    return;
  }
  if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
    throw std::runtime_error("Failed to acquire swapchain image");
  }
  currentImageIndex_ = imageIndex;

  // Record render commands using the chosen particle buffer
  auto &cb = commandBuffers_[frameIdx];
  cb.begin({});
  recordGraphicsCommands(*cb, renderBuf);
  cb.end();

  // Submit
  vk::PipelineStageFlags waitStage =
      vk::PipelineStageFlagBits::eColorAttachmentOutput;
  vk::SubmitInfo submit(*imageAvailableSemaphores_[frameIdx], waitStage, *cb,
                        *renderFinishedSemaphores_[imageIndex]);
  {
    std::lock_guard<std::mutex> lk(queueMtx_);
    graphicsQueue_.submit(submit, *inFlightFences_[frameIdx]);
  }

  // Present
  vk::PresentInfoKHR present(*renderFinishedSemaphores_[imageIndex],
                             *swapchain_, imageIndex);
  {
    std::lock_guard<std::mutex> lk(queueMtx_);
    result = graphicsQueue_.presentKHR(present);
  }
  if (result == vk::Result::eErrorOutOfDateKHR) {
    // TODO: recreate swapchain
  }

  // Mark rendering done so compute thread can reuse this buffer
  {
    std::lock_guard<std::mutex> lk(sharedMtx_);
    renderingBuf_ = -1;
  }
  sharedCv_.notify_one();

  ++currentFrame_;
}

void RenderDevice::recordGraphicsCommands(vk::CommandBuffer cb, int bufIdx) {
  // Compute MVP matrix – orbiting camera
  static float time = 0.0f;
  time += 0.001f;
  float camX = std::sin(time * 0.1f) * 25.0f;
  float camZ = std::cos(time * 0.1f) * 25.0f;
  auto view = lookAt(camX, 18.0f, camZ, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

  float aspect = static_cast<float>(surfaceExtent_.width) /
                 static_cast<float>(surfaceExtent_.height);
  auto proj = ortho(-20.0f, 20.0f, -20.0f / aspect, 20.0f / aspect, 0.1f,
                    200.0f);

  MVPMatrix mvp;
  mvp.mvp = mul(proj, view);

  // Transition swapchain image to colour attachment layout
  vk::ImageMemoryBarrier toColor(
      {}, vk::AccessFlagBits::eColorAttachmentWrite,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::eColorAttachmentOptimal, VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED, swapchainImages_[currentImageIndex_],
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                     vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, {},
                     {}, toColor);

  // Dynamic rendering begin
  vk::RenderingAttachmentInfo colorAttach(
      *imageViews_[currentImageIndex_],
      vk::ImageLayout::eColorAttachmentOptimal, vk::ResolveModeFlagBits::eNone,
      {}, vk::ImageLayout::eUndefined, vk::AttachmentLoadOp::eClear,
      vk::AttachmentStoreOp::eStore,
      vk::ClearValue(
          vk::ClearColorValue(std::array{0.02f, 0.02f, 0.05f, 1.0f})));

  vk::RenderingInfo renderingInfo({}, {{0, 0}, surfaceExtent_}, 1, 0,
                                  colorAttach);
  cb.beginRendering(renderingInfo);

  cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline_);
  cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                        *graphicsPipelineLayout_, 0,
                        *descriptorSets_[bufIdx], {});
  cb.pushConstants<MVPMatrix>(*graphicsPipelineLayout_,
                              vk::ShaderStageFlagBits::eVertex, 0, mvp);

  vk::Viewport viewport(0.0f, 0.0f, static_cast<float>(surfaceExtent_.width),
                        static_cast<float>(surfaceExtent_.height), 0.0f, 1.0f);
  vk::Rect2D scissor({0, 0}, surfaceExtent_);
  cb.setViewport(0, viewport);
  cb.setScissor(0, scissor);

  cb.draw(kParticleCount, 1, 0, 0);
  cb.endRendering();

  // Transition to present layout
  vk::ImageMemoryBarrier toPresent(
      vk::AccessFlagBits::eColorAttachmentWrite, {},
      vk::ImageLayout::eColorAttachmentOptimal,
      vk::ImageLayout::ePresentSrcKHR, VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED, swapchainImages_[currentImageIndex_],
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                     vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {},
                     toPresent);
}

// ===================================================================
// Utilities
// ===================================================================
void RenderDevice::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                vk::MemoryPropertyFlags properties,
                                vk::raii::Buffer &buffer,
                                vk::raii::DeviceMemory &memory) {
  buffer = logicalDevice_.createBuffer({{}, size, usage});
  auto memReqs = buffer.getMemoryRequirements();
  std::uint32_t memType =
      findMemoryType(memReqs.memoryTypeBits, properties);
  memory = logicalDevice_.allocateMemory(
      {memReqs.size, memType});
  buffer.bindMemory(*memory, 0);
}

std::uint32_t RenderDevice::findMemoryType(
    std::uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
  auto memProps = physicalDevice_.getMemoryProperties();
  for (std::uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((typeFilter & (1u << i)) &&
        (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("No suitable memory type found");
}

vk::raii::ShaderModule
RenderDevice::createShaderModule(const std::string &filepath) const {
  std::ifstream file(filepath, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open shader file: " + filepath);
  }
  std::size_t size = static_cast<std::size_t>(file.tellg());
  std::vector<char> code(size);
  file.seekg(0);
  file.read(code.data(), static_cast<std::streamsize>(size));
  return logicalDevice_.createShaderModule(
      {{}, size, reinterpret_cast<const std::uint32_t *>(code.data())});
}

RenderDevice::SwapchainDetails RenderDevice::querySwapchainDetails() const {
  SwapchainDetails details;
  details.capabilities =
      physicalDevice_.getSurfaceCapabilitiesKHR(*surface_);
  details.formats = physicalDevice_.getSurfaceFormatsKHR(*surface_);
  details.presentModes = physicalDevice_.getSurfacePresentModesKHR(*surface_);
  return details;
}

vk::SurfaceFormatKHR RenderDevice::chooseSurfaceFormat() const {
  auto details = querySwapchainDetails();
  for (auto &fmt : details.formats) {
    if (fmt.format == vk::Format::eB8G8R8A8Srgb &&
        fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      return fmt;
    }
  }
  return details.formats[0];
}

vk::PresentModeKHR RenderDevice::choosePresentMode() const {
  auto details = querySwapchainDetails();
  // Prefer mailbox
  for (auto &mode : details.presentModes) {
    if (mode == vk::PresentModeKHR::eMailbox) return mode;
  }
  return vk::PresentModeKHR::eFifo; // guaranteed
}

vk::Extent2D RenderDevice::chooseSurfaceExtent() const {
  auto caps = physicalDevice_.getSurfaceCapabilitiesKHR(*surface_);
  if (caps.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
    return caps.currentExtent;
  }
  auto [w, h] = window_.getFramebufferSize();
  return {std::clamp(static_cast<std::uint32_t>(w), caps.minImageExtent.width,
                     caps.maxImageExtent.width),
          std::clamp(static_cast<std::uint32_t>(h), caps.minImageExtent.height,
                     caps.maxImageExtent.height)};
}

std::vector<const char *> RenderDevice::getRequiredInstanceExtensions() const {
  std::uint32_t glfwCount = 0;
  const char **glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
  std::vector<const char *> extensions(glfwExts, glfwExts + glfwCount);
  if (kEnableValidation) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  return extensions;
}

bool RenderDevice::checkValidationLayerSupport() const {
  auto layers = vk::raii::Context{}.enumerateInstanceLayerProperties();
  for (const char *name : kValidationLayers) {
    bool found = false;
    for (auto &layer : layers) {
      if (std::strcmp(layer.layerName, name) == 0) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

VKAPI_ATTR VkBool32 VKAPI_CALL RenderDevice::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT *data, void * /*user*/) {
  if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    fprintf(stderr, "[Vulkan] %s\n", data->pMessage);
  }
  return VK_FALSE;
}

} // namespace vkrd

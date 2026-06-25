#ifndef RENDER_DEVICE_HPP
#define RENDER_DEVICE_HPP

#include "window.hpp"

#include <vulkan/vulkan_raii.hpp>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace vkrd {

// ---------------------------------------------------------------------------
// Simulation data structures (std140 / std430 compatible)
// ---------------------------------------------------------------------------
struct Particle {
  float posX, posY, posZ, mass; // position (xyz) + mass
  float velX, velY, velZ, pad;  // velocity (xyz) + padding
};
static_assert(sizeof(Particle) == 32, "Particle must be 32 bytes");

struct SimParams {
  float deltaTime;
  std::uint32_t particleCount;
  float gravityConstant;
  float softening;
};
static_assert(sizeof(SimParams) == 16, "SimParams must be 16 bytes");

struct MVPMatrix {
  std::array<float, 16> mvp; // model-view-projection
};

// ---------------------------------------------------------------------------
// Render device – owns all Vulkan resources
// ---------------------------------------------------------------------------
class RenderDevice {
public:
  RenderDevice(Window &window);
  ~RenderDevice();

  RenderDevice(const RenderDevice &) = delete;
  RenderDevice &operator=(const RenderDevice &) = delete;

  void initialize();
  void drawFrame();

private:
  // --- Helper types --------------------------------------------------------
  struct SwapchainDetails {
    vk::SurfaceCapabilitiesKHR capabilities{};
    std::vector<vk::SurfaceFormatKHR> formats{};
    std::vector<vk::PresentModeKHR> presentModes{};
  };

  struct QueueFamilyIndices {
    std::optional<std::uint32_t> graphics{};
    std::optional<std::uint32_t> compute{};
    std::optional<std::uint32_t> present{};

    bool isComplete() const {
      return graphics.has_value() && compute.has_value() && present.has_value();
    }
  };

  // --- Vulkan resource handles (RAII) --------------------------------------
  Window &window_;
  vk::raii::Instance instance_{nullptr};
  vk::raii::SurfaceKHR surface_{nullptr};
  vk::raii::DebugUtilsMessengerEXT debugMessenger_{nullptr};
  vk::raii::PhysicalDevice physicalDevice_{nullptr};
  vk::raii::Device logicalDevice_{nullptr};

  // Queues
  vk::raii::Queue graphicsQueue_{nullptr}; // graphics + present
  vk::raii::Queue computeQueue_{nullptr};  // compute (may be same family)
  std::uint32_t graphicsQueueFamily_{};
  std::uint32_t computeQueueFamily_{};

  // Swapchain
  vk::raii::SwapchainKHR swapchain_{nullptr};
  std::vector<vk::Image> swapchainImages_{};
  vk::SurfaceFormatKHR surfaceFormat_{};
  vk::Extent2D surfaceExtent_{};
  std::vector<vk::raii::ImageView> imageViews_;

  // Descriptor set layouts
  vk::raii::DescriptorSetLayout descriptorSetLayout_{nullptr};

  // Pipelines
  vk::raii::PipelineLayout computePipelineLayout_{nullptr};
  vk::raii::Pipeline computePipeline_{nullptr};
  vk::raii::PipelineLayout graphicsPipelineLayout_{nullptr};
  vk::raii::Pipeline graphicsPipeline_{nullptr};

  // Command
  vk::raii::CommandPool commandPool_{nullptr};
  std::vector<vk::raii::CommandBuffer> commandBuffers_; // per-frame

  // Buffers
  vk::raii::Buffer particleBuffer_{nullptr};
  vk::raii::DeviceMemory particleBufferMemory_{nullptr};
  vk::raii::Buffer uniformBuffer_{nullptr};
  vk::raii::DeviceMemory uniformBufferMemory_{nullptr};
  SimParams *mappedUniform_{nullptr};

  // Descriptor pool & sets
  vk::raii::DescriptorPool descriptorPool_{nullptr};
  std::vector<vk::raii::DescriptorSet> descriptorSets_;

  // Synchronisation (per swapchain image)
  std::vector<vk::raii::Fence> inFlightFences_;
  std::vector<vk::raii::Semaphore> imageAvailableSemaphores_;
  std::vector<vk::raii::Semaphore> renderFinishedSemaphores_;

  // Constants
  static constexpr std::uint32_t kParticleCount = 204800;
  static constexpr std::uint32_t kMaxFramesInFlight = 1;
  std::uint32_t currentFrame_{0};
  std::uint32_t currentImageIndex_{0};

  // --- Initialisation steps ------------------------------------------------
  void createInstance();
  void createSurface();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createSwapchain();
  void createImageViews();
  void createDescriptorSetLayout();
  void createComputePipeline();
  void createGraphicsPipeline();
  void createCommandPool();
  void createParticleBuffer();
  void createUniformBuffer();
  void createDescriptorPoolAndSets();
  void createCommandBuffers();
  void createSyncObjects();
  void initParticles();

  // --- Per-frame helpers ---------------------------------------------------
  void updateUniformBuffer(float dt);
  void recordComputeCommands(vk::CommandBuffer cb);
  void recordGraphicsCommands(vk::CommandBuffer cb);

  // --- Utility -------------------------------------------------------------
  void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties,
                    vk::raii::Buffer &buffer,
                    vk::raii::DeviceMemory &memory);
  std::uint32_t findMemoryType(std::uint32_t typeFilter,
                               vk::MemoryPropertyFlags properties) const;
  vk::raii::ShaderModule createShaderModule(const std::string &filepath) const;
  SwapchainDetails querySwapchainDetails() const;
  QueueFamilyIndices findQueueFamilies() const;
  vk::SurfaceFormatKHR chooseSurfaceFormat() const;
  vk::PresentModeKHR choosePresentMode() const;
  vk::Extent2D chooseSurfaceExtent() const;
  std::vector<const char *> getRequiredInstanceExtensions() const;
  bool checkValidationLayerSupport() const;
  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                VkDebugUtilsMessageTypeFlagsEXT type,
                const VkDebugUtilsMessengerCallbackDataEXT *data, void *user);
};

} // namespace vkrd

#endif // RENDER_DEVICE_HPP
#include "render_device.hpp"
#include "window.hpp"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

int main() {
  // --- Initialise GLFW ---------------------------------------------------
  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialise GLFW\n");
    return EXIT_FAILURE;
  }
  // Ensure cleanup on exit
  struct GLFWGuard {
    ~GLFWGuard() { glfwTerminate(); }
  } glfwGuard;

  try {
    // --- Create window ---------------------------------------------------
    vkrd::Window window(1280, 720, "N-Body Gravity Simulation (Vulkan)");

    // --- Create and initialise render device -----------------------------
    vkrd::RenderDevice renderer(window);
    renderer.initialize();

    // --- Main loop -------------------------------------------------------
    while (!window.shouldClose()) {
      glfwPollEvents();
      renderer.drawFrame();
    }

  } catch (const std::exception &e) {
    fprintf(stderr, "Fatal error: %s\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

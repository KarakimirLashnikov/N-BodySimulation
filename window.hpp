#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <memory>
#include <string>

namespace vkrd {

class Window {
public:
  Window(int width, int height, const char *title);
  ~Window();

  Window(const Window &) = delete;
  Window &operator=(const Window &) = delete;
  Window(Window &&) = delete;
  Window &operator=(Window &&) = delete;

  GLFWwindow *getGLFWwindow() const;
  bool shouldClose() const;
  std::pair<int, int> getFramebufferSize() const;

private:
  struct GLFWwindowDeleter {
    void operator()(GLFWwindow *window) const {
      if (window) {
        glfwDestroyWindow(window);
      }
    }
  };
  std::unique_ptr<GLFWwindow, GLFWwindowDeleter> window_;
};

} // namespace vkrd

#endif // WINDOW_HPP
#include "window.hpp"

#include <stdexcept>

namespace vkrd {

Window::Window(int width, int height, const char *title) {
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  GLFWwindow *raw = glfwCreateWindow(width, height, title, nullptr, nullptr);
  if (!raw) {
    throw std::runtime_error("Failed to create GLFW window");
  }
  window_.reset(raw);
}

Window::~Window() = default;

GLFWwindow *Window::getGLFWwindow() const { return window_.get(); }

bool Window::shouldClose() const {
  return glfwWindowShouldClose(window_.get());
}

std::pair<int, int> Window::getFramebufferSize() const {
  int w = 0, h = 0;
  glfwGetFramebufferSize(window_.get(), &w, &h);
  return {w, h};
}

} // namespace vkrd

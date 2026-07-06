#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include "dashboard.h"

#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstring>

static void glfw_error_callback(int error, const char* desc) {
    std::fprintf(stderr, "GLFW error %d: %s\n", error, desc);
}

int main(int, char**) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) return 1;

    // OpenGL 3.3 core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        1600, 900,
        "Llama.cpp Benchmark Dashboard",
        nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // ImGui init
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext(); 
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();
    

    // Sizing
    style.FramePadding      = ImVec2(8, 6);   // button/input inner padding
    style.ItemSpacing       = ImVec2(8, 6);   // space between widgets
    style.WindowPadding     = ImVec2(10, 10);
    style.ScrollbarSize     = 14.0f;
    style.WindowRounding    = 4.0f;
    style.FrameRounding     = 4.0f;           // rounded button corners

    // Colors — indexed by ImGuiCol_*
    style.Colors[ImGuiCol_Button]        = ImVec4(0.20f, 0.45f, 0.20f, 1.00f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.30f, 0.60f, 0.30f, 1.00f);
    style.Colors[ImGuiCol_ButtonActive]  = ImVec4(0.15f, 0.35f, 0.15f, 1.00f);

    
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Dark theme with slight customization
    
    ImGui::StyleColorsClassic();
    
    style.WindowRounding    = 4.0f;
    style.FrameRounding     = 3.0f;
    style.ScrollbarRounding = 3.0f;
    style.GrabRounding      = 3.0f;
    style.TabRounding       = 3.0f;
    style.WindowPadding     = {10.0f, 10.0f};
    style.FramePadding      = {6.0f, 3.0f};
    style.ItemSpacing       = {8.0f, 5.0f};

    // Accent colors: slightly blue-tinted
    // style.Colors[ImGuiCol_TitleBgActive]  = {0.12f, 0.22f, 0.35f, 1.0f};
    // style.Colors[ImGuiCol_Tab]            = {0.10f, 0.18f, 0.28f, 1.0f};
    // style.Colors[ImGuiCol_TabActive]      = {0.18f, 0.35f, 0.55f, 1.0f};
    // style.Colors[ImGuiCol_TabHovered]     = {0.22f, 0.42f, 0.65f, 1.0f};
    // style.Colors[ImGuiCol_Header]         = {0.15f, 0.28f, 0.45f, 1.0f};
    // style.Colors[ImGuiCol_HeaderHovered]  = {0.20f, 0.38f, 0.58f, 1.0f};
    // style.Colors[ImGuiCol_HeaderActive]   = {0.25f, 0.45f, 0.68f, 1.0f};
    // style.Colors[ImGuiCol_Button]         = {0.15f, 0.28f, 0.45f, 1.0f};
    // style.Colors[ImGuiCol_ButtonHovered]  = {0.22f, 0.42f, 0.65f, 1.0f};
    // style.Colors[ImGuiCol_ButtonActive]   = {0.28f, 0.52f, 0.78f, 1.0f};
    // style.Colors[ImGuiCol_FrameBg]        = {0.10f, 0.10f, 0.14f, 1.0f};
    // style.Colors[ImGuiCol_FrameBgHovered] = {0.14f, 0.20f, 0.30f, 1.0f};
    // style.Colors[ImGuiCol_CheckMark]      = {0.30f, 0.80f, 0.40f, 1.0f};
    // style.Colors[ImGuiCol_SliderGrab]     = {0.25f, 0.55f, 0.80f, 1.0f};

    // Load default font slightly larger
    io.Fonts->AddFontDefault();
    ImFontConfig font_cfg;
    font_cfg.SizePixels = 15.0f;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Init dashboard
    Dashboard dashboard;
    if (!dashboard.init()) {
        std::fprintf(stderr, "Dashboard init failed\n");
        return 1;
    }

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Handle window resize
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Set display size for layout calculations
        ImGui::GetIO().DisplaySize =
            ImVec2((float)display_w, (float)display_h);

        dashboard.render();

        ImGui::Render();
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    dashboard.shutdown();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
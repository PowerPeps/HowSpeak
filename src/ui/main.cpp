#include "syntax/Conllu.hpp"
#include "transfer/Pipeline.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace howspeak;

struct Choice {
    std::string code;
    std::string label;
};

struct Result {
    std::string output;
    std::string error;
};

Result run(const transfer::Pipeline* pipeline, std::string text, std::string from, std::string to,
           transfer::Engine engine, bool trace) {
    Result result;
    try {
        std::ostringstream out;
        for (const auto& t : pipeline->transfer(pipeline->parse(text, from), from, to, engine)) {
            if (trace) {
                syntax::writeConllu(out, t.sentence);
            }
            out << t.calque << '\n';
        }
        result.output = out.str();
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}

std::vector<Choice> choices(const transfer::Pipeline& pipeline, const transfer::Paths& paths, bool sources) {
    std::vector<Choice> out;
    for (const auto& l : pipeline.database().languages()) {
        const bool keep = sources ? !l.iso.empty() && std::filesystem::exists(paths.udpipe(l.iso)) : l.sample200;
        if (keep) {
            out.push_back({l.id, std::format("{} ({})", l.name, l.id)});
        }
    }
    std::ranges::sort(out, {}, &Choice::label);
    return out;
}

int indexOf(const std::vector<Choice>& items, std::string_view code) {
    const auto it = std::ranges::find(items, code, &Choice::code);
    return it == items.end() ? 0 : static_cast<int>(it - items.begin());
}

void combo(const char* label, const std::vector<Choice>& items, int& current) {
    if (items.empty()) {
        ImGui::TextDisabled("%s: -", label);
        return;
    }
    if (ImGui::BeginCombo(label, items[static_cast<std::size_t>(current)].label.c_str(), ImGuiComboFlags_HeightLarge)) {
        for (std::size_t i = 0; i < items.size(); ++i) {
            const bool selected = static_cast<int>(i) == current;
            if (ImGui::Selectable(items[i].label.c_str(), selected)) {
                current = static_cast<int>(i);
            }
            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

void loadFonts(ImGuiIO& io) {
#ifdef _MSC_VER
#pragma warning(suppress : 4996)
#endif
    const char* windir = std::getenv("WINDIR");
    const std::filesystem::path fonts = std::filesystem::path(windir != nullptr ? windir : "C:/Windows") / "Fonts";
    if (!std::filesystem::exists(fonts / "segoeui.ttf")) {
        return;
    }
    io.Fonts->AddFontFromFileTTF((fonts / "segoeui.ttf").string().c_str(), 18.0F);
    for (const char* fallback : {"YuGothM.ttc", "msgothic.ttc", "malgun.ttf"}) {
        if (std::filesystem::exists(fonts / fallback)) {
            ImFontConfig merge;
            merge.MergeMode = true;
            io.Fonts->AddFontFromFileTTF((fonts / fallback).string().c_str(), 18.0F, &merge);
            break;
        }
    }
}

}

int main() {
    const transfer::Paths paths{HOWSPEAK_ROOT};
    std::unique_ptr<transfer::Pipeline> pipeline;
    std::string error;
    std::vector<Choice> sources;
    std::vector<Choice> targets;
    try {
        pipeline = std::make_unique<transfer::Pipeline>(paths);
        sources = choices(*pipeline, paths, true);
        targets = choices(*pipeline, paths, false);
    } catch (const std::exception& e) {
        error = e.what();
    }

    if (glfwInit() == GLFW_FALSE) {
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(960, 640, "HowSpeak", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();
    loadFonts(io);
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    int source = indexOf(sources, "fre");
    int target = indexOf(targets, "jpn");
    int engine = 0;
    bool trace = false;
    std::string input = "Il était une fois un petit chat qui ne mangeait pas les souris.";
    std::string output;
    std::future<Result> job;

    while (glfwWindowShouldClose(window) == GLFW_FALSE) {
        glfwWaitEventsTimeout(job.valid() ? 0.05 : 0.5);
        if (job.valid() && job.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            auto result = job.get();
            output = std::move(result.output);
            error = std::move(result.error);
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::Begin("HowSpeak", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove);

        ImGui::SetNextItemWidth(260.0F);
        combo("Source", sources, source);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(260.0F);
        combo("Cible", targets, target);
        ImGui::SameLine();
        ImGui::RadioButton("Neuronal", &engine, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Règles", &engine, 1);
        ImGui::SameLine();
        ImGui::Checkbox("Arbre", &trace);

        const float half = (ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing() * 2.0F) / 2.0F;
        ImGui::InputTextMultiline("##input", &input, ImVec2(-1.0F, half));

        const bool ready = pipeline && !job.valid() && !sources.empty() && !targets.empty();
        ImGui::BeginDisabled(!ready);
        const bool submit = ImGui::Button(job.valid() ? "Analyse..." : "Transformer  (Ctrl+Entrée)") ||
                            (ImGui::IsKeyDown(ImGuiMod_Ctrl) && ImGui::IsKeyPressed(ImGuiKey_Enter, false));
        ImGui::EndDisabled();
        if (submit && ready) {
            job = std::async(std::launch::async, run, pipeline.get(), input,
                             sources[static_cast<std::size_t>(source)].code,
                             targets[static_cast<std::size_t>(target)].code,
                             engine == 0 ? transfer::Engine::Neural : transfer::Engine::Rules, trace);
        }
        if (!error.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0F, 0.4F, 0.4F, 1.0F), "%s", error.c_str());
        }

        ImGui::InputTextMultiline("##output", &output, ImVec2(-1.0F, -1.0F), ImGuiInputTextFlags_ReadOnly);
        ImGui::End();

        ImGui::Render();
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.1F, 0.1F, 0.12F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (job.valid()) {
        job.wait();
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

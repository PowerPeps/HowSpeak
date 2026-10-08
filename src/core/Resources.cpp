#include "core/Resources.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace howspeak::core {

namespace {

bool hasData(const std::filesystem::path& root) {
    std::error_code ec;
    return std::filesystem::exists(root / "data" / "wals" / "values.csv", ec);
}

}

std::filesystem::path executablePath() {
#if defined(_WIN32)
    std::wstring buffer(1024, L'\0');
    while (true) {
        const DWORD size = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (size == 0) {
            return {};
        }
        if (size < buffer.size()) {
            buffer.resize(size);
            return std::filesystem::path(buffer);
        }
        buffer.resize(buffer.size() * 2);
    }
#elif defined(__APPLE__)
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size + 1, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return {};
    }
    std::error_code ec;
    const auto path = std::filesystem::weakly_canonical(std::filesystem::path(buffer.data()), ec);
    return ec ? std::filesystem::path(buffer.data()) : path;
#else
    std::error_code ec;
    const auto path = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::filesystem::path() : path;
#endif
}

std::filesystem::path projectRoot() {
#ifdef _MSC_VER
#pragma warning(suppress : 4996)
#endif
    if (const char* env = std::getenv("HOWSPEAK_ROOT"); env != nullptr && hasData(env)) {
        return env;
    }
    const auto exe = executablePath();
    if (!exe.empty()) {
        const auto dir = exe.parent_path();
        for (const auto& candidate : std::array{dir, dir.parent_path(), dir.parent_path() / "Resources"}) {
            if (hasData(candidate)) {
                return candidate;
            }
        }
    }
    return HOWSPEAK_ROOT;
}

}

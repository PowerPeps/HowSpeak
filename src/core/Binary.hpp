#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

namespace howspeak::core {

class BinaryWriter {
public:
    BinaryWriter(const std::filesystem::path& path, std::string_view magic, std::uint32_t version);

    template <class T>
        requires std::is_trivially_copyable_v<T>
    void write(const T& value) {
        raw(&value, sizeof(T));
    }

    template <class T>
        requires std::is_trivially_copyable_v<T>
    void writeArray(std::span<const T> values) {
        write<std::uint64_t>(values.size());
        raw(values.data(), values.size_bytes());
    }

    void writeString(std::string_view text);
    void close();

private:
    void raw(const void* data, std::size_t size);

    std::filesystem::path path_;
    std::ofstream out_;
};

class BinaryReader {
public:
    BinaryReader(const std::filesystem::path& path, std::string_view magic, std::uint32_t version);

    template <class T>
        requires std::is_trivially_copyable_v<T>
    T read() {
        T value{};
        raw(&value, sizeof(T));
        return value;
    }

    template <class T>
        requires std::is_trivially_copyable_v<T>
    void readArray(std::span<T> values) {
        const auto count = read<std::uint64_t>();
        if (count != values.size()) {
            fail("array size mismatch");
        }
        raw(values.data(), values.size_bytes());
    }

    std::string readString();

    [[noreturn]] void fail(std::string_view what) const;

private:
    void raw(void* data, std::size_t size);

    std::filesystem::path path_;
    std::ifstream in_;
};

}

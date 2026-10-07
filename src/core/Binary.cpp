#include "core/Binary.hpp"

#include <format>
#include <stdexcept>

namespace howspeak::core {

BinaryWriter::BinaryWriter(const std::filesystem::path& path, std::string_view magic, std::uint32_t version)
    : path_(path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    out_.open(path, std::ios::binary | std::ios::trunc);
    if (!out_) {
        throw std::runtime_error(std::format("cannot write '{}'", path.string()));
    }
    writeString(magic);
    write(version);
}

void BinaryWriter::writeString(std::string_view text) {
    write<std::uint64_t>(text.size());
    raw(text.data(), text.size());
}

void BinaryWriter::close() {
    out_.close();
    if (!out_) {
        throw std::runtime_error(std::format("cannot finish '{}'", path_.string()));
    }
}

void BinaryWriter::raw(const void* data, std::size_t size) {
    out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!out_) {
        throw std::runtime_error(std::format("write error on '{}'", path_.string()));
    }
}

BinaryReader::BinaryReader(const std::filesystem::path& path, std::string_view magic, std::uint32_t version)
    : path_(path), in_(path, std::ios::binary) {
    if (!in_) {
        throw std::runtime_error(std::format("cannot open '{}'", path.string()));
    }
    if (readString() != magic) {
        fail("bad magic");
    }
    if (read<std::uint32_t>() != version) {
        fail("unsupported version");
    }
}

std::string BinaryReader::readString() {
    const auto size = read<std::uint64_t>();
    if (size > (1U << 20U)) {
        fail("string too long");
    }
    std::string text(size, '\0');
    raw(text.data(), text.size());
    return text;
}

void BinaryReader::fail(std::string_view what) const {
    throw std::runtime_error(std::format("'{}': {}", path_.string(), what));
}

void BinaryReader::raw(void* data, std::size_t size) {
    in_.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!in_) {
        fail("truncated file");
    }
}

}

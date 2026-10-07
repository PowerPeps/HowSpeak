#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace howspeak::core {

class CsvTable {
public:
    static CsvTable load(const std::filesystem::path& path);
    static CsvTable parse(std::string_view text, std::string_view origin);

    [[nodiscard]] std::size_t rows() const noexcept { return rows_.size(); }
    [[nodiscard]] std::size_t column(std::string_view name) const;
    [[nodiscard]] const std::string& at(std::size_t row, std::size_t col) const;

private:
    std::string origin_;
    std::vector<std::string> header_;
    std::vector<std::vector<std::string>> rows_;
};

std::string readFile(const std::filesystem::path& path);

}

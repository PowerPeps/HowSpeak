#include "core/Csv.hpp"

#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace howspeak::core {

std::string readFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(std::format("cannot open '{}'", path.string()));
    }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

CsvTable CsvTable::load(const std::filesystem::path& path) {
    return parse(readFile(path), path.string());
}

CsvTable CsvTable::parse(std::string_view text, std::string_view origin) {
    if (text.starts_with("\xEF\xBB\xBF")) {
        text.remove_prefix(3);
    }

    std::vector<std::vector<std::string>> records;
    std::vector<std::string> record;
    std::string field;
    bool quoted = false;
    bool fieldStarted = false;
    std::size_t line = 1;

    const auto endField = [&] {
        record.push_back(std::move(field));
        field.clear();
        fieldStarted = false;
    };
    const auto endRecord = [&] {
        endField();
        records.push_back(std::move(record));
        record.clear();
    };

    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (quoted) {
            if (c == '"') {
                if (i + 1 < text.size() && text[i + 1] == '"') {
                    field.push_back('"');
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                if (c == '\n') {
                    ++line;
                }
                field.push_back(c);
            }
            continue;
        }
        switch (c) {
        case '"':
            if (fieldStarted) {
                throw std::runtime_error(std::format("{}:{}: stray quote", origin, line));
            }
            quoted = true;
            fieldStarted = true;
            break;
        case ',':
            endField();
            break;
        case '\r':
            break;
        case '\n':
            endRecord();
            ++line;
            break;
        default:
            field.push_back(c);
            fieldStarted = true;
            break;
        }
    }
    if (quoted) {
        throw std::runtime_error(std::format("{}: unterminated quoted field", origin));
    }
    if (fieldStarted || !record.empty()) {
        endRecord();
    }
    if (records.empty()) {
        throw std::runtime_error(std::format("{}: empty table", origin));
    }

    CsvTable table;
    table.origin_ = origin;
    table.header_ = std::move(records.front());
    for (std::size_t r = 1; r < records.size(); ++r) {
        if (records[r].size() == 1 && records[r].front().empty()) {
            continue;
        }
        if (records[r].size() != table.header_.size()) {
            throw std::runtime_error(std::format("{}: record {} has {} fields, expected {}", origin, r,
                                                 records[r].size(), table.header_.size()));
        }
        table.rows_.push_back(std::move(records[r]));
    }
    return table;
}

std::size_t CsvTable::column(std::string_view name) const {
    for (std::size_t i = 0; i < header_.size(); ++i) {
        if (header_[i] == name) {
            return i;
        }
    }
    throw std::runtime_error(std::format("{}: missing column '{}'", origin_, name));
}

const std::string& CsvTable::at(std::size_t row, std::size_t col) const {
    return rows_.at(row).at(col);
}

}

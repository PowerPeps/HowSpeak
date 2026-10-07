#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace howspeak::core::utf8 {

std::u32string decode(std::string_view text);
std::string encode(std::u32string_view text);

char32_t toLower(char32_t c) noexcept;
char32_t toUpper(char32_t c) noexcept;
bool isUpper(char32_t c) noexcept;

std::string lower(std::string_view text);
std::string capitalize(std::string_view text);
bool startsUpper(std::string_view text);
std::size_t length(std::string_view text);

std::optional<std::string> toLatin1(std::string_view text);
std::string fromLatin1(std::string_view text);

}

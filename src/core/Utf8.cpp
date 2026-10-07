#include "core/Utf8.hpp"

namespace howspeak::core::utf8 {

namespace {

constexpr char32_t kReplacement = 0xFFFD;

bool pairedLatinExtendedA(char32_t c) noexcept {
    return (c >= 0x100 && c <= 0x12F) || (c >= 0x132 && c <= 0x137) || (c >= 0x14A && c <= 0x177);
}

bool oddPairedLatinExtendedA(char32_t c) noexcept {
    return (c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17E);
}

}

std::u32string decode(std::string_view text) {
    std::u32string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        const auto b0 = static_cast<unsigned char>(text[i]);
        std::size_t extra = 0;
        char32_t cp = 0;
        if (b0 < 0x80) {
            cp = b0;
        } else if ((b0 & 0xE0U) == 0xC0U) {
            cp = b0 & 0x1FU;
            extra = 1;
        } else if ((b0 & 0xF0U) == 0xE0U) {
            cp = b0 & 0x0FU;
            extra = 2;
        } else if ((b0 & 0xF8U) == 0xF0U) {
            cp = b0 & 0x07U;
            extra = 3;
        } else {
            out.push_back(kReplacement);
            ++i;
            continue;
        }
        if (i + extra >= text.size()) {
            out.push_back(kReplacement);
            break;
        }
        bool valid = true;
        for (std::size_t k = 1; k <= extra; ++k) {
            const auto b = static_cast<unsigned char>(text[i + k]);
            if ((b & 0xC0U) != 0x80U) {
                valid = false;
                break;
            }
            cp = (cp << 6U) | (b & 0x3FU);
        }
        if (!valid) {
            out.push_back(kReplacement);
            ++i;
            continue;
        }
        out.push_back(cp);
        i += extra + 1;
    }
    return out;
}

std::string encode(std::u32string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char32_t cp : text) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0U | (cp >> 6U)));
            out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0U | (cp >> 12U)));
            out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        } else {
            out.push_back(static_cast<char>(0xF0U | (cp >> 18U)));
            out.push_back(static_cast<char>(0x80U | ((cp >> 12U) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | ((cp >> 6U) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        }
    }
    return out;
}

// Covers ASCII, Latin-1, Latin Extended-A, Greek and Cyrillic; other scripts are returned unchanged.
char32_t toLower(char32_t c) noexcept {
    if ((c >= U'A' && c <= U'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7) || (c >= 0x391 && c <= 0x3A9) ||
        (c >= 0x410 && c <= 0x42F)) {
        return c + 0x20;
    }
    if (c >= 0x400 && c <= 0x40F) {
        return c + 0x50;
    }
    if (pairedLatinExtendedA(c) && c % 2 == 0) {
        return c + 1;
    }
    if (oddPairedLatinExtendedA(c) && c % 2 == 1) {
        return c + 1;
    }
    if (c == 0x178) {
        return 0xFF;
    }
    return c;
}

char32_t toUpper(char32_t c) noexcept {
    if ((c >= U'a' && c <= U'z') || (c >= 0xE0 && c <= 0xFE && c != 0xF7) || (c >= 0x3B1 && c <= 0x3C9 && c != 0x3C2) ||
        (c >= 0x430 && c <= 0x44F)) {
        return c - 0x20;
    }
    if (c >= 0x450 && c <= 0x45F) {
        return c - 0x50;
    }
    if (pairedLatinExtendedA(c) && c % 2 == 1) {
        return c - 1;
    }
    if (oddPairedLatinExtendedA(c) && c % 2 == 0) {
        return c - 1;
    }
    if (c == 0xFF) {
        return 0x178;
    }
    return c;
}

bool isUpper(char32_t c) noexcept {
    return toLower(c) != c;
}

std::string lower(std::string_view text) {
    std::u32string cps = decode(text);
    for (char32_t& c : cps) {
        c = toLower(c);
    }
    return encode(cps);
}

std::string capitalize(std::string_view text) {
    std::u32string cps = decode(text);
    if (!cps.empty()) {
        cps.front() = toUpper(cps.front());
    }
    return encode(cps);
}

bool startsUpper(std::string_view text) {
    const std::u32string cps = decode(text);
    return !cps.empty() && isUpper(cps.front());
}

std::size_t length(std::string_view text) {
    return decode(text).size();
}

std::optional<std::string> toLatin1(std::string_view text) {
    std::string out;
    for (const char32_t cp : decode(text)) {
        if (cp > 0xFF) {
            return std::nullopt;
        }
        out.push_back(static_cast<char>(static_cast<unsigned char>(cp)));
    }
    return out;
}

std::string fromLatin1(std::string_view text) {
    std::u32string cps;
    cps.reserve(text.size());
    for (const char c : text) {
        cps.push_back(static_cast<unsigned char>(c));
    }
    return encode(cps);
}

}

#pragma once

#include "transfer/Groups.hpp"

#include <cstddef>
#include <span>

namespace howspeak::transfer {

// Purely structural description of a group member: no lexical information, so the
// network transfers to any source language with a dependency parse.
class Encoder {
public:
    [[nodiscard]] static std::size_t width();
    static void encode(const Group& group, const Item& item, std::span<float> out);
};

}

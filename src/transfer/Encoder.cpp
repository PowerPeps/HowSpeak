#include "transfer/Encoder.hpp"

#include <algorithm>
#include <array>
#include <string_view>

namespace howspeak::transfer {

namespace {

constexpr std::array<std::string_view, 38> kRelations = {
    "HEAD",  "nsubj",  "obj",   "iobj",     "csubj",      "ccomp", "xcomp", "obl",      "vocative", "expl",
    "dislocated", "advcl", "advmod", "discourse", "aux",   "cop",   "mark",  "nmod",     "appos",    "nummod",
    "acl",   "amod",   "det",   "clf",      "case",       "conj",  "cc",    "fixed",    "flat",     "compound",
    "list",  "parataxis", "orphan", "goeswith", "reparandum", "punct", "dep",  "root",
};
constexpr std::array<std::string_view, 6> kSubtypes = {":relcl", ":poss", ":tmod", ":prt", ":pass", ":q"};
constexpr std::array<std::string_view, 17> kUpos = {
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
};
constexpr std::size_t kFlags = 16;
constexpr int kOffsetRange = 8;  // signed offsets to the head beyond +-8 share the edge bins
constexpr std::size_t kPosition = 3 + (2 * kOffsetRange + 1) + 5;

template <std::size_t N>
std::size_t lookup(const std::array<std::string_view, N>& table, std::string_view value, std::size_t fallback) {
    const auto it = std::ranges::find(table, value);
    return it == table.end() ? fallback : static_cast<std::size_t>(it - table.begin());
}

class Writer {
public:
    explicit Writer(std::span<float> out) : out_(out) { std::ranges::fill(out_, 0.0F); }

    void oneHot(std::size_t size, std::size_t index) {
        out_[offset_ + index] = 1.0F;
        offset_ += size;
    }
    void flag(bool value) { out_[offset_++] = value ? 1.0F : 0.0F; }
    void scalar(double value) { out_[offset_++] = static_cast<float>(value); }

private:
    std::span<float> out_;
    std::size_t offset_ = 0;
};

}

std::size_t Encoder::width() {
    return kRelations.size() + kSubtypes.size() + 2 * (kUpos.size() + 1) + kRelations.size() + kFlags + kPosition;
}

void Encoder::encode(const Group& group, const Item& item, std::span<float> out) {
    Writer w(out);
    const auto dep = lookup(kRelations, "dep", 0);
    w.oneHot(kRelations.size(), lookup(kRelations, item.baseRelation, dep));
    for (const auto subtype : kSubtypes) {
        w.flag(item.relation.ends_with(subtype));
    }
    w.oneHot(kUpos.size() + 1, lookup(kUpos, item.upos, kUpos.size()));
    w.oneHot(kUpos.size() + 1, lookup(kUpos, group.headUpos, kUpos.size()));
    w.oneHot(kRelations.size(), lookup(kRelations, group.headRelation, dep));

    w.flag(item.article);
    w.flag(item.definite);
    w.flag(item.indefinite);
    w.flag(item.demonstrative);
    w.flag(item.possessive);
    w.flag(item.interrogative);
    w.flag(item.relativePronoun);
    w.flag(item.personalPronoun);
    w.flag(item.negative);
    w.flag(item.lastNegative);
    w.flag(item.marker);
    w.flag(group.headNominal);
    w.flag(group.headAdjectival);
    w.flag(group.polarQuestion);
    w.flag(group.negatives > 1);
    w.flag(std::ranges::any_of(group.items, [](const Item& i) {
        return i.baseRelation == "aux" || i.baseRelation == "cop";
    }));

    const int offset = item.side * (item.isHead ? 0 : item.proximity + 1);
    w.oneHot(3, static_cast<std::size_t>(item.side + 1));
    w.oneHot(2 * kOffsetRange + 1, static_cast<std::size_t>(std::clamp(offset, -kOffsetRange, kOffsetRange) + kOffsetRange));
    w.scalar(static_cast<double>(offset) / 8.0);
    w.scalar(static_cast<double>(item.proximity) / 8.0);
    w.scalar(item.rank);
    w.scalar(static_cast<double>(group.items.size()) / 10.0);
    w.scalar(static_cast<double>(group.negatives) / 2.0);
}

}

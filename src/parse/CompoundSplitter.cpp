#include "parse/CompoundSplitter.hpp"

#include "core/Utf8.hpp"

#include <hunspell/hunspell.hxx>

#include <array>
#include <format>
#include <limits>
#include <optional>
#include <stdexcept>

namespace howspeak::parse {

namespace {

constexpr std::size_t kMinPart = 3;
constexpr std::size_t kMinWord = 6;
constexpr std::size_t kHomographMin = 5;
// Linking elements are tried first so that "Orts|teil" yields "Ort", not the genitive "Orts".
const std::array<std::u32string, 7> kLinkers = {U"es", U"en", U"er", U"s", U"n", U"e", U""};

std::u32string lowerAll(std::u32string s) {
    for (auto& c : s) {
        c = core::utf8::toLower(c);
    }
    return s;
}

std::u32string capital(std::u32string s) {
    s = lowerAll(std::move(s));
    if (!s.empty()) {
        s.front() = core::utf8::toUpper(s.front());
    }
    return s;
}

}

CompoundSplitter::CompoundSplitter(const std::filesystem::path& aff, const std::filesystem::path& dic,
                                   NounOracle oracle)
    : oracle_(std::move(oracle)) {
    if (!std::filesystem::exists(aff) || !std::filesystem::exists(dic)) {
        throw std::runtime_error(std::format("missing Hunspell dictionary '{}'", dic.string()));
    }
    hunspell_ = std::make_unique<Hunspell>(aff.string().c_str(), dic.string().c_str());
    const auto& encoding = hunspell_->get_dict_encoding();
    if (encoding == "ISO8859-1") {
        latin1_ = true;
    } else if (encoding != "UTF-8") {
        throw std::runtime_error(std::format("unsupported Hunspell encoding '{}'", encoding));
    }
}

CompoundSplitter::~CompoundSplitter() = default;

bool CompoundSplitter::spell(const std::u32string& word) const {
    const std::string utf8 = core::utf8::encode(word);
    if (!latin1_) {
        return hunspell_->spell(utf8);
    }
    const auto latin = core::utf8::toLatin1(utf8);
    return latin && hunspell_->spell(*latin);
}

bool CompoundSplitter::capitalisedEntry(const std::u32string& word) const {
    std::string text = core::utf8::encode(word);
    if (latin1_) {
        const auto latin = core::utf8::toLatin1(text);
        if (!latin) {
            return false;
        }
        text = *latin;
    }
    for (const auto& stem : hunspell_->stem(text)) {
        if (core::utf8::startsUpper(latin1_ ? core::utf8::fromLatin1(stem) : stem)) {
            return true;
        }
    }
    return false;
}

// A part is a noun when its dictionary lemma is capitalised (German-style orthography), otherwise when the
// oracle tags it NOUN; without oracle, short parts must also be rejected in lower case so that function
// words such as "bei" are not taken for nouns.
bool CompoundSplitter::isNoun(const std::u32string& part) const {
    const auto cached = nouns_.find(part);
    if (cached != nouns_.end()) {
        return cached->second;
    }
    const std::u32string cap = capital(part);
    bool noun = false;
    if (spell(cap) || spell(lowerAll(part))) {
        if (capitalisedEntry(cap)) {
            noun = true;
        } else if (oracle_) {
            noun = oracle_(core::utf8::encode(lowerAll(part)));
        } else {
            noun = spell(cap) && (part.size() >= kHomographMin || !spell(lowerAll(part)));
        }
    }
    nouns_.emplace(part, noun);
    return noun;
}

std::vector<std::string> CompoundSplitter::split(std::string_view word) const {
    const std::u32string w = core::utf8::decode(word);
    const std::size_t n = w.size();
    if (n < kMinWord) {
        return {};
    }

    // best[i]: most parts covering w[i..n), every part being a noun (nested compounds are flattened).
    constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> best(n + 1, kNone);
    std::vector<std::size_t> next(n + 1, 0);
    std::vector<std::size_t> linkLength(n + 1, 0);
    best[n] = 0;
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t j = i + kMinPart; j <= n; ++j) {
            if (best[j] == kNone || (i == 0 && j == n)) {
                continue;
            }
            const bool last = j == n;
            for (const auto& linker : kLinkers) {
                if (last && !linker.empty()) {
                    continue;
                }
                if (j - i < kMinPart + linker.size()) {
                    continue;
                }
                const std::u32string segment = w.substr(i, j - i);
                if (!linker.empty() && !segment.ends_with(linker)) {
                    continue;
                }
                const std::u32string stem = segment.substr(0, segment.size() - linker.size());
                if (!isNoun(stem)) {
                    continue;
                }
                if (best[i] == kNone || best[j] + 1 > best[i]) {
                    best[i] = best[j] + 1;
                    next[i] = j;
                    linkLength[i] = linker.size();
                }
                break;
            }
        }
    }
    if (best[0] == kNone || best[0] < 2) {
        return {};
    }

    std::vector<std::string> parts;
    for (std::size_t i = 0; i < n; i = next[i]) {
        std::u32string part = w.substr(i, next[i] - i - linkLength[i]);
        if (i != 0) {
            part = capital(part);
        }
        parts.push_back(core::utf8::encode(part));
    }
    return parts;
}

void CompoundSplitter::apply(syntax::Sentence& sentence) const {
    for (std::size_t i = 0; i < sentence.words.size(); ++i) {
        if (sentence.words[i].upos != "NOUN" || sentence.words[i].marker) {
            continue;
        }
        const auto parts = split(sentence.words[i].form);
        if (parts.size() < 2) {
            continue;
        }
        const auto lemmaParts = split(sentence.words[i].lemma);
        const bool lemmasAlign = lemmaParts.size() == parts.size();
        const double baseOrder = sentence.words[i].order;
        const auto count = parts.size();

        sentence.words[i].form = parts.back();
        sentence.words[i].lemma = lemmasAlign ? lemmaParts.back() : parts.back();
        for (std::size_t k = 0; k + 1 < count; ++k) {
            syntax::Word modifier;
            modifier.form = parts[k];
            modifier.lemma = lemmasAlign ? lemmaParts[k] : parts[k];
            modifier.upos = "NOUN";
            modifier.feats.emplace("Number", "Sing");
            modifier.deprel = "compound";
            modifier.order = baseOrder - 0.5 + 0.5 * static_cast<double>(k) / static_cast<double>(count);
            static_cast<void>(sentence.insert(i + k, std::move(modifier)));
        }
        const std::size_t headPosition = i + count - 1;
        for (std::size_t k = 0; k + 1 < count; ++k) {
            sentence.words[i + k].head = static_cast<int>(headPosition);
        }
        i = headPosition;
    }
}

}
